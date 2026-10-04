"""
Stage A of the pipeline experiments (GPU). For whole nuScenes test scenes,
runs the detector on every keyframe, clean and under each attack (accuracy
PGD: linf/l2/l1; sponge: inflate, sponge_act, uap -- see src/sponge.py), and saves
what the edge pipeline would receive:

  - detections above the 0.4 forwarding threshold (box, score, class)
  - an appearance embedding per detection (the ReID feature DeepSORT needs),
    computed from the image the detector actually saw, attacked or not
  - ground-truth boxes, classes and object ids for tracking metrics

Stage B (scripts/replay_pipeline.py, CPU) replays these through the tracker
with and without the admission guard. Splitting the stages means the GPU work
runs once per checkpoint and every guard variant is compared on identical inputs.

    python scripts/extract_dets.py --config configs/nuscenes/train_baseline.yaml \
        --ckpt checkpoints/nuscenes_baseline_s0.pth

Output: results/pipeline/<ckpt stem>/<attack>.pkl
"""

import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from torchvision.io import ImageReadMode, read_image

from src.attacks import pgd
from src.sponge import apply_uap, sponge_pgd
from src.logutil import start_log
from src.models import load_detector
from src.reid import EMB_DIM, Embedder, crops_from


def pick_scenes(records, conditions, per_cond, split="test"):
    """Evenly spaced scenes of `split` per condition (deterministic)."""
    by_scene = defaultdict(list)
    for r in records:
        if r["split"] == split and r["condition"] in conditions:
            by_scene[r["scene"]].append(r)
    chosen = []
    for cond in conditions:
        names = sorted(s for s, fr in by_scene.items() if fr[0]["condition"] == cond)
        idx = np.linspace(0, len(names) - 1, num=min(per_cond, len(names))).round().astype(int)
        chosen += [names[i] for i in sorted(set(idx.tolist()))]
    return [(s, sorted(by_scene[s], key=lambda r: r["timestamp"])) for s in chosen]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--attacks", nargs="+", default=["clean", "linf", "l2", "l1", "inflate", "sponge_act", "uap"],
                    help="accuracy PGD: linf l2 l1 | sponge: inflate sponge_act uap | adaptive: inflate_hi")
    ap.add_argument("--uap", default=None, help="universal perturbation .pt (default results/uap/<ckpt stem>.pt)")
    ap.add_argument("--conditions", nargs="+", default=["day", "night", "rain", "night_rain"])
    ap.add_argument("--scenes-per-cond", type=int, default=5)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--score-thr", type=float, default=0.4)
    ap.add_argument("--max-dets", type=int, default=100, help="deployment cap per frame")
    ap.add_argument("--split", default="test", choices=["test", "select"],
                    help="select = day-only held-out training scenes, for choosing guard settings without touching test")
    args = ap.parse_args()
    stem = Path(args.ckpt).stem
    start_log(ROOT, f"extract_{stem}")

    cfg = yaml.safe_load(open(ROOT / args.config))
    ncfg = cfg["nuscenes"]
    data_root = ROOT / ncfg["root"] if not Path(ncfg["root"]).is_absolute() else Path(ncfg["root"])
    index = ROOT / ncfg["index"] if not Path(ncfg["index"]).is_absolute() else Path(ncfg["index"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = bool(cfg["train"]["amp"]) and dev == "cuda"

    records = json.load(open(index))["records"]
    conds = args.conditions if args.split == "test" else ["day"]          # select split is day-only
    scenes = pick_scenes(records, conds, args.scenes_per_cond, args.split)
    n_frames = sum(len(f) for _, f in scenes)
    print(f"{len(scenes)} scenes, {n_frames} keyframes, attacks {args.attacks}, device {dev}")

    model, _ = load_detector(str(ROOT / args.ckpt), dev)
    delta = None
    if "uap" in args.attacks:
        uap_path = Path(args.uap) if args.uap else ROOT / "results" / "uap" / f"{stem}.pt"
        if not uap_path.is_file():
            raise FileNotFoundError(f"{uap_path} missing: run scripts/train_uap.py first, or drop 'uap'")
        delta = torch.load(uap_path, map_location="cpu")["delta"].to(dev)
    model.roi_heads.detections_per_img = args.max_dets
    embedder = Embedder().eval().to(dev)
    out_dir = ROOT / "results" / "pipeline" / (stem if args.split == "test" else f"{stem}_{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for attack in args.attacks:
        t0, done, result = time.time(), 0, []
        for scene, frames in scenes:
            seq = {"scene": scene, "condition": frames[0]["condition"], "frames": []}
            for r in frames:
                x = read_image(str(data_root / r["file"]), ImageReadMode.RGB).float().div(255).to(dev)
                gt_boxes = torch.tensor(r["boxes"], dtype=torch.float32).reshape(-1, 4)
                gt_labels = torch.tensor(r["labels"], dtype=torch.int64)
                if attack in ("linf", "l2", "l1"):                 # accuracy attack (loss-max PGD)
                    tgt = {"boxes": gt_boxes.to(dev), "labels": gt_labels.to(dev)}
                    x = pgd(model, [x], [tgt], attack, cfg["adv"], args.steps, amp)[0]
                    x = torch.round(x * 255) / 255          # what a camera / saved frame carries
                elif attack in ("inflate", "inflate_hi", "sponge_act"):   # sponge attacks, L_inf budget
                    x = sponge_pgd(model, x, attack, cfg["adv"], args.steps, "linf", amp)
                elif attack == "uap":                              # universal, same delta every frame
                    x = apply_uap(x, delta)
                elif attack != "clean":
                    raise ValueError(attack)
                model.eval()
                with torch.no_grad():
                    out = model([x])[0]
                    keep = out["scores"] >= args.score_thr
                    boxes = out["boxes"][keep]
                    emb = embedder(crops_from(x, boxes)) if len(boxes) else torch.zeros(0, EMB_DIM, device=dev)
                dets = torch.cat([boxes, out["scores"][keep, None], out["labels"][keep, None].float()], 1)
                seq["frames"].append({
                    "file": r["file"], "timestamp": r["timestamp"],
                    "dets": dets.cpu().numpy().astype(np.float32),        # x1,y1,x2,y2,score,label
                    "emb": emb.cpu().numpy().astype(np.float16),
                    "gt_boxes": gt_boxes.numpy(), "gt_labels": gt_labels.numpy(),
                    "gt_ids": list(r["ids"]),
                })
                done += 1
            result.append(seq)
            el = time.time() - t0
            print(f"  [{attack}] {done}/{n_frames} frames, {el / 60:.1f} min, "
                  f"~{el / done * (n_frames - done) / 60:.1f} min left", flush=True)
        path = out_dir / f"{attack}.pkl"
        with open(path, "wb") as fh:
            pickle.dump({"ckpt": stem, "attack": attack, "score_thr": args.score_thr,
                         "max_dets": args.max_dets, "scenes": result}, fh)
        mean_d = np.mean([len(f["dets"]) for s in result for f in s["frames"]])
        print(f"wrote {path}  (mean {mean_d:.2f} forwarded detections/frame)")


if __name__ == "__main__":
    main()
