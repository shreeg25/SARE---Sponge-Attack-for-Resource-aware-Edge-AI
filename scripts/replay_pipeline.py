"""
Stage B of the pipeline experiments (CPU; runs unchanged on a Raspberry Pi 5).
Replays saved detections through guard -> DeepSORT and measures what the edge
node pays and how well it tracks.

Per frame it counts:
  dets_in      detections forwarded by the detector (score >= 0.4)
  reid_calls   detections admitted by the guard = appearance-embedding passes
  tracks_out   confirmed tracks updated this frame = objects sent upstream
  uplink_B     tracks_out x 32 bytes (payload model from the MNAT paper, Alg. 1)
  tracker_ms   DeepSORT update time, guard_ms guard decision time

and per scene the CLEAR-MOT / identity metrics (MOTA, MOTP, IDF1, switches)
against nuScenes ground-truth object ids, class-agnostic, IoU >= 0.5.

    python scripts/replay_pipeline.py --run results/pipeline/nuscenes_baseline_s0 \
        --attacks clean linf l2 l1 --guards none cap10

--reid-ms (per-crop embedding cost from scripts/bench_reid.py on the target
device) converts reid_calls into milliseconds.
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

import motmetrics as mm
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from src.guard import build_guard
from src.logutil import start_log

PAYLOAD_BYTES = 32
MOT_METRICS = ["mota", "motp", "idf1", "idtp", "idfp", "idfn", "num_switches",
               "num_false_positives", "num_misses", "mostly_tracked", "num_objects"]


def ltwh(b):
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)


def iou_distance(gt, hyp, min_iou=0.5):
    """1 - IoU between ltwh boxes, NaN below min_iou (motmetrics convention).
    Own implementation: motmetrics.distances.iou_matrix breaks on NumPy 2."""
    if len(gt) == 0 or len(hyp) == 0:
        return np.empty((len(gt), len(hyp)))
    g, h = gt[:, None, :], hyp[None, :, :]
    iw = np.clip(np.minimum(g[..., 0] + g[..., 2], h[..., 0] + h[..., 2]) - np.maximum(g[..., 0], h[..., 0]), 0, None)
    ih = np.clip(np.minimum(g[..., 1] + g[..., 3], h[..., 1] + h[..., 3]) - np.maximum(g[..., 1], h[..., 1]), 0, None)
    inter = iw * ih
    iou = inter / (g[..., 2] * g[..., 3] + h[..., 2] * h[..., 3] - inter)
    d = 1.0 - iou
    d[iou < min_iou] = np.nan
    return d


def replay_scene(seq, guard, args):
    tracker = DeepSort(max_age=args.max_age, n_init=args.n_init,
                       max_cosine_distance=args.max_cos, embedder=None)
    guard.reset()
    acc = mm.MOTAccumulator(auto_id=True)
    gt_num = {}          # motmetrics needs numeric ids; nuScenes ids are tokens
    rows = []
    for fr in seq["frames"]:
        keep = fr["dets"][:, 4] >= args.score_thr if len(fr["dets"]) else np.zeros(0, dtype=bool)
        dets, emb = fr["dets"][keep], fr["emb"][keep]
        t0 = time.perf_counter()
        idx = guard.admit(dets, tracker)
        guard_ms = (time.perf_counter() - t0) * 1e3

        raw, embeds = [], []
        for i in idx:
            x1, y1, x2, y2, s, lab = dets[i]
            if x2 - x1 > 0 and y2 - y1 > 0:
                raw.append(([float(x1), float(y1), float(x2 - x1), float(y2 - y1)], float(s), int(lab)))
                embeds.append(emb[i].astype(np.float32))
        t0 = time.perf_counter()
        tracks = tracker.update_tracks(raw, embeds=embeds)
        tracker_ms = (time.perf_counter() - t0) * 1e3

        out = [t for t in tracks if t.is_confirmed() and t.time_since_update == 0]
        hyp_boxes = np.array([t.to_ltwh(orig=True) for t in out]).reshape(-1, 4)
        gt = ltwh(fr["gt_boxes"])
        acc.update([gt_num.setdefault(g, len(gt_num)) for g in fr["gt_ids"]],
                   [int(t.track_id) for t in out],
                   iou_distance(gt, hyp_boxes))
        rows.append({"dets_in": len(dets), "reid_calls": len(raw), "tracks_out": len(out),
                     "uplink_B": PAYLOAD_BYTES * len(out), "tracker_ms": tracker_ms,
                     "guard_ms": guard_ms, "gt": len(gt)})
    return acc, rows


def summarise(rows, reid_ms):
    a = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}
    s = {"frames": len(rows)}
    for k in ("dets_in", "reid_calls", "tracks_out", "uplink_B", "gt"):
        s[f"{k}_per_frame"] = float(a[k].mean())
    s["reid_calls_p95"] = float(np.percentile(a["reid_calls"], 95))
    s["tracker_ms_mean"] = float(a["tracker_ms"].mean())
    s["tracker_ms_p95"] = float(np.percentile(a["tracker_ms"], 95))
    s["guard_ms_mean"] = float(a["guard_ms"].mean())
    if reid_ms is not None:
        s["reid_ms_per_frame"] = float(a["reid_calls"].mean() * reid_ms)
        s["pipeline_ms_per_frame"] = s["reid_ms_per_frame"] + s["tracker_ms_mean"] + s["guard_ms_mean"]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="results/pipeline/<ckpt stem> written by extract_dets.py")
    ap.add_argument("--attacks", nargs="+", default=["clean", "linf", "l2", "l1"])
    ap.add_argument("--guards", nargs="+", default=["none"])
    ap.add_argument("--n-init", type=int, default=3)
    ap.add_argument("--max-age", type=int, default=4, help="frames; keyframes are 2 Hz, so 4 = 2 s")
    ap.add_argument("--max-cos", type=float, default=0.4)
    ap.add_argument("--reid-ms", type=float, default=None, help="per-crop embedding ms on the target device")
    ap.add_argument("--score-thr", type=float, default=0.4, help="forwarding threshold applied to saved detections")
    ap.add_argument("--prior", default="data/geometry_prior.json", help="geometry prior for sare guards")
    args = ap.parse_args()
    run = ROOT / args.run if not Path(args.run).is_absolute() else Path(args.run)
    start_log(ROOT, f"replay_{run.name}")

    mh = mm.metrics.create()
    table = []
    for attack in args.attacks:
        data = pickle.load(open(run / f"{attack}.pkl", "rb"))
        for gspec in args.guards:
            guard = build_guard(gspec, ROOT / args.prior if not Path(args.prior).is_absolute() else Path(args.prior))
            accs, names, rows_by_cond = defaultdict(list), defaultdict(list), defaultdict(list)
            for seq in data["scenes"]:
                acc, rows = replay_scene(seq, guard, args)
                accs[seq["condition"]].append(acc)
                names[seq["condition"]].append(seq["scene"])
                rows_by_cond[seq["condition"]] += rows
            result = {}
            for cond in rows_by_cond:
                mot = mh.compute_many(accs[cond], metrics=MOT_METRICS, names=names[cond],
                                      generate_overall=True).loc["OVERALL"].to_dict()
                result[cond] = {**summarise(rows_by_cond[cond], args.reid_ms),
                                **{k: float(v) for k, v in mot.items()}}
                table.append((cond, attack, guard.name, result[cond]))
            with open(run / f"replay_{attack}_{guard.name}.json", "w") as fh:
                json.dump({"attack": attack, "guard": guard.name, "args": vars(args), "results": result}, fh, indent=2)

    print(f"\n{run.name}  (n_init={args.n_init}, max_age={args.max_age}, max_cos={args.max_cos})")
    head = (f"{'condition':11s} {'attack':6s} {'guard':6s} {'det/f':>6s} {'reid/f':>6s} {'p95':>5s} "
            f"{'out/f':>6s} {'up B/f':>7s} {'trk ms':>7s} {'MOTA':>7s} {'IDF1':>6s} {'IDsw':>5s}")
    print(head)
    order = {c: i for i, c in enumerate(["day", "night", "rain", "night_rain"])}
    for cond, attack, g, r in sorted(table, key=lambda t: (order.get(t[0], 9), t[1] != "clean", t[1], t[2])):
        print(f"{cond:11s} {attack:6s} {g:6s} {r['dets_in_per_frame']:6.2f} {r['reid_calls_per_frame']:6.2f} "
              f"{r['reid_calls_p95']:5.0f} {r['tracks_out_per_frame']:6.2f} {r['uplink_B_per_frame']:7.1f} "
              f"{r['tracker_ms_mean']:7.2f} {r['mota']:7.3f} {r['idf1']:6.3f} {int(r['num_switches']):5d}")


if __name__ == "__main__":
    main()
