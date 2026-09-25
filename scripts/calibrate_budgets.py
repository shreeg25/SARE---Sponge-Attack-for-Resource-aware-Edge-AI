"""
Pick L2 / L1 budgets that hurt the trained BASELINE as much as L_inf 8/255 does.
Run this after the baseline finishes and before training MNAT / L_inf-only,
then copy the chosen l2_rms / l1_mean into all three train configs.

    python scripts/calibrate_budgets.py --ckpt checkpoints/baseline_s0.pth
"""

import argparse
import copy
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from src.attacks import pgd
from src.data import build_datasets
from src.logutil import start_log
from src.metrics import detection_report
from src.models import load_detector

GRID = {
    "linf": ("linf_eps", [0.25 / 255, 0.5 / 255, 1 / 255, 2 / 255, 4 / 255, 8 / 255]),
    "l2": ("l2_rms", [0.0625 / 255, 0.125 / 255, 0.25 / 255, 0.5 / 255, 1 / 255, 2 / 255]),
    "l1": ("l1_mean", [0.00390625 / 255, 0.0078125 / 255, 0.015625 / 255, 0.03125 / 255,
                       0.0625 / 255, 0.125 / 255, 0.25 / 255, 0.5 / 255, 1 / 255]),
}


def ap_under(model, ds, idx, adv_cfg, norm, steps, dev, amp, num_classes):
    preds, gts = [], []
    for i in idx:
        img, tgt = ds[int(i)]
        img, tgt = img.to(dev), {k: v.to(dev) for k, v in tgt.items()}
        if norm:
            img = pgd(model, [img], [tgt], norm, adv_cfg, steps, amp)[0]
        model.eval()
        with torch.no_grad():
            out = model([img])[0]
        preds.append({k: v.cpu() for k, v in out.items()})
        gts.append({k: v.cpu() for k, v in tgt.items()})
    return detection_report(preds, gts, num_classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/train_baseline.yaml")
    ap.add_argument("--images", type=int, default=100)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--norms", nargs="+", default=list(GRID), choices=list(GRID),
                    help="only run these norms, e.g. --norms l1")
    args = ap.parse_args()
    stem = Path(args.ckpt).stem
    start_log(ROOT, f"calibrate_{stem}")

    cfg = yaml.safe_load(open(ROOT / args.config))
    ds_cfg = cfg[cfg["dataset"]]
    for key in ("root", "index"):
        if key in ds_cfg and not Path(ds_cfg[key]).is_absolute():
            ds_cfg[key] = str(ROOT / ds_cfg[key])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = bool(cfg["train"]["amp"]) and dev == "cuda"

    _, val = build_datasets(cfg)   # select split: budgets are fixed before any test data is seen
    idx = np.linspace(0, len(val) - 1, num=min(args.images, len(val))).astype(int)
    model, _ = load_detector(str(ROOT / args.ckpt), dev)
    ncls = cfg["model"]["num_classes"]

    suffix = "" if set(args.norms) == set(GRID) else "_" + "_".join(args.norms)
    out_csv = ROOT / "results" / "calibration" / f"{stem}{suffix}.csv"   # partial reruns never overwrite the full table
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out_csv, "w", newline="")
    w = csv.writer(fh)
    cols = ["ap50", "precision", "recall", "f1", "fp_per_frame", "dets_per_frame"]
    w.writerow(["norm", "budget_x255", *cols, "ap50_drop", "images", "steps"])

    c = ap_under(model, val, idx, cfg["adv"], None, 0, dev, amp, ncls)
    clean = c["ap50"]
    w.writerow(["clean", 0, *[round(c[k], 5) for k in cols], 0, len(idx), 0])
    head = f"{'norm':5s} {'budget(x/255)':>13s} {'AP50':>7s} {'P':>6s} {'R':>6s} {'F1':>6s} {'FP/frm':>7s} {'det/frm':>7s} {'drop':>7s}"
    row = lambda nm, b, r: (f"{nm:5s} {b:13.4f} {r['ap50']:7.4f} {r['precision']:6.3f} {r['recall']:6.3f} "
                            f"{r['f1']:6.3f} {r['fp_per_frame']:7.2f} {r['dets_per_frame']:7.2f} {clean - r['ap50']:7.4f}")
    print(head)
    print(row("clean", 0.0, c), flush=True)
    for norm, (key, values) in GRID.items():
        if norm not in args.norms:
            continue
        for v in values:
            adv_cfg = copy.deepcopy(cfg["adv"])
            adv_cfg[key] = v
            r = ap_under(model, val, idx, adv_cfg, norm, args.steps, dev, amp, ncls)
            print(row(norm, v * 255, r), flush=True)
            w.writerow([norm, round(v * 255, 5), *[round(r[k], 5) for k in cols],
                        round(clean - r["ap50"], 5), len(idx), args.steps])
            fh.flush()
    fh.close()
    print(f"\ntable saved to {out_csv}")
    print("Send this table back: budgets are matched at the L_inf eps where AP50 is ~half of clean.")


if __name__ == "__main__":
    main()