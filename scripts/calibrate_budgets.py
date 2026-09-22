"""
Pick L2 / L1 budgets that hurt the trained BASELINE as much as L_inf 8/255 does.
Run this after the baseline finishes and before training MNAT / L_inf-only,
then copy the chosen l2_rms / l1_mean into all three train configs.

    python scripts/calibrate_budgets.py --ckpt checkpoints/baseline_s0.pth
"""

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from src.attacks import pgd
from src.data import build_datasets
from src.metrics import ap50
from src.models import load_detector

GRID = {
    "linf": ("linf_eps", [0.25 / 255, 0.5 / 255, 1 / 255, 2 / 255, 4 / 255, 8 / 255]),
    "l2": ("l2_rms", [0.0625 / 255, 0.125 / 255, 0.25 / 255, 0.5 / 255, 1 / 255, 2 / 255]),
    "l1": ("l1_mean", [0.03125 / 255, 0.0625 / 255, 0.125 / 255, 0.25 / 255, 0.5 / 255, 1 / 255]),
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
    return ap50(preds, gts, num_classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/train_baseline.yaml")
    ap.add_argument("--images", type=int, default=100)
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / args.config))
    ds_cfg = cfg[cfg["dataset"]]
    for key in ("root", "index"):
        if key in ds_cfg and not Path(ds_cfg[key]).is_absolute():
            ds_cfg[key] = str(ROOT / ds_cfg[key])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = bool(cfg["train"]["amp"]) and dev == "cuda"

    _, val = build_datasets(cfg)
    idx = np.linspace(0, len(val) - 1, num=min(args.images, len(val))).astype(int)
    model, _ = load_detector(str(ROOT / args.ckpt), dev)
    ncls = cfg["model"]["num_classes"]

    clean = ap_under(model, val, idx, cfg["adv"], None, 0, dev, amp, ncls)
    print(f"clean AP50 {clean:.4f}\n")
    print(f"{'norm':5s} {'budget (x/255)':>15s} {'AP50':>8s} {'drop':>8s}")
    for norm, (key, values) in GRID.items():
        for v in values:
            adv_cfg = copy.deepcopy(cfg["adv"])
            adv_cfg[key] = v
            a = ap_under(model, val, idx, adv_cfg, norm, args.steps, dev, amp, ncls)
            print(f"{norm:5s} {v * 255:15.2f} {a:8.4f} {clean - a:8.4f}", flush=True)
    print("\nSend this table back: budgets are matched at the L_inf eps where AP50 is ~half of clean.")


if __name__ == "__main__":
    main()
