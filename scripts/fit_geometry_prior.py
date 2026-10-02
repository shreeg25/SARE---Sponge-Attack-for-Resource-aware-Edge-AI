"""
Fit the guard's geometry prior from TRAIN-split ground truth (day scenes only;
no select or test data). For a forward-facing road camera, an object's box
height is tied to where its bottom edge sits in the image (perspective): near
the horizon objects are small, near the bottom large. Per class and per band
of bottom-edge height, store the 1st-99th percentile of box height.

    python scripts/fit_geometry_prior.py      -> data/geometry_prior.json
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.logutil import start_log

IMG_H = 900          # nuScenes CAM_FRONT is 1600 x 900
N_BINS = 18
MIN_SAMPLES = 30


def main():
    start_log(ROOT, "fit_geometry_prior")
    recs = json.load(open(ROOT / "data" / "nuscenes_2d.json"))["records"]
    pts = {}
    for r in recs:
        if r["split"] != "train":
            continue
        for (x1, y1, x2, y2), lab in zip(r["boxes"], r["labels"]):
            pts.setdefault(int(lab), []).append((y2 / IMG_H, (y2 - y1) / IMG_H))
    prior = {"img_h": IMG_H, "n_bins": N_BINS, "classes": {}}
    edges = np.linspace(0, 1, N_BINS + 1)
    for lab, p in sorted(pts.items()):
        p = np.array(p)
        glo, ghi = np.percentile(p[:, 1], [1, 99])
        bins = []
        for b in range(N_BINS):
            m = (p[:, 0] >= edges[b]) & (p[:, 0] < edges[b + 1] if b < N_BINS - 1 else p[:, 0] <= 1)
            hs = p[m, 1]
            lo, hi = (np.percentile(hs, [1, 99]) if len(hs) >= MIN_SAMPLES else (glo, ghi))
            bins.append([float(lo), float(hi), int(len(hs))])
        prior["classes"][str(lab)] = {"bins": bins, "global": [float(glo), float(ghi)], "n": int(len(p))}
        print(f"class {lab}: {len(p)} boxes, height 1-99% {glo * IMG_H:.0f}-{ghi * IMG_H:.0f} px")
    out = ROOT / "data" / "geometry_prior.json"
    json.dump(prior, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
