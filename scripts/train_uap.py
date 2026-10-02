"""
Universal sponge perturbation (Phantom Sponges, Shapira et al. WACV 2023),
optimised white-box against one checkpoint on SELECT-split frames only (never
test), with the inflate objective from src/sponge.py:

    delta <- clip_eps( delta + alpha * sign( grad_delta  L_inflate(x + delta) ) )

over random select frames, one frame per step. The same delta is then added
to every test frame (extract_dets.py --attacks uap), so its phantoms appear at
the same places in consecutive frames and can survive into confirmed tracks.

    python scripts/train_uap.py --config configs/nuscenes/train_baseline.yaml \
        --ckpt checkpoints/nuscenes_linf_at_aug_s0.pth

Writes results/uap/<ckpt stem>.pt  ({"delta": 3xHxW tensor, "eps", "iters", ...})
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from src.data import NuScenes2D
from src.logutil import start_log
from src.models import load_detector
from src.sponge import Unsuppressed, inflation_objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--eps", type=float, default=None, help="L_inf budget; default: adv.linf_eps from the config")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    stem = Path(args.ckpt).stem
    start_log(ROOT, f"uap_{stem}")

    cfg = yaml.safe_load(open(ROOT / args.config))
    c = cfg["nuscenes"]
    root = ROOT / c["root"] if not Path(c["root"]).is_absolute() else Path(c["root"])
    index = ROOT / c["index"] if not Path(c["index"]).is_absolute() else Path(c["index"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = bool(cfg["train"]["amp"]) and dev == "cuda"
    eps = args.eps if args.eps is not None else cfg["adv"]["linf_eps"]
    alpha = eps / 10

    ds = NuScenes2D(str(index), str(root), "select", ["day"], stride=1)
    model, _ = load_detector(str(ROOT / args.ckpt), dev)
    model.requires_grad_(False)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    x0, _ = ds[0]
    delta = ((torch.rand_like(x0) * 2 - 1) * eps).to(dev)
    print(f"{len(ds)} select frames, eps {eps * 255:.2f}/255, {args.iters} iterations, device {dev}")
    t0, hist = time.time(), []
    for it in range(1, args.iters + 1):
        x, _ = ds[int(rng.integers(0, len(ds)))]
        x = x.to(dev)
        d = delta.clone().requires_grad_(True)
        with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=amp), Unsuppressed(model):
            v = inflation_objective(model([(x + d).clamp(0, 1)])[0])
        (g,) = torch.autograd.grad(v, d, allow_unused=True)
        if g is not None:
            with torch.no_grad():
                delta = (delta + alpha * g.float().sign()).clamp(-eps, eps)
        hist.append(float(v))
        if it % 50 == 0:
            print(f"  iter {it:5d}  inflate objective (last 50 mean) {np.mean(hist[-50:]):7.2f}  "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)

    out = ROOT / "results" / "uap" / f"{stem}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"delta": delta.cpu(), "eps": eps, "iters": args.iters, "ckpt": stem,
                "objective": "inflate", "split": "select/day"}, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
