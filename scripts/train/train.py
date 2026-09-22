"""
SARE detector training. One script for every arm; the configs differ only in
the `adv` block, so baseline / MNAT / L_inf-only share an identical recipe.

    loss = L_det(clean) + lambda * L_det(PGD adversarial)      (no second term for baseline)

MNAT cycles the attack norm per iteration: linf, l2, l1, linf, ...

Usage (from the repo root):
    python scripts/train/train.py --config configs/train_baseline.yaml --smoke
    python scripts/train/train.py --config configs/train_mnat.yaml --seed 1

Writes checkpoints/<arm>_s<seed>.pth after every epoch, a JSONL log in
results/train/, and a final clean + robust AP50 summary.
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.attacks import pgd
from src.data import build_datasets, collate
from src.metrics import ap50
from src.models import build_detector


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def evaluate(model, ds, n, cfg, dev, norm=None, amp=False):
    """AP50 on n evenly spaced val frames, clean or under PGD with `norm`.
    Also returns mean detections/frame above the 0.4 forwarding threshold."""
    idx = np.linspace(0, len(ds) - 1, num=min(n, len(ds))).astype(int)
    preds, gts, ndets = [], [], []
    for i in idx:
        img, tgt = ds[int(i)]
        img = img.to(dev)
        tgt = {k: v.to(dev) for k, v in tgt.items()}
        if norm:
            img = pgd(model, [img], [tgt], norm, cfg["adv"], cfg["eval"]["robust_steps"], amp)[0]
        model.eval()
        with torch.no_grad():
            out = model([img])[0]
        preds.append({k: v.cpu() for k, v in out.items()})
        gts.append({k: v.cpu() for k, v in tgt.items()})
        ndets.append(int((out["scores"] > 0.4).sum()))
    model.train()
    return ap50(preds, gts, cfg["model"]["num_classes"]), float(np.mean(ndets))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="5 iterations + tiny eval, to catch errors fast")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    seed = cfg["seed"] if args.seed is None else args.seed
    seed_all(seed)

    if cfg["dataset"] == "mot17" and not Path(cfg["mot17"]["root"]).is_absolute():
        cfg["mot17"]["root"] = str(ROOT / cfg["mot17"]["root"])
    if args.smoke:
        cfg["eval"].update(val_images=4, robust_images=2, robust_steps=2)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, m, adv_cfg = cfg["train"], cfg["model"], cfg["adv"]
    amp = bool(tr["amp"]) and dev == "cuda"

    train_ds, val_ds = build_datasets(cfg)
    print(f"train frames {len(train_ds)}, val frames {len(val_ds)}, device {dev}, amp {amp}")

    gen = torch.Generator()
    gen.manual_seed(seed)
    dl = DataLoader(train_ds, batch_size=tr["batch_size"], shuffle=True, drop_last=True,
                    num_workers=tr["num_workers"], collate_fn=collate, generator=gen,
                    persistent_workers=tr["num_workers"] > 0, pin_memory=dev == "cuda")

    model = build_detector(m["arch"], m["num_classes"], pretrained=True, min_size=m["min_size"],
                           max_size=m["max_size"], trainable_layers=m["trainable_layers"]).to(dev)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=tr["lr"], momentum=tr["momentum"], weight_decay=tr["weight_decay"])

    total = tr["epochs"] * len(dl)
    warm = min(tr["warmup_iters"], max(1, total // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: (it + 1) / warm if it < warm
        else 0.5 * (1 + math.cos(math.pi * (it - warm) / max(1, total - warm))))

    norms = adv_cfg.get("norms") or []
    name = f"{cfg['arm']}_s{seed}" + ("_smoke" if args.smoke else "")
    (ROOT / "checkpoints").mkdir(exist_ok=True)
    log_dir = ROOT / "results" / "train"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = open(log_dir / f"{name}.jsonl", "a")
    meta = {"arm": cfg["arm"], "seed": seed, "arch": m["arch"], "num_classes": m["num_classes"],
            "min_size": m["min_size"], "max_size": m["max_size"], "adv": adv_cfg,
            "dataset": cfg["dataset"]}

    step, t0, skipped = 0, time.time(), 0
    for epoch in range(tr["epochs"]):
        for imgs, tgts in dl:
            imgs = [i.to(dev, non_blocking=True) for i in imgs]
            tgts = [{k: v.to(dev) for k, v in t.items()} for t in tgts]

            with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=amp):
                loss_clean = sum(model(imgs, tgts).values())
            loss, loss_adv, norm = loss_clean, None, None
            if norms:
                norm = norms[step % len(norms)]
                adv = pgd(model, imgs, tgts, norm, adv_cfg, adv_cfg["train_steps"], amp)
                with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=amp):
                    loss_adv = sum(model(adv, tgts).values())
                loss = loss_clean + adv_cfg["lambda"] * loss_adv

            if not torch.isfinite(loss):
                skipped += 1
                opt.zero_grad(set_to_none=True)
                step += 1
                sched.step()
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, tr["grad_clip"])
            opt.step()
            sched.step()
            step += 1

            if step % 20 == 0 or args.smoke:
                rec = {"step": step, "epoch": epoch, "lr": opt.param_groups[0]["lr"],
                       "loss_clean": loss_clean.item(), "norm": norm,
                       "loss_adv": None if loss_adv is None else loss_adv.item(),
                       "s_per_it": (time.time() - t0) / step, "skipped": skipped}
                print(json.dumps(rec))
                log.write(json.dumps(rec) + "\n")
                log.flush()
            if args.smoke and step >= 5:
                break

        ap_clean, nd = evaluate(model, val_ds, cfg["eval"]["val_images"], cfg, dev, None, amp)
        rec = {"epoch": epoch + 1, "val_ap50": ap_clean, "val_dets_per_frame": nd,
               "elapsed_min": (time.time() - t0) / 60}
        print(json.dumps(rec))
        log.write(json.dumps(rec) + "\n")
        log.flush()
        torch.save({"model": model.state_dict(), "meta": meta, "epoch": epoch + 1},
                   ROOT / "checkpoints" / f"{name}.pth")
        if args.smoke:
            break

    # Did hardening work? Clean vs PGD AP50 under every norm, for every arm.
    final = {"name": name, "clean_ap50": ap_clean}
    for rn in cfg["eval"]["robust_norms"]:
        final[f"{rn}_ap50"] = evaluate(model, val_ds, cfg["eval"]["robust_images"], cfg, dev, rn, amp)[0]
    print(json.dumps(final, indent=2))
    with open(log_dir / f"{name}_final.json", "w") as fh:
        json.dump(final, fh, indent=2)
    log.close()


if __name__ == "__main__":
    main()
