"""
SARE detector training. One script for every arm; the configs differ only in
the `adv` block, so baseline / MNAT / L_inf-only share an identical recipe.

    loss = L_det(clean) + lambda * L_det(PGD adversarial)      (no second term for baseline)

MNAT cycles the attack norm per iteration: linf, l2, l1, linf, ...

Usage (from the repo root):
    python scripts/train/train.py --config configs/nuscenes/train_baseline.yaml --smoke
    python scripts/train/train.py --config configs/nuscenes/train_mnat.yaml --seed 1

Keeps the best epoch (by the selection score) as checkpoints/<dataset>_<arm>_s<seed>.pth
and the latest as ..._last.pth; stops early after `patience` epochs
without improvement. Logs to results/train/ and ends with a clean + robust
AP50 summary of the selected checkpoint.
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
from src.data import build_datasets, build_test_sets, collate
from src.logutil import start_log
from src.metrics import ap50
from src.models import build_detector


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def evaluate(model, ds, n, cfg, dev, norm=None, amp=False):
    """AP50 on n evenly spaced frames of ds (all frames if n is null), clean or
    under PGD with `norm`. Evenly spaced = deterministic, so every arm sees the
    same frames.
    Also returns mean detections/frame above the 0.4 forwarding threshold."""
    idx = np.linspace(0, len(ds) - 1, num=min(n or len(ds), len(ds))).astype(int)
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
    start_log(ROOT, f"train_{cfg['dataset']}_{cfg['arm']}_s{seed}" + ("_smoke" if args.smoke else ""))
    seed_all(seed)

    ds_cfg = cfg[cfg["dataset"]]
    for key in ("root", "index"):
        if key in ds_cfg and not Path(ds_cfg[key]).is_absolute():
            ds_cfg[key] = str(ROOT / ds_cfg[key])
    if args.smoke:
        cfg["eval"].update(select_clean_images=4, select_robust_images=2, test_clean_images=4,
                           test_robust_images=2, robust_steps=2)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, m, adv_cfg = cfg["train"], cfg["model"], cfg["adv"]
    amp = bool(tr["amp"]) and dev == "cuda"

    train_ds, select_ds = build_datasets(cfg)   # select split: checkpoint selection only
    print(f"train frames {len(train_ds)}, select frames {len(select_ds)}, device {dev}, amp {amp}")

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
    name = f"{cfg['dataset']}_{cfg['arm']}_s{seed}" + ("_smoke" if args.smoke else "")
    (ROOT / "checkpoints").mkdir(exist_ok=True)
    log_dir = ROOT / "results" / "train"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = open(log_dir / f"{name}.jsonl", "a")
    meta = {"arm": cfg["arm"], "seed": seed, "arch": m["arch"], "num_classes": m["num_classes"],
            "min_size": m["min_size"], "max_size": m["max_size"], "adv": adv_cfg,
            "dataset": cfg["dataset"]}

    step, t0, skipped = 0, time.time(), 0
    best_score, best_epoch, bad = -1.0, 0, 0
    best_path = ROOT / "checkpoints" / f"{name}.pth"
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

        ap_clean, nd = evaluate(model, select_ds, cfg["eval"]["select_clean_images"], cfg, dev, None, amp)
        # Same selection rule for every arm: mean of clean AP50 and L_inf-PGD AP50.
        # Guards against robust overfitting in the adversarial arms (Rice et al., 2020).
        ap_rob = evaluate(model, select_ds, cfg["eval"]["select_robust_images"], cfg, dev, "linf", amp)[0]
        score = 0.5 * (ap_clean + ap_rob)
        state = {"model": model.state_dict(), "meta": meta, "epoch": epoch + 1}
        torch.save(state, ROOT / "checkpoints" / f"{name}_last.pth")
        if score > best_score:
            best_score, best_epoch, bad = score, epoch + 1, 0
            torch.save(state, best_path)
        else:
            bad += 1
        rec = {"epoch": epoch + 1, "val_ap50": ap_clean, "val_linf_ap50": ap_rob,
               "select_score": score, "best_epoch": best_epoch,
               "val_dets_per_frame": nd, "elapsed_min": (time.time() - t0) / 60}
        print(json.dumps(rec))
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if args.smoke:
            break
        if bad >= tr["patience"]:
            print(f"early stop: no improvement for {bad} epochs, best epoch {best_epoch}")
            break

    # Did hardening work? Clean vs PGD AP50 under every norm, on the selected checkpoint,
    # measured on held-out test scenes per condition (select split only for MOT17).
    model.load_state_dict(torch.load(best_path, map_location=dev, weights_only=False)["model"])
    tests = build_test_sets(cfg)
    final = {"name": name, "best_epoch": best_epoch, "eval_split": "test" if tests else "select"}
    n_clean, n_rob = cfg["eval"]["test_clean_images"], cfg["eval"]["test_robust_images"]
    for cond, ds in (tests or {"select": select_ds}).items():
        res = {"frames": len(ds),
               "clean_frames": min(n_clean or len(ds), len(ds)),
               "robust_frames": min(n_rob or len(ds), len(ds)),
               "clean_ap50": evaluate(model, ds, n_clean, cfg, dev, None, amp)[0]}
        for rn in cfg["eval"]["robust_norms"]:
            res[f"{rn}_ap50"] = evaluate(model, ds, n_rob, cfg, dev, rn, amp)[0]
        final[cond] = res
    print(json.dumps(final, indent=2))
    with open(log_dir / f"{name}_final.json", "w") as fh:
        json.dump(final, fh, indent=2)
    log.close()


if __name__ == "__main__":
    main()
