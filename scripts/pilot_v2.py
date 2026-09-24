"""
MNAT x resource-attack study, week-1 pilot (v2)

For every checkpoint arm, every frame, every attack and every epsilon, logs one
CSV row with:
  mechanism metrics   rho (active fraction) at 3 thresholds, spatial variance,
                      channel entropy, sponge surrogate -- for layer2/3/4
  system-side metric  pedestrian detections above the 0.4 forwarding threshold

Attacks (all L_inf, same eps grid, same seeds across arms so frames are paired):
  random         Rademacher noise at +-eps (magnitude-matched control)
  evasion        PGD minimising pedestrian scores (the "make it wrong" attack)
  sponge_l0      PGD maximising activation density (Shumailov-style surrogate)
  det_inflation  PGD maximising the number of pedestrian boxes crossing 0.4

Adversarial images are quantised to 8-bit before measurement, since that is
what a camera or a saved frame would actually carry.

First run:   python pilot_v2.py --smoke      (2 frames, 3 steps, ~1 min)
Full pilot:  python pilot_v2.py
Then:        python analyze_pilot.py --csv pilot_v2_out/results.csv --ref baseline_s0 --cmp mnat_s0
"""

import argparse
import csv
import glob
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torchvision.io import ImageReadMode, read_image
from torchvision.utils import save_image

from src.logutil import start_log
from src.models import load_detector, tap_layers


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # arm name -> checkpoint. Add seeds / the L_inf-only arm here later.
    checkpoints: Dict[str, str] = field(default_factory=lambda: {
        "baseline_s0": "checkpoints/baseline_s0.pth",
        "mnat_s0": "checkpoints/mnat_s0.pth",
    })
    # One directory of frames per sequence. Add an IDD folder here if you can;
    # the detection-inflation channel is where IDD differed most.
    frame_dirs: List[str] = field(default_factory=lambda: [
        "data/MOT17/train/MOT17-02-FRCNN/img1",
        "data/MOT17/train/MOT17-04-FRCNN/img1",
        "data/MOT17/train/MOT17-09-FRCNN/img1",
    ])
    frame_start_frac: float = 0.7   # skip the training part; must match 1 - val_frac in train configs
    frame_stride: int = 30          # 1 frame/sec at 30 fps, cuts temporal correlation
    frames_per_dir: int = 10

    ped_label: int = 1              # person in both COCO and 2-class setups
    arch: str = "mobilenet_v3"      # only used for checkpoints saved without meta
    num_classes: int = 2            # only used for checkpoints saved without meta
    det_thresh: float = 0.4         # same forwarding threshold as the MNAT paper
    score_floor: float = 0.001      # low floor so attack gradients see weak boxes
    max_dets: int = 1000            # torchvision default is 100; 100 would cap inflation

    layers: tuple = ("stage2", "stage3", "stage4")   # stride 8/16/32, resolved per architecture
    rho_thresholds: tuple = (1e-4, 1e-3, 1e-2)

    epsilons: tuple = (2 / 255, 4 / 255, 8 / 255)   # 8/255 = your 0.03137 budget
    attacks: tuple = ("random", "evasion", "sponge_l0", "det_inflation")
    pgd_steps: int = 20
    restarts: int = 3
    sponge_beta: float = 20.0
    inflation_k: float = 20.0

    seed: int = 0
    save_adv: bool = False          # PNGs for the week-6 Pi 5 replay
    out_dir: str = "pilot_v2_out"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_model(path: str, cfg: Config) -> torch.nn.Module:
    model, _ = load_detector(str(ROOT / path), "cpu",
                             fallback={"arch": cfg.arch, "num_classes": cfg.num_classes})

    model.roi_heads.score_thresh = cfg.score_floor
    model.roi_heads.detections_per_img = cfg.max_dets
    model.eval().requires_grad_(False)
    return model.to(cfg.device)


def load_frames(cfg: Config):
    frames = []
    for d in cfg.frame_dirs:
        d = str(ROOT / d)
        paths = sorted(glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.png")))
        paths = paths[int(len(paths) * cfg.frame_start_frac):][::cfg.frame_stride][:cfg.frames_per_dir]
        if not paths:
            raise FileNotFoundError(f"no frames in {d}")
        seq = os.path.basename(os.path.dirname(d.rstrip("/"))) or os.path.basename(d)
        for p in paths:
            x = read_image(p, ImageReadMode.RGB).float().div(255.0)
            frames.append((seq, os.path.basename(p), x.to(cfg.device)))
    return frames


# ---------------------------------------------------------------------------
# Hooks and forward passes
# ---------------------------------------------------------------------------

class Taps:
    def __init__(self, model, layers):
        self.acts = {}
        body, where = model.backbone.body, tap_layers(model)
        self.handles = [getattr(body, where[n]).register_forward_hook(self._hook(n)) for n in layers]

    def _hook(self, name):
        def fn(module, inp, out):
            self.acts[name] = out
        return fn

    def remove(self):
        for h in self.handles:
            h.remove()


def backbone_pass(model, x):
    """Transform + ResNet body only. Enough for the sponge objective, much cheaper."""
    images, _ = model.transform([x])
    model.backbone.body(images.tensors)


def ped_scores(model, x, cfg):
    out = model([x])[0]
    return out["scores"][out["labels"] == cfg.ped_label]


def sponge_value(taps, cfg):
    return torch.stack([torch.tanh(cfg.sponge_beta * taps.acts[l].abs()).mean()
                        for l in cfg.layers]).mean()


def objective(name, model, taps, x, cfg):
    """Quantity each attack maximises."""
    if name == "sponge_l0":
        backbone_pass(model, x)
        return sponge_value(taps, cfg)
    s = ped_scores(model, x, cfg)
    if name == "evasion":
        return -s.sum()
    if name == "det_inflation":
        return torch.sigmoid(cfg.inflation_k * (s - cfg.det_thresh)).sum()
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------

def quantize(x):
    return torch.round(x.clamp(0, 1) * 255.0) / 255.0


def random_attack(x, eps, gen):
    sign = torch.randint(0, 2, x.shape, generator=gen, device=x.device).float() * 2 - 1
    return quantize(x + eps * sign)


def pgd_attack(name, model, taps, x, eps, cfg, gen):
    alpha = 2.5 * eps / cfg.pgd_steps
    best_val, best_x = -math.inf, None
    for _ in range(cfg.restarts):
        delta = (torch.rand(x.shape, generator=gen, device=x.device) * 2 - 1) * eps
        delta = (x + delta).clamp(0, 1) - x
        for _ in range(cfg.pgd_steps):
            delta.requires_grad_(True)
            obj = objective(name, model, taps, x + delta, cfg)
            if not obj.requires_grad:
                break
            (g,) = torch.autograd.grad(obj, delta, allow_unused=True)
            if g is None:
                break
            with torch.no_grad():
                delta = (delta + alpha * g.sign()).clamp(-eps, eps)
                delta = (x + delta).clamp(0, 1) - x
        with torch.no_grad():
            x_adv = quantize(x + delta.detach())
            val = objective(name, model, taps, x_adv, cfg).item()
        if val > best_val:
            best_val, best_x = val, x_adv
    return best_x


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def metric_keys(cfg):
    keys = ["n_dets", "score_sum_above", "sponge_obj"]
    for l in cfg.layers:
        keys += [f"{l}_rho_{t:g}" for t in cfg.rho_thresholds]
        keys += [f"{l}_svar", f"{l}_chent"]
    return keys


@torch.no_grad()
def measure(model, taps, x, cfg):
    s = ped_scores(model, x, cfg)          # full forward also fills the taps
    above = s[s > cfg.det_thresh]
    row = {"n_dets": int(above.numel()),
           "score_sum_above": float(above.sum()),
           "sponge_obj": float(sponge_value(taps, cfg))}
    for l in cfg.layers:
        a = taps.acts[l][0]                               # C,H,W
        for t in cfg.rho_thresholds:
            row[f"{l}_rho_{t:g}"] = float((a.abs() > t).float().mean())
        # spatial variance of the channel-mean map; entropy of the channel-mean distribution
        row[f"{l}_svar"] = float(a.mean(0).var())         # variance of channel-mean map over H,W
        cm = a.mean((1, 2))
        p = cm / cm.sum().clamp_min(1e-12)
        row[f"{l}_chent"] = float(-(p * p.clamp_min(1e-12).log()).sum())
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg: Config):
    os.makedirs(cfg.out_dir, exist_ok=True)
    frames = load_frames(cfg)
    print(f"{len(frames)} frames, {len(cfg.checkpoints)} arms, device={cfg.device}")

    fields = ["arm", "seq", "frame", "attack", "eps_255"] + metric_keys(cfg)
    csv_path = os.path.join(cfg.out_dir, "results.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()

        for arm, ckpt in cfg.checkpoints.items():
            model = load_model(ckpt, cfg)
            taps = Taps(model, cfg.layers)
            t0 = time.time()

            for fi, (seq, fname, x) in enumerate(frames):
                meta = {"arm": arm, "seq": seq, "frame": fname}
                writer.writerow({**meta, "attack": "clean", "eps_255": 0, **measure(model, taps, x, cfg)})

                for ai, attack in enumerate(cfg.attacks):
                    for ei, eps in enumerate(cfg.epsilons):
                        # identical seed across arms -> paired perturbation starts
                        gen = torch.Generator(device=cfg.device)
                        gen.manual_seed(cfg.seed * 1_000_003 + fi * 1009 + ai * 101 + ei)
                        if attack == "random":
                            x_adv = random_attack(x, eps, gen)
                        else:
                            x_adv = pgd_attack(attack, model, taps, x, eps, cfg, gen)
                        eps_255 = round(eps * 255)
                        writer.writerow({**meta, "attack": attack, "eps_255": eps_255,
                                         **measure(model, taps, x_adv, cfg)})
                        if cfg.save_adv:
                            d = os.path.join(cfg.out_dir, "adv", arm, attack, f"eps{eps_255}", seq)
                            os.makedirs(d, exist_ok=True)
                            save_image(x_adv, os.path.join(d, os.path.splitext(fname)[0] + ".png"))
                fh.flush()
                el = time.time() - t0
                print(f"  [{arm}] {fi + 1}/{len(frames)} frames, {el / 60:.1f} min, "
                      f"~{el / (fi + 1) * (len(frames) - fi - 1) / 60:.1f} min left for this arm")

            taps.remove()
            del model
            if cfg.device == "cuda":
                torch.cuda.empty_cache()

    print(f"wrote {csv_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 frames, 1 eps, 3 steps, 1 restart")
    ap.add_argument("--save_adv", action="store_true")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    start_log(ROOT, "pilot_smoke" if args.smoke else "pilot")

    cfg = Config()
    if args.smoke:
        cfg.frame_dirs = cfg.frame_dirs[:1]
        cfg.frames_per_dir, cfg.epsilons, cfg.pgd_steps, cfg.restarts = 2, (8 / 255,), 3, 1
        cfg.out_dir = "pilot_v2_smoke"
    if args.save_adv:
        cfg.save_adv = True
    if args.out_dir:
        cfg.out_dir = args.out_dir
    run(cfg)
