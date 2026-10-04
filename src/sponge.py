"""
Resource (sponge) attacks on a torchvision Faster R-CNN.

inflate      Per-frame PGD that maximises the number of detections the edge node
             forwards. Objective (Phantom-Sponge-style, adapted to two-stage):
                 L = sum_{i in NMS-kept} sigmoid(k (s_i - tau))       (survivor count)
                     * (1 - lam * overlap)                          (spread them out)
             where overlap is the score-weighted mean IoU among the top candidates,
             so phantoms do not overlap and NMS keeps them. Candidates are read with
             NMS switched off inside the box head; NMS is then re-applied by us to
             know which ones would survive, exactly as in deployment.

sponge_act   Shumailov et al. activation-density sponge: maximise the mean
             tanh(beta |a|) over backbone stages. Kept as the classic sponge
             baseline; on a dense CPU it changes no FLOPs, which the paper reports.

uap          Universal perturbation (Phantom Sponges, Shapira et al. WACV 2023):
             one delta, optimised offline on select frames with the inflate
             objective (scripts/train_uap.py), added to every test frame. Its
             phantoms sit at the same image positions frame after frame, so they
             persist into tracks: the natural stress test for a tracking guard.

All attacks use the same norm budgets as the accuracy attacks (src/attacks.py).
"""

import torch
from torchvision.ops import batched_nms, box_iou

from src.attacks import _step, budget
from src.models import tap_layers


class Unsuppressed:
    """Inside this context the box head returns every candidate: NMS off, very low
    score floor, many detections. Restores the deployment settings on exit."""

    def __init__(self, model, max_dets=1000, score_floor=1e-3):
        self.rh, self.max_dets, self.floor = model.roi_heads, max_dets, score_floor

    def __enter__(self):
        self.saved = (self.rh.nms_thresh, self.rh.score_thresh, self.rh.detections_per_img)
        self.rh.nms_thresh, self.rh.score_thresh, self.rh.detections_per_img = 1.0, self.floor, self.max_dets
        return self

    def __exit__(self, *exc):
        self.rh.nms_thresh, self.rh.score_thresh, self.rh.detections_per_img = self.saved


def inflation_objective(out, nms_iou=0.5, tau=0.4, k=20.0, lam=0.5, topk=300):
    """out: one image's unsuppressed detections. Higher = more forwarded boxes."""
    s, b, lab = out["scores"], out["boxes"], out["labels"]
    if s.numel() == 0:
        return s.sum()
    w = torch.sigmoid(k * (s - tau))
    keep = batched_nms(b.detach(), s.detach(), lab, nms_iou)
    count = w[keep].sum()
    top = torch.topk(s, min(topk, s.numel())).indices
    bb, ww = b[top], w[top]
    iou = box_iou(bb, bb)
    iou = iou - torch.diag_embed(torch.diagonal(iou))
    pair = ww[:, None] * ww[None, :]
    overlap = (pair * iou).sum() / pair.sum().clamp_min(1e-6)
    return count * (1 - lam * overlap)


def activation_objective(model, x, beta=20.0):
    """Mean tanh(beta |a|) over the stride-8/16/32 backbone stages."""
    body, where = model.backbone.body, tap_layers(model)
    acts, hooks = [], []
    for name in where.values():
        hooks.append(getattr(body, name).register_forward_hook(lambda m, i, o: acts.append(o)))
    try:
        images, _ = model.transform([x])
        body(images.tensors)
    finally:
        for h in hooks:
            h.remove()
    return torch.stack([torch.tanh(beta * a.abs()).mean() for a in acts]).mean()


def sponge_pgd(model, x, objective, adv_cfg, steps, norm="linf", amp=False, restarts=1):
    """Maximise `objective` ('inflate' | 'inflate_hi' | 'sponge_act') within the norm budget.
    x: one CHW image in [0, 1]. Returns the best 8-bit-quantised adversarial image."""
    was_training = model.training
    model.eval()
    eps = budget(norm, x, adv_cfg)
    sparsity = adv_cfg.get("l1_sparsity", 0.99)

    def value(z):
        with torch.autocast(device_type=z.device.type, dtype=torch.bfloat16, enabled=amp):
            if objective in ("inflate", "inflate_hi"):
                # inflate_hi = adaptive attacker aware of a score-aware guard: phantoms pushed past 0.8
                with Unsuppressed(model):
                    return inflation_objective(model([z])[0], tau=0.8 if objective == "inflate_hi" else 0.4)
            if objective == "sponge_act":
                return activation_objective(model, z)
        raise ValueError(objective)

    best, best_val = None, -float("inf")
    for _ in range(restarts):
        d = (torch.rand_like(x) * 2 - 1) * eps if norm == "linf" else torch.zeros_like(x)
        d = ((x + d).clamp(0, 1) - x).detach()
        for _ in range(steps):
            d.requires_grad_(True)
            v = value(x + d)
            if not v.requires_grad:
                break
            (g,) = torch.autograd.grad(v, d, allow_unused=True)
            if g is None:
                break
            with torch.no_grad():
                d = ((x + _step(d, g.float(), norm, eps, steps, sparsity)).clamp(0, 1) - x).detach()
        with torch.no_grad():
            xa = torch.round((x + d).clamp(0, 1) * 255) / 255
            v = float(value(xa))
        if v > best_val:
            best, best_val = xa, v
    model.train(was_training)
    return best


def apply_uap(x, delta):
    """Add a universal perturbation (resized if the frame size differs), 8-bit quantised."""
    if delta.shape[-2:] != x.shape[-2:]:
        delta = torch.nn.functional.interpolate(delta[None], size=x.shape[-2:], mode="bilinear",
                                                align_corners=False)[0]
    return torch.round((x + delta.to(x)).clamp(0, 1) * 255) / 255
