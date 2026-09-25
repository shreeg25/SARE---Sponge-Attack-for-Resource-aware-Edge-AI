"""PGD on the detector's own training loss, used for adversarial training
and for the robustness check at the end of training.

Budgets for L2 and L1 are given per pixel so they scale with image size:
    eps_l2 = l2_rms  * sqrt(n)    (per-pixel RMS perturbation)
    eps_l1 = l1_mean * n          (per-pixel mean absolute perturbation)
where n = C*H*W of the image being attacked.
"""

import math

import torch


OBJECTIVES = {
    "all": None,                             # full detection loss (standard PGD adversarial training)
    "objectness": ("loss_objectness",),      # RPN objectness only: the Underload-style proxy
}


def detection_loss(model, images, targets, keys=None):
    losses = model(images, targets)
    return sum(v for k, v in losses.items() if keys is None or k in keys)


def background_mask(x, boxes, ratio):
    """1 on background, 0 inside each ground-truth box scaled by `ratio` about its
    centre (Underload-style background-attentive masking; ratio 0.9 was their best)."""
    m = torch.ones_like(x[:1])
    H, W = x.shape[1:]
    for x1, y1, x2, y2 in boxes.tolist():
        cx, cy, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * ratio / 2, (y2 - y1) * ratio / 2
        m[:, max(0, int(cy - hh)):min(H, int(cy + hh) + 1), max(0, int(cx - hw)):min(W, int(cx + hw) + 1)] = 0
    return m


def budget(norm, x, adv_cfg):
    n = x.numel()
    if norm == "linf":
        return adv_cfg["linf_eps"]
    if norm == "l2":
        return adv_cfg["l2_rms"] * math.sqrt(n)
    if norm == "l1":
        return adv_cfg["l1_mean"] * n
    raise ValueError(norm)


def project_l1(v, eps):
    """Euclidean projection onto the L1 ball (Duchi et al., 2008)."""
    flat = v.flatten()
    u = flat.abs()
    if float(u.sum()) <= eps:
        return v
    s, _ = torch.sort(u, descending=True)
    cs = s.double().cumsum(0)
    idx = torch.arange(1, s.numel() + 1, device=v.device, dtype=torch.float64)
    rho = int(torch.nonzero(s.double() - (cs - eps) / idx > 0)[-1]) + 1
    theta = float((cs[rho - 1] - eps) / rho)
    return (flat.sign() * (u - theta).clamp_min(0)).view_as(v)


def _step(d, g, norm, eps, steps, sparsity):
    alpha = 2.5 * eps / steps
    if norm == "linf":
        return (d + alpha * g.sign()).clamp(-eps, eps)
    if norm == "l2":
        d = d + alpha * g / (g.norm() + 1e-12)
        return d * min(1.0, eps / (float(d.norm()) + 1e-12))
    if norm == "l1":
        # steepest L1 ascent restricted to the top (1 - sparsity) coordinates
        a = g.abs().flatten()
        k = max(1, int(a.numel() * (1 - sparsity)))
        thr = a.topk(k).values[-1]
        e = g.sign() * (g.abs() >= thr)
        e = e / e.abs().sum().clamp_min(1e-12)
        return project_l1(d + alpha * e, eps)
    raise ValueError(norm)


def pgd(model, images, targets, norm, adv_cfg, steps, amp=False, objective="all", bg_mask_ratio=None):
    """Untargeted PGD maximising the detection loss. images: list of CHW in [0,1].
    Returns detached adversarial images; restores the model's train/eval mode.
    `objective` and `bg_mask_ratio` are training-time options only; evaluation
    always calls with the defaults (full loss, whole image)."""
    was_training = model.training
    model.train()
    dev_type = images[0].device.type
    eps = [budget(norm, x, adv_cfg) for x in images]
    sparsity = adv_cfg.get("l1_sparsity", 0.99)
    keys = OBJECTIVES[objective]
    masks = [background_mask(x, t["boxes"], bg_mask_ratio) if bg_mask_ratio else None
             for x, t in zip(images, targets)]

    deltas = []
    for x, e, m in zip(images, eps, masks):
        d = (torch.rand_like(x) * 2 - 1) * e if norm == "linf" else torch.zeros_like(x)
        if m is not None:
            d = d * m
        deltas.append(((x + d).clamp(0, 1) - x).detach())

    for _ in range(steps):
        for d in deltas:
            d.requires_grad_(True)
        with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=amp):
            loss = detection_loss(model, [x + d for x, d in zip(images, deltas)], targets, keys)
        grads = torch.autograd.grad(loss, deltas, allow_unused=True)
        with torch.no_grad():
            new = []
            for x, d, g, e, m in zip(images, deltas, grads, eps, masks):
                g = torch.zeros_like(d) if g is None else g.float()
                if m is not None:
                    g = g * m
                d = _step(d, g, norm, e, steps, sparsity)
                if m is not None:
                    d = d * m
                new.append(((x + d).clamp(0, 1) - x).detach())
            deltas = new

    model.train(was_training)
    return [(x + d).detach() for x, d in zip(images, deltas)]
