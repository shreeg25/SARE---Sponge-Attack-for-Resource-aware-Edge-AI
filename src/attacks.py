"""PGD on the detector's own training loss, used for adversarial training
and for the robustness check at the end of training.

Budgets for L2 and L1 are given per pixel so they scale with image size:
    eps_l2 = l2_rms  * sqrt(n)    (per-pixel RMS perturbation)
    eps_l1 = l1_mean * n          (per-pixel mean absolute perturbation)
where n = C*H*W of the image being attacked.
"""

import math

import torch


def detection_loss(model, images, targets):
    return sum(model(images, targets).values())


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


def pgd(model, images, targets, norm, adv_cfg, steps, amp=False):
    """Untargeted PGD maximising the detection loss. images: list of CHW in [0,1].
    Returns detached adversarial images; restores the model's train/eval mode."""
    was_training = model.training
    model.train()
    dev_type = images[0].device.type
    eps = [budget(norm, x, adv_cfg) for x in images]
    sparsity = adv_cfg.get("l1_sparsity", 0.99)

    deltas = []
    for x, e in zip(images, eps):
        d = (torch.rand_like(x) * 2 - 1) * e if norm == "linf" else torch.zeros_like(x)
        deltas.append(((x + d).clamp(0, 1) - x).detach())

    for _ in range(steps):
        for d in deltas:
            d.requires_grad_(True)
        with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=amp):
            loss = detection_loss(model, [x + d for x, d in zip(images, deltas)], targets)
        grads = torch.autograd.grad(loss, deltas)
        with torch.no_grad():
            deltas = [((x + _step(d, g.float(), norm, e, steps, sparsity)).clamp(0, 1) - x).detach()
                      for x, d, g, e in zip(images, deltas, grads, eps)]

    model.train(was_training)
    return [(x + d).detach() for x, d in zip(images, deltas)]
