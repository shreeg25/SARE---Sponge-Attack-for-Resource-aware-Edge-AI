"""
Condition augmentation for training on daytime frames only.

night_aug builds on 2PCNet's NightAug (Kennerley et al., CVPR 2023). Always:
a dark tone curve, sky darkening (bright pixels in the upper frame), and
desaturation with a lamp tint. Then, each with probability 0.5: soft lamp-light
pools, glare, sensor noise and blur, with NightAug's random partial reverts
(soft-edged, so no rectangle artefacts).

rain_aug covers what nuScenes rain frames actually show more than visible
streaks: darker exposure, haze-like contrast loss, blur, plus rain streaks
with probability 0.5.

Photometric only (boxes never move). Works on H x W x 3 float32 arrays in
[0, 1]; numpy + scipy so it runs inside DataLoader workers without the GPU.
Test frames are never augmented: night/rain evaluation stays zero-shot.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def _soft_mask(H, W, rng, max_regions=3):
    """Soft-edged random blobs in [0, 1] (built at 1/8 res, blurred, upsampled):
    no hard edges a detector could learn as a cue."""
    h8, w8 = max(1, H // 8), max(1, W // 8)
    m = np.zeros((h8, w8), dtype=np.float32)
    for _ in range(int(rng.integers(1, max_regions + 1))):
        h, w = max(1, int(h8 * rng.uniform(0.15, 0.45))), max(1, int(w8 * rng.uniform(0.15, 0.45)))
        y, x = int(rng.integers(0, h8 - h + 1)), int(rng.integers(0, w8 - w + 1))
        m[y:y + h, x:x + w] = 1.0
    m = gaussian_filter(m, sigma=0.12 * min(h8, w8))
    m = m / max(m.max(), 1e-6)
    m = np.repeat(np.repeat(m, 8, 0), 8, 1)
    m = np.pad(m, ((0, max(0, H - m.shape[0])), (0, max(0, W - m.shape[1]))), mode="edge")[:H, :W]
    return gaussian_filter(m, sigma=4)[..., None]


def _revert_regions(new, old, rng, p=0.3, max_regions=3):
    """Partially restore soft-edged patches of the pre-step image (NightAug's
    'ignore the augmentation in random regions')."""
    if rng.random() >= p:
        return new
    m = _soft_mask(*new.shape[:2], rng, max_regions) * rng.uniform(0.5, 1.0)
    return new * (1 - m) + old * m


def _brightness(x, rng, lo, hi):
    return x * rng.uniform(lo, hi)


def _gamma(x, rng):
    return np.power(x, rng.uniform(1.3, 2.5))


def _contrast(x, rng):
    m = x.mean()
    return m + (x - m) * rng.uniform(0.5, 0.9)


def _noise(x, rng):
    return x + rng.normal(0.0, rng.uniform(0.01, 0.04), x.shape).astype(np.float32)


def _blur(x, rng, lo=0.5, hi=1.5):
    s = rng.uniform(lo, hi)
    return gaussian_filter(x, sigma=(s, s, 0))


def _glare(x, rng):
    """Street lamps / headlights: warm Gaussian blobs added to the image."""
    H, W = x.shape[:2]
    yy, xx = np.ogrid[:H, :W]
    for _ in range(int(rng.integers(1, 5))):
        cy, cx = rng.uniform(0.1, 0.9) * H, rng.uniform(0, 1) * W
        r = rng.uniform(0.02, 0.08) * W
        g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r)).astype(np.float32)
        color = np.array([1.0, rng.uniform(0.8, 1.0), rng.uniform(0.5, 0.9)], dtype=np.float32)
        x = x + rng.uniform(0.3, 0.8) * g[..., None] * color
    return x


def _night_tone(x, rng):
    """Always-on night look: dark tone curve, sky darkening, desaturation, tint."""
    H, W = x.shape[:2]
    lum = x.mean(-1, keepdims=True)
    # bright pixels in the upper part of the frame are sky: push them far down
    y = (np.arange(H, dtype=np.float32) / H)[:, None, None]
    upper = np.clip((0.6 - y) / 0.3, 0, 1)
    bright = 1 / (1 + np.exp(-(lum - 0.6) / 0.05))
    sky = upper * bright
    x = x * (1 - sky * (1 - rng.uniform(0.05, 0.2)))
    # dark tone curve: shadows fall off faster than highlights
    x = rng.uniform(0.35, 1.0) * np.power(np.clip(x, 0, 1), rng.uniform(1.2, 2.0))   # calibrated to real nuScenes night
    # desaturate and tint (sodium orange, LED blue, or neutral)
    g = x.mean(-1, keepdims=True)
    x = g + (x - g) * rng.uniform(0.3, 0.7)
    tint = [np.array([1.0, 0.82, 0.55]), np.array([0.82, 0.9, 1.0]), np.ones(3)][int(rng.integers(0, 3))]
    return x * tint.astype(np.float32)


def _light_pools(x, rng):
    """Street-lamp pools: brighten soft regions of the (already dark) image."""
    m = _soft_mask(*x.shape[:2], rng, max_regions=4)
    return x * (1 + rng.uniform(0.8, 2.5) * m)


def night_aug(x, rng):
    x = _night_tone(x, rng)
    steps = [lambda z: _light_pools(z, rng), lambda z: _glare(z, rng),
             lambda z: _noise(z, rng), lambda z: _blur(z, rng)]
    for f in steps:
        if rng.random() < 0.5:
            old = x
            x = _revert_regions(np.clip(f(x.copy()), 0, 1), old, rng)
    return np.clip(x, 0, 1)


def _streaks(x, rng):
    """Rain streaks: sparse drops smeared along a near-vertical line, screen-blended."""
    H, W = x.shape[:2]
    drops = (rng.random((H, W)) < rng.uniform(0.001, 0.004)) * rng.uniform(0.5, 1.0, (H, W))
    L = int(rng.integers(20, 50))
    ang = np.deg2rad(rng.uniform(-20, 20))
    layer = np.zeros((H, W), dtype=np.float32)
    for t in range(L):
        dy, dx = int(round(t * np.cos(ang))), int(round(t * np.sin(ang)))
        layer = np.maximum(layer, np.roll(np.roll(drops, dy, 0), dx, 1))
    layer = gaussian_filter(layer, 0.7)
    a = rng.uniform(0.3, 0.7) * np.clip(layer / max(layer.max(), 1e-6), 0, 1)[..., None]
    return 1 - (1 - x) * (1 - a)


def rain_aug(x, rng):
    x = _brightness(x, rng, 0.85, 1.0)          # nuScenes rain is not darker than day (auto-exposure)
    t, A = rng.uniform(0.7, 0.95), rng.uniform(0.5, 0.8)
    x = x * t + A * (1 - t)                          # haze-like contrast loss
    if rng.random() < 0.5:
        x = _streaks(x, rng)
    if rng.random() < 0.5:
        x = _blur(x, rng, 0.5, 1.2)
    return np.clip(x, 0, 1)


def condition_aug(x, rng, night_p=0.3, rain_p=0.2):
    """Returns (augmented image, 'night' | 'rain' | 'none')."""
    r = rng.random()
    if r < night_p:
        return np.clip(night_aug(x, rng), 0, 1).astype(np.float32), "night"
    if r < night_p + rain_p:
        return np.clip(rain_aug(x, rng), 0, 1).astype(np.float32), "rain"
    return x, "none"
