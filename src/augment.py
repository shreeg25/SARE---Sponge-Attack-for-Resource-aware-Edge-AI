"""
Condition augmentation for training on daytime frames only.

night_aug follows 2PCNet's NightAug (Kennerley et al., CVPR 2023): random
brightness, gamma, contrast, Gaussian noise, Gaussian blur and glare, each
applied with probability 0.5 and random strength, and after each step random
image regions are reverted so lighting ends up uneven, as it is at night.
It is darkening-biased: at least one of brightness / gamma is always applied.

rain_aug covers what nuScenes rain frames actually show more than visible
streaks: darker exposure, haze-like contrast loss, blur, plus rain streaks
with probability 0.5.

Photometric only (boxes never move). Works on H x W x 3 float32 arrays in
[0, 1]; numpy + scipy so it runs inside DataLoader workers without the GPU.
Test frames are never augmented: night/rain evaluation stays zero-shot.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def _revert_regions(new, old, rng, p=0.3, max_regions=3):
    """Put back random rectangles of the pre-step image (uneven lighting)."""
    if rng.random() >= p:
        return new
    H, W = new.shape[:2]
    for _ in range(int(rng.integers(1, max_regions + 1))):
        h, w = int(H * rng.uniform(0.15, 0.45)), int(W * rng.uniform(0.15, 0.45))
        y, x = int(rng.integers(0, H - h + 1)), int(rng.integers(0, W - w + 1))
        new[y:y + h, x:x + w] = old[y:y + h, x:x + w]
    return new


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


def night_aug(x, rng):
    steps = [lambda z: _brightness(z, rng, 0.2, 0.7), lambda z: _gamma(z, rng),
             lambda z: _contrast(z, rng), lambda z: _noise(z, rng),
             lambda z: _blur(z, rng), lambda z: _glare(z, rng)]
    use = rng.random(len(steps)) < 0.5
    if not (use[0] or use[1]):                 # always darken
        use[int(rng.integers(0, 2))] = True
    for f, on in zip(steps, use):
        if on:
            old = x
            x = _revert_regions(np.clip(f(x.copy()), 0, 1), old, rng)
    return x


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
    x = _brightness(x, rng, 0.6, 0.9)
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
