"""Save augmented versions of a few real training frames, side by side with a
real night and a real rain test frame, so you can check by eye that the
augmentation looks like the target conditions.

    python scripts/preview_aug.py            -> results/aug_preview/*.jpg
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from src.augment import night_aug, rain_aug

index = json.load(open(ROOT / "data" / "nuscenes_2d.json"))["records"]
root = ROOT / "data" / "nuscenes"
out = ROOT / "results" / "aug_preview"
out.mkdir(parents=True, exist_ok=True)
load = lambda r: np.asarray(Image.open(root / r["file"]).convert("RGB"), dtype=np.float32) / 255
save = lambda a, name: Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((800, 450)).save(out / name, quality=85)

train = [r for r in index if r["split"] == "train"][::500][:4]
rng = np.random.default_rng(0)
for k, r in enumerate(train):
    x = load(r)
    save(x, f"{k}_day_original.jpg")
    save(night_aug(x.copy(), rng), f"{k}_day_to_night.jpg")
    save(rain_aug(x.copy(), rng), f"{k}_day_to_rain.jpg")
for cond in ("night", "rain"):
    r = next(r for r in index if r["split"] == "test" and r["condition"] == cond)
    save(load(r), f"real_{cond}.jpg")
print(f"wrote previews to {out}")
