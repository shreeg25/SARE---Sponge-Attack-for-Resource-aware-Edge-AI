"""Datasets. MOT17 was the debugging set; nuScenes CAM_FRONT is the study
dataset. Both return (image in [0,1], {"boxes", "labels"})."""

import glob
import json
import os
import random

import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image


class MOT17Det(Dataset):
    """Per-frame pedestrian detection from MOT17 gt.txt (label 1 = pedestrian).

    Split is temporal within each sequence: the first (1 - val_frac) of frames
    train, the rest validate. The pilot samples from the same tail, so pilot
    frames are never training frames.
    """

    def __init__(self, seq_dirs, split, val_frac=0.3, stride=1, min_vis=0.25, hflip=False):
        self.items, self.hflip = [], hflip
        for sd in seq_dirs:
            imgs = sorted(glob.glob(os.path.join(sd, "img1", "*.jpg")))
            gt_path = os.path.join(sd, "gt", "gt.txt")
            if not imgs:
                raise FileNotFoundError(f"no frames in {sd}/img1")
            if not os.path.isfile(gt_path):
                raise FileNotFoundError(f"missing {gt_path}")
            gt = self._read_gt(gt_path, min_vis)
            cut = int(len(imgs) * (1 - val_frac))
            idx = range(0, cut, stride) if split == "train" else range(cut, len(imgs), stride)
            for i in idx:
                frame = int(os.path.splitext(os.path.basename(imgs[i]))[0])
                boxes = gt.get(frame, [])
                if split == "train" and not boxes:
                    continue
                self.items.append((imgs[i], torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)))

    @staticmethod
    def _read_gt(path, min_vis):
        # frame, id, left, top, w, h, consider_flag, class, visibility
        out = {}
        with open(path) as fh:
            for line in fh:
                f, _, x, y, w, h, flag, cls, vis = line.strip().split(",")[:9]
                if int(float(flag)) != 1 or int(float(cls)) != 1 or float(vis) < min_vis:
                    continue
                x, y, w, h = float(x), float(y), float(w), float(h)
                out.setdefault(int(f), []).append([x, y, x + w, y + h])
        return out

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, boxes = self.items[i]
        img = read_image(path, ImageReadMode.RGB).float().div(255.0)
        H, W = img.shape[1:]
        boxes = boxes.clone()
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, W)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, H)
        boxes = boxes[(boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)]
        if self.hflip and random.random() < 0.5:
            img = img.flip(-1)
            boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
        return img, {"boxes": boxes, "labels": torch.ones(len(boxes), dtype=torch.int64)}


class NuScenes2D(Dataset):
    """CAM_FRONT keyframes from the index written by scripts/prepare_nuscenes.py.
    Labels: 1 pedestrian, 2 vehicle, 3 cyclist."""

    def __init__(self, index_path, data_root, split, conditions, stride=1, hflip=False):
        with open(index_path) as fh:
            recs = json.load(fh)["records"]
        recs = [r for r in recs if r["split"] == split and r["condition"] in conditions]
        if split == "train":
            recs = [r for r in recs if r["boxes"]]
        if not recs:
            raise ValueError(f"no frames for split={split} conditions={conditions} in {index_path}")
        self.items, self.root, self.hflip = recs[::stride], data_root, hflip

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        r = self.items[i]
        img = read_image(os.path.join(self.root, r["file"]), ImageReadMode.RGB).float().div(255.0)
        W = img.shape[2]
        boxes = torch.tensor(r["boxes"], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(r["labels"], dtype=torch.int64)
        if self.hflip and random.random() < 0.5:
            img = img.flip(-1)
            boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
        return img, {"boxes": boxes, "labels": labels}


def collate(batch):
    return tuple(zip(*batch))


def build_datasets(cfg):
    name = cfg["dataset"]
    if name == "mot17":
        c = cfg["mot17"]
        dirs = [os.path.join(c["root"], s) for s in c["sequences"]]
        train = MOT17Det(dirs, "train", c["val_frac"], c["train_stride"], c["min_visibility"], hflip=True)
        val = MOT17Det(dirs, "val", c["val_frac"], c["val_stride"], c["min_visibility"], hflip=False)
        return train, val
    if name == "nuscenes":
        c = cfg["nuscenes"]
        train = NuScenes2D(c["index"], c["root"], "train", ["day"], c["train_stride"], hflip=True)
        val = NuScenes2D(c["index"], c["root"], "select", ["day"], c["val_stride"], hflip=False)
        return train, val
    raise ValueError(f"unknown dataset {name}")


TEST_CONDITIONS = ("day", "night", "rain", "night_rain")


def build_test_sets(cfg):
    """Held-out test sets, one per condition, never used for training or
    checkpoint selection. Conditions absent from the index are skipped.
    MOT17 has no separate test split, so it returns {}."""
    if cfg["dataset"] != "nuscenes":
        return {}
    c, out = cfg["nuscenes"], {}
    for cond in TEST_CONDITIONS:
        try:
            out[cond] = NuScenes2D(c["index"], c["root"], "test", [cond], c.get("test_stride", 1))
        except ValueError:
            pass
    return out
