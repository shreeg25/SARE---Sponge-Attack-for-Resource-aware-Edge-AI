"""Admission control between the detector and the tracker. A guard decides,
per frame, which forwarded detections are admitted to the expensive stages
(ReID embedding, DeepSORT association, uplink). Everything not admitted costs
nothing downstream.

    admit(dets, tracker) -> indices of admitted detections
        dets:    N x 6 array (x1, y1, x2, y2, score, label), score-sorted or not
        tracker: the live DeepSort object (for guards that use track state)

Guards: NoGuard (no defence), FixedCap (the naive bound), SAREGuard (ours).
"""

import json

import numpy as np


class NoGuard:
    name = "none"

    def reset(self):
        pass

    def admit(self, dets, tracker):
        return np.arange(len(dets))


class FixedCap:
    """The naive fix from the literature: admit at most k detections per frame,
    highest score first. Bounds cost, but drops real objects in busy scenes."""

    def __init__(self, k):
        self.k, self.name = int(k), f"cap{int(k)}"

    def reset(self):
        pass

    def admit(self, dets, tracker):
        if len(dets) <= self.k:
            return np.arange(len(dets))
        return np.sort(np.argsort(-dets[:, 4])[: self.k])


class GeometryPrior:
    """Plausible box height for a given bottom-edge position (perspective), per class,
    fitted on train-split ground truth by scripts/fit_geometry_prior.py."""

    def __init__(self, path, slack=1.5):
        p = json.load(open(path))
        self.h, self.n_bins, self.slack = p["img_h"], p["n_bins"], slack
        self.bins = {int(k): np.array(v["bins"])[:, :2] for k, v in p["classes"].items()}

    def plausible(self, dets):
        y2 = np.clip(dets[:, 3] / self.h, 0, 1)
        hh = (dets[:, 3] - dets[:, 1]) / self.h
        b = np.minimum((y2 * self.n_bins).astype(int), self.n_bins - 1)
        ok = np.ones(len(dets), dtype=bool)
        for lab, tab in self.bins.items():
            m = dets[:, 5].astype(int) == lab
            lo, hi = tab[b[m], 0] / self.slack, tab[b[m], 1] * self.slack
            ok[m] = (hh[m] >= lo) & (hh[m] <= hi)
        return ok


def _iou(a, b):
    """a: N x 4, b: M x 4 (x1, y1, x2, y2) -> N x M IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


class SAREGuard:
    """Pipeline admission guard, per frame, with budget K:
      1. geometry: drop boxes the perspective prior finds implausible;
      2. if what is left fits in K, admit all of it (no attack suspected: identical
         to no guard on normal frames);
      3. otherwise, anchored first: each existing track (tentative or confirmed)
         gets at most one slot, for the detection best overlapping its Kalman-
         predicted box -- objects already being followed keep being followed;
      4. remaining slots go to newcomers chosen uniformly at random, not by score:
         scores are attacker-controlled, so ranking by them hands the slots to the
         most confident phantoms.
    Per-frame phantoms move from frame to frame, never match a predicted box, and
    churn; real objects, once admitted, stay anchored. Downstream cost is bounded
    by K ReID passes per frame whatever the attack -- the same bound as a fixed cap."""

    def __init__(self, k_max=10, anchor_iou=0.3, prior=None, seed=0):
        self.k, self.anchor_iou, self.prior, self.seed = int(k_max), anchor_iou, prior, seed
        self.name = f"sare{self.k}" + ("" if prior else "_nogeo")

    def reset(self):
        self.rng = np.random.default_rng(self.seed)

    @staticmethod
    def _predicted_boxes(tracker):
        boxes = []
        for t in tracker.tracker.tracks:
            if t.is_deleted():
                continue
            x, y, a, h = t.mean[:4] + t.mean[4:8]          # one constant-velocity step ahead
            w = a * h
            boxes.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
        return np.array(boxes, dtype=np.float64).reshape(-1, 4)

    def admit(self, dets, tracker):
        n = len(dets)
        if n == 0:
            return np.arange(0)
        ok = self.prior.plausible(dets) if self.prior is not None else np.ones(n, dtype=bool)
        cand = np.flatnonzero(ok)
        if len(cand) <= self.k:
            return cand
        pred = self._predicted_boxes(tracker)
        anchored = []
        if len(pred):
            iou = _iou(pred, dets[:, :4].astype(np.float64))
            iou[:, ~ok] = 0
            taken = np.zeros(n, dtype=bool)
            for ti in np.argsort(-iou.max(1)):                 # strongest matches first
                j = int(np.argmax(np.where(taken, -1, iou[ti])))
                if iou[ti, j] >= self.anchor_iou and not taken[j]:
                    taken[j] = True
                    anchored.append(j)
        anchored = anchored[: self.k]
        rest = np.setdiff1d(cand, anchored)
        free = self.k - len(anchored)
        newcomers = self.rng.choice(rest, size=min(free, len(rest)), replace=False) if free > 0 else []
        return np.sort(np.concatenate([np.array(anchored, dtype=int), np.asarray(newcomers, dtype=int)]))


def build_guard(spec, prior_path=None):
    """'none' | 'cap<k>' | 'sare<k>' | 'sare<k>_nogeo'   (e.g. cap10, sare10)"""
    if spec == "none":
        return NoGuard()
    if spec.startswith("cap"):
        return FixedCap(spec[3:])
    if spec.startswith("sare"):
        nogeo = spec.endswith("_nogeo")
        k = int(spec[4:].replace("_nogeo", "") or 10)
        prior = None if nogeo else GeometryPrior(prior_path)
        return SAREGuard(k_max=k, prior=prior)
    raise ValueError(f"unknown guard {spec}")
