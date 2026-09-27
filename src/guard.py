"""Admission control between the detector and the tracker. A guard decides,
per frame, which forwarded detections are admitted to the expensive stages
(ReID embedding, DeepSORT association, uplink). Everything not admitted costs
nothing downstream.

    admit(dets, tracker) -> indices of admitted detections
        dets:    N x 6 array (x1, y1, x2, y2, score, label), score-sorted or not
        tracker: the live DeepSort object (for guards that use track state)

The SARE guard (probation + adaptive budget + geometry prior) is added here
next; these two are the reference points it is compared against.
"""

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


def build_guard(spec):
    """'none' | 'cap<k>' (e.g. cap10)"""
    if spec == "none":
        return NoGuard()
    if spec.startswith("cap"):
        return FixedCap(spec[3:])
    raise ValueError(f"unknown guard {spec}")
