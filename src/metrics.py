"""Detection evaluation: AP at IoU 0.5, averaged over foreground classes.
Greedy matching by score to the best unmatched GT (COCO-style), all-point
interpolated precision. Enough to verify training and hardening; the paper's
final numbers can go through pycocotools if reviewers want COCO mAP."""

import numpy as np
import torch
from torchvision.ops import box_iou


@torch.no_grad()
def ap50(preds, gts, num_classes, iou_thr=0.5):
    aps = []
    for c in range(1, num_classes):
        scores, tps, npos = [], [], 0
        for p, g in zip(preds, gts):
            gb = g["boxes"][g["labels"] == c]
            npos += len(gb)
            m = p["labels"] == c
            pb, ps = p["boxes"][m], p["scores"][m]
            order = ps.argsort(descending=True)
            pb, ps = pb[order], ps[order]
            matched = torch.zeros(len(gb), dtype=torch.bool)
            iou = box_iou(pb, gb) if len(gb) and len(pb) else None
            for j in range(len(pb)):
                tp = 0
                if iou is not None:
                    row = iou[j].clone()
                    row[matched] = -1.0
                    k = int(row.argmax())
                    if row[k] >= iou_thr:
                        matched[k] = True
                        tp = 1
                scores.append(float(ps[j]))
                tps.append(tp)
        if npos == 0:
            continue
        if not scores:
            aps.append(0.0)
            continue
        order = np.argsort(-np.asarray(scores))
        tp = np.cumsum(np.asarray(tps)[order])
        fp = np.cumsum(1 - np.asarray(tps)[order])
        rec = tp / npos
        prec = tp / np.maximum(tp + fp, 1e-12)
        mrec = np.concatenate([[0.0], rec, [1.0]])
        mpre = np.concatenate([[0.0], prec, [0.0]])
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        aps.append(float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])))
    return float(np.mean(aps)) if aps else float("nan")
