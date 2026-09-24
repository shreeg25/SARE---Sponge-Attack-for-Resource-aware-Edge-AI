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


@torch.no_grad()
def detection_report(preds, gts, num_classes, class_names=None, score_thr=0.4, iou_thr=0.5):
    """Everything worth reporting for a detector on one set of frames.

    AP50 is threshold-free (the whole ranked list). Precision, recall, F1 and
    false positives per frame are at the operating point: detections scoring
    above `score_thr` (0.4, the edge node's forwarding threshold), matched
    greedily by score to the best unmatched ground truth at IoU >= `iou_thr`.
    """
    names = class_names or {c: str(c) for c in range(1, num_classes)}
    rep = {"ap50": ap50(preds, gts, num_classes)}
    for c in range(1, num_classes):
        rep[f"ap50_{names[c]}"] = _ap50_single(preds, gts, c)
    tp = fp = fn = ndet = 0
    for p, g in zip(preds, gts):
        for c in range(1, num_classes):
            gb = g["boxes"][g["labels"] == c]
            m = (p["labels"] == c) & (p["scores"] > score_thr)
            pb, ps = p["boxes"][m], p["scores"][m]
            pb = pb[ps.argsort(descending=True)]
            ndet += len(pb)
            matched = torch.zeros(len(gb), dtype=torch.bool)
            if len(gb) and len(pb):
                iou = box_iou(pb, gb)
                for j in range(len(pb)):
                    row = iou[j].clone()
                    row[matched] = -1.0
                    k = int(row.argmax())
                    if row[k] >= iou_thr:
                        matched[k] = True
                        tp += 1
                    else:
                        fp += 1
            else:
                fp += len(pb)
            fn += int((~matched).sum())
    n = max(len(preds), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    rep.update(precision=prec, recall=rec, f1=2 * prec * rec / max(prec + rec, 1e-12),
               tp=tp, fp=fp, fn=fn, dets_per_frame=ndet / n, fp_per_frame=fp / n, frames=len(preds))
    return rep


def _ap50_single(preds, gts, c):
    """AP50 for class c alone: relabel c -> 1, drop everything else."""
    def keep(d):
        m = d["labels"] == c
        out = {k: v[m] for k, v in d.items() if k in ("boxes", "scores")}
        out["labels"] = torch.ones(int(m.sum()), dtype=torch.int64)
        return out
    return ap50([keep(p) for p in preds], [keep(g) for g in gts], 2)
