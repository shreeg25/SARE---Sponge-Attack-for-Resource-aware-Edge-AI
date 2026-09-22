"""Detector construction, checkpoint loading, and backbone tap points."""

from typing import Dict, Optional

import torch
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d

ARCHS = {
    "mobilenet_v3": (fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1),
    "resnet50": (fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights.COCO_V1),
}


def build_detector(arch: str, num_classes: int, pretrained: bool = True,
                   min_size: int = 800, max_size: int = 1333, trainable_layers: int = 3):
    """COCO-pretrained detector with a fresh box head (training), or an empty
    shell of the same shape (for loading a checkpoint)."""
    fn, weights = ARCHS[arch]
    if pretrained:
        model = fn(weights=weights, trainable_backbone_layers=trainable_layers,
                   min_size=min_size, max_size=max_size)
        in_f = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes)
    else:
        model = fn(weights=None, weights_backbone=None, num_classes=num_classes,
                   min_size=min_size, max_size=max_size)
    return model


def load_detector(path: str, device: str = "cpu", fallback: Optional[dict] = None):
    """Loads a SARE checkpoint ({"model", "meta"}) or a bare state_dict.
    Architecture comes from the checkpoint's meta; `fallback` covers old
    checkpoints that have none."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, torch.nn.Module):
        return ckpt.eval().to(device), {}

    meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
    sd = ckpt
    for key in ("model", "state_dict", "model_state_dict"):
        if isinstance(sd, dict) and isinstance(sd.get(key), dict):
            sd = sd[key]
            break
    sd = {k.removeprefix("module."): v for k, v in sd.items()}

    m = {**(fallback or {}), **meta}
    model = build_detector(m["arch"], m["num_classes"], pretrained=False,
                           min_size=m.get("min_size", 800), max_size=m.get("max_size", 1333))
    # Training used FrozenBatchNorm (pretrained path); the empty shell uses
    # BatchNorm2d. Same buffers, same eval-mode maths, but no
    # num_batches_tracked in the checkpoint -- that is the only allowed gap.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    freeze_bn(model)   # PGD on the training loss puts the model in train() mode; stats must not move
    return model.eval().to(device), m


@torch.no_grad()
def freeze_bn(module):
    """Swap every BatchNorm2d for FrozenBatchNorm2d with the same statistics,
    matching how the model was trained."""
    for name, child in module.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            f = FrozenBatchNorm2d(child.num_features, eps=child.eps)
            f.weight.copy_(child.weight)
            f.bias.copy_(child.bias)
            f.running_mean.copy_(child.running_mean)
            f.running_var.copy_(child.running_var)
            setattr(module, name, f)
        else:
            freeze_bn(child)
    return module


def tap_layers(model) -> Dict[str, str]:
    """Maps stride-8/16/32 stage aliases to module names inside backbone.body,
    so pilot metrics are named the same way for ResNet and MobileNet."""
    body = model.backbone.body
    children = list(body.named_children())
    names = [n for n, _ in children]
    if "layer4" in names:
        return {"stage2": "layer2", "stage3": "layer3", "stage4": "layer4"}
    # MobileNetV3: a stage ends right before each stride-2 block (_is_cn),
    # and the last child ends the final stage.
    ends = [children[i - 1][0] for i, (_, m) in enumerate(children)
            if i > 0 and getattr(m, "_is_cn", False)]
    ends.append(children[-1][0])
    if len(ends) < 3:
        raise RuntimeError(f"could not find stage boundaries in backbone.body: {names}")
    return {f"stage{i + 2}": n for i, n in enumerate(ends[-3:])}
