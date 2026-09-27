"""Appearance descriptor used in place of DeepSORT's ReID network: ImageNet
MobileNetV3-Small, global-average-pooled (576-d), L2-normalised, on 128x128
crops. Same role as a ReID net (one CNN pass per admitted detection), small
enough to run on a Raspberry Pi 5 CPU."""

import torch
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.ops import roi_align

IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
EMB_DIM = 576


class Embedder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.features, self.pool = m.features, torch.nn.AdaptiveAvgPool2d(1)

    @torch.no_grad()
    def forward(self, crops):                       # N,3,128,128 in [0,1]
        x = (crops - IMNET_MEAN.to(crops)) / IMNET_STD.to(crops)
        return F.normalize(self.pool(self.features(x)).flatten(1), dim=1)


def crops_from(img, boxes, size=128):
    """img: 3,H,W in [0,1]; boxes: N,4 xyxy in pixels -> N,3,size,size."""
    return roi_align(img[None], [boxes], output_size=(size, size), spatial_scale=1.0, aligned=True)
