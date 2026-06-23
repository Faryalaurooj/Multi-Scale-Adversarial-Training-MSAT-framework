import torch
import torch.nn as nn


class DetectionHead(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()

        self.cls = nn.Conv2d(channels, num_classes, 1)
        self.box = nn.Conv2d(channels, 4, 1)
        self.obj = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        return {
            "cls": self.cls(x),
            "box": self.box(x),
            "obj": self.obj(x)
        }


class Detector(nn.Module):
    """
    MSAT-based object detector
    """

    def __init__(self, backbone, num_classes=15):
        super().__init__()

        self.backbone = backbone

        self.heads = nn.ModuleList([
            DetectionHead(64, num_classes),
            DetectionHead(64, num_classes),
            DetectionHead(64, num_classes),
            DetectionHead(64, num_classes)
        ])

    def forward(self, x):
        features = self.backbone(x)

        outputs = []

        for feat, head in zip(features, self.heads):
            outputs.append(head(feat))

        return outputs
