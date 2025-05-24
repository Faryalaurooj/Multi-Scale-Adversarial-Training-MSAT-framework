# models/yolov10/yolov10.py
import torch
import torch.nn as nn
import torchvision.models as models

class YOLOv10(nn.Module):
    def __init__(self, num_classes=20):
        super(YOLOv10, self).__init__()
        
        # Using a pre-trained backbone (e.g., MobileNetV2)
        self.backbone = models.mobilenet_v2(pretrained=True).features

        # Detection Head
        self.head = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_classes * 5, kernel_size=1)  # (x, y, w, h, conf) * num_classes
        )

        self.num_classes = num_classes

    def forward(self, x):
        features = self.backbone(x)
        out = self.head(features)
        return out

    def loss_fn(self, preds, targets):
        # Dummy loss - to be replaced with real YOLOv10 loss computation
        # Assume preds and targets are already aligned for demonstration
        return nn.functional.mse_loss(preds, targets)

