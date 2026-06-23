import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CBAM


class MSATBlock(nn.Module):
    """
    Multi-scale + attention fusion block
    """

    def __init__(self, channels):
        super().__init__()

        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(channels)

        self.attn = CBAM(channels)
        self.act = nn.SiLU()

    def forward(self, x):
        residual = x

        x = self.conv(x)
        x = self.bn(x)

        x = self.attn(x)

        return self.act(x + residual)


class MSAT(nn.Module):
    """
    Backbone network for RS object detection
    """

    def __init__(self, in_channels=3, base=64, layers=4):
        super().__init__()

        self.stem = nn.Conv2d(in_channels, base, 3, padding=1)

        self.blocks = nn.ModuleList([
            MSATBlock(base) for _ in range(layers)
        ])

        self.down = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.stem(x)

        features = []

        for block in self.blocks:
            x = block(x)
            features.append(x)
            x = self.down(x)

        return features  # multi-scale features
