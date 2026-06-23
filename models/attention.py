import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape

        avg = self.mlp(self.avg(x).view(b, c))
        max = self.mlp(self.max(x).view(b, c))

        out = self.sigmoid(avg + max).view(b, c, 1, 1)
        return x * out


class SpatialAttention(nn.Module):
    def __init__(self, kernel=7):
        super().__init__()
        padding = 3 if kernel == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)

        x = torch.cat([avg, max_], dim=1)
        return x * self.sigmoid(self.conv(x))


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))
