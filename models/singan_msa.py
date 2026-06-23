import torch
import torch.nn as nn
import torch.nn.functional as F


class SinGANBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

        self.norm = nn.InstanceNorm2d(channels, affine=True)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x
        x = self.act(self.norm(self.conv1(x)))
        x = self.act(self.norm(self.conv2(x)))
        return x + residual


class SinGAN_MSA(nn.Module):
    """
    Multi-scale SinGAN generator for remote sensing augmentation
    """

    def __init__(self, channels=64, num_layers=5):
        super().__init__()

        self.layers = nn.ModuleList([
            SinGANBlock(channels) for _ in range(num_layers)
        ])

        self.to_rgb = nn.Conv2d(channels, 3, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return torch.tanh(self.to_rgb(x))
