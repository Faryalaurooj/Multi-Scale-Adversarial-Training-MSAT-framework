import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, base=64):
        super().__init__()

        def block(in_c, out_c, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, stride, 1),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.net = nn.Sequential(
            block(in_channels, base),
            block(base, base * 2),
            block(base * 2, base * 4),
            block(base * 4, base * 8),

            nn.Conv2d(base * 8, 1, 3, 1, 1)
        )

    def forward(self, x):
        return self.net(x)
