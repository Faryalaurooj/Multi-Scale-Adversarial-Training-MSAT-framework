# models/discriminator/discriminator.py
import torch
import torch.nn as nn

class MultiscaleDiscriminator(nn.Module):
    def __init__(self):
        super(MultiscaleDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

    def loss_fn(self, preds, labels):
        # Binary Cross Entropy Loss for real/fake classification
        return nn.BCELoss()(preds.squeeze(), labels.float())

