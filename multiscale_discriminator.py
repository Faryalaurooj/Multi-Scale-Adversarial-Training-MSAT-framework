# models/discriminator/discriminator.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiscaleDiscriminator(nn.Module):
    def __init__(self):
        super(MultiscaleDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.fc = nn.Linear(512 * 16 * 16, 1)  # Assumes 256x256 input size

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))

    def loss_fn(self, predictions, labels):
        return F.binary_cross_entropy(predictions, labels.float())

