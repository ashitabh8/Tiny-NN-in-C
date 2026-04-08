"""MNISTConvNet -- CNN for MNIST using only compiler-supported ops."""

import torch
import torch.nn as nn


class MNISTConvNet(nn.Module):
    """Conv2d + BatchNorm2d + ReLU + mean (global avg pool) + Linear.

    Architecture:
      Conv2d(1, 64, 3, pad=1)        -> BN -> ReLU   [28x28]
      Conv2d(64, 128, 3, pad=1, s=2) -> BN -> ReLU   [14x14]
      Conv2d(128, 256, 3, pad=1, s=2)-> BN -> ReLU   [7x7]
      Conv2d(256, 384, 3, pad=1, s=2)-> BN -> ReLU   [4x4]
      mean(dim=[2,3])                                 [384]
      Linear(384, 10)                                 [10]
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(256, 384, 3, padding=1, stride=2)
        self.bn4 = nn.BatchNorm2d(384)
        self.relu4 = nn.ReLU()
        self.fc = nn.Linear(384, 10)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = x.mean(dim=[2, 3])
        x = self.fc(x)
        return x
