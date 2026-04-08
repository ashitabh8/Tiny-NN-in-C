"""ResNetBlock -- simplified ResNet block for testing skip connections."""

import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU -> Add (skip connection)."""

    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = out + identity
        return out
