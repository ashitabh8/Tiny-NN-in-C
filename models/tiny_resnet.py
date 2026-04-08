"""TinyResNet1D -- ResNet-style model for 1D signals (sensor data, audio)."""

import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
    """1D residual block using Conv2d with kernel (1, k).

    Input shape: (B, C, 1, W) where W is the sequence length.
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = (0, kernel_size // 2)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), padding=padding)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), padding=padding)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu2(out)
        return out


class TinyResNet1D(nn.Module):
    """Tiny ResNet for 1D signals (~100KB int8 quantized).

    Architecture: stem conv -> 4 ResBlocks -> global avg pool -> FC
    """

    def __init__(self, in_channels: int = 10, num_classes: int = 10, hidden_channels: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.conv_init = nn.Conv2d(in_channels, hidden_channels, kernel_size=(1, 7), padding=(0, 3))
        self.bn_init = nn.BatchNorm2d(hidden_channels)
        self.relu_init = nn.ReLU()
        self.block1 = ResBlock1D(hidden_channels)
        self.block2 = ResBlock1D(hidden_channels)
        self.block3 = ResBlock1D(hidden_channels)
        self.block4 = ResBlock1D(hidden_channels)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        x = self.conv_init(x)
        x = self.bn_init(x)
        x = self.relu_init(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.mean(dim=[2, 3])
        x = self.fc(x)
        return x
