"""TinyResNet -- ResNet-style models for the Tiny-NN-in-C compiler.

Two variants:
  TinyResNet   -- 2D ResNet for images (MNIST, CIFAR-style).
  TinyResNet1D -- 1D ResNet for sensor / audio signals.
"""

import torch
import torch.nn as nn


# -----------------------------------------------------------------------
# 2D variant -- for images (e.g. MNIST 28x28)
# -----------------------------------------------------------------------

class ResBlock2D(nn.Module):
    """2D residual block: two 3x3 convs with skip connection.

    Input and output have the same spatial size and channel count.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = self.relu2(out)
        return out


class TinyResNet(nn.Module):
    """Tiny 2D ResNet for small images (e.g. MNIST).

    Architecture:
      stem  Conv2d(in, 16, 3)  + BN + ReLU             [28x28]
      block1  ResBlock2D(16)                             [28x28]
      down1 Conv2d(16, 32, 3, s=2) + BN + ReLU          [14x14]
      block2  ResBlock2D(32)                             [14x14]
      down2 Conv2d(32, 64, 3, s=2) + BN + ReLU          [7x7]
      block3  ResBlock2D(64)                             [7x7]
      global avg pool (mean dim=[2,3])                   [64]
      Linear(64, num_classes)                            [num_classes]

    Uses only compiler-supported ops: Conv2d, BatchNorm2d, ReLU,
    element-wise add (residual), mean, Linear.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10,
                 channels: tuple = (16, 32, 64)):
        super().__init__()
        c0, c1, c2 = channels

        self.conv_stem = nn.Conv2d(in_channels, c0, 3, padding=1)
        self.bn_stem = nn.BatchNorm2d(c0)
        self.relu_stem = nn.ReLU()
        self.block1 = ResBlock2D(c0)

        self.conv_down1 = nn.Conv2d(c0, c1, 3, stride=2, padding=1)
        self.bn_down1 = nn.BatchNorm2d(c1)
        self.relu_down1 = nn.ReLU()
        self.block2 = ResBlock2D(c1)

        self.conv_down2 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.bn_down2 = nn.BatchNorm2d(c2)
        self.relu_down2 = nn.ReLU()
        self.block3 = ResBlock2D(c2)

        self.fc = nn.Linear(c2, num_classes)

    def forward(self, x):
        x = self.relu_stem(self.bn_stem(self.conv_stem(x)))
        x = self.block1(x)
        x = self.relu_down1(self.bn_down1(self.conv_down1(x)))
        x = self.block2(x)
        x = self.relu_down2(self.bn_down2(self.conv_down2(x)))
        x = self.block3(x)
        x = x.mean(dim=[2, 3])
        x = self.fc(x)
        return x


# -----------------------------------------------------------------------
# 1D variant -- for sensor / audio signals
# -----------------------------------------------------------------------

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
