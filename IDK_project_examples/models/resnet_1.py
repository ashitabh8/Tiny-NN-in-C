import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Identity-shortcut basic block (in_channels == out_channels, stride == 1).

    No projected shortcut needed, so no if-statement required in generated code.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + x       # identity shortcut — always valid, no if needed
        out = self.relu2(out)
        return out


class BasicBlockDown(nn.Module):
    """Projected-shortcut basic block (in_channels != out_channels or stride > 1).

    Shortcut is always a 1x1 conv projection, so no if-statement required in
    generated code.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.shortcut_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = self.shortcut_conv(x)
        identity = self.shortcut_bn(identity)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu2(out)
        return out


class TinyResNet(nn.Module):
    """ResNet18-style architecture for embedded inference.

    Specs:
        layers:       [2, 2, 2, 2]
        filter_sizes: [16, 16, 32, 64]
        num_classes:  10
        fc_dim:       64
        stem_kernel:  5
        stem_stride:  2

    Two distinct block classes are used instead of a single conditional block so
    that the generated C code never requires an if-statement to implement the
    shortcut connection.

    Supported ops: conv, batchnorm, relu, +, view, avgpool, linear, softmax.
    """

    def __init__(self, in_channels: int = 6, num_classes: int = 10):
        super().__init__()

        filter_sizes = [2, 2, 2, 2]
        fc_dim = 32

        # Stem: in_channels -> 16, 5x5 conv, stride=2
        self.stem_conv = nn.Conv2d(in_channels, filter_sizes[0], kernel_size=5, stride=2, padding=2, bias=False)
        self.stem_bn   = nn.BatchNorm2d(filter_sizes[0])
        self.stem_relu = nn.ReLU()

        # Stage 0: 16 -> 16, stride=1, 2 blocks — identity shortcut
        self.stage0_block0 = BasicBlock(filter_sizes[0])
        self.stage0_block1 = BasicBlock(filter_sizes[0])

        # Stage 1: 16 -> 16, stride=1, 2 blocks — identity shortcut
        self.stage1_block0 = BasicBlock(filter_sizes[1])
        self.stage1_block1 = BasicBlock(filter_sizes[1])

        # Stage 2: 16 -> 32, stride=2, 2 blocks — projected shortcut then identity
        self.stage2_block0 = BasicBlockDown(filter_sizes[1], filter_sizes[2], stride=2)
        self.stage2_block1 = BasicBlock(filter_sizes[2])

        # Stage 3: 32 -> 64, stride=2, 2 blocks — projected shortcut then identity
        self.stage3_block0 = BasicBlockDown(filter_sizes[2], filter_sizes[3], stride=2)
        self.stage3_block1 = BasicBlock(filter_sizes[3])

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1      = nn.Linear(filter_sizes[3], fc_dim)
        self.fc1_relu = nn.ReLU()
        self.fc2      = nn.Linear(fc_dim, num_classes)
        self.final_softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Stem
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_relu(x)

        # Stage 0
        x = self.stage0_block0(x)
        x = self.stage0_block1(x)

        # Stage 1
        x = self.stage1_block0(x)
        x = self.stage1_block1(x)

        # Stage 2
        x = self.stage2_block0(x)
        x = self.stage2_block1(x)

        # Stage 3
        x = self.stage3_block0(x)
        x = self.stage3_block1(x)

        # Head
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc1_relu(x)
        x = self.fc2(x)
        x = self.final_softmax(x)

        return x
