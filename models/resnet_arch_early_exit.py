import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Basic residual block WITHOUT shortcut projection (in_channels == out_channels, stride == 1)."""
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
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


class BasicBlockDown(nn.Module):
    """Basic residual block WITH shortcut projection (channel change and/or stride > 1)."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        
        # Shortcut projection
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


class SimpleResNet(nn.Module):
    """
    Simple ResNet for inference - no conditionals.
    
    Config:
        filter_sizes: [16, 16, 32, 64]
        layers: [2, 2, 2, 2]
        stem_kernel: 5, stem_stride: 2
        fc_dim: 64
        use_maxpool: False
    
    Input: (B, 6, 7, 256)
    Output: (B, num_classes)
    """
    
    def __init__(self, in_channels: int = 6, num_classes: int = 10):
        super().__init__()
        
        # Stem
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        
        # Stage 0: 16 -> 16, stride=1, 2 blocks
        self.stage0_block0 = BasicBlock(16)
        self.stage0_block1 = BasicBlock(16)
        
        # Stage 1: 16 -> 16, stride=2 at first block, 2 blocks
        self.stage1_block0 = BasicBlockDown(16, 16, stride=2)
        self.stage1_block1 = BasicBlock(16)
        
        # Stage 2: 16 -> 32, stride=2 at first block, 2 blocks
        self.stage2_block0 = BasicBlockDown(16, 32, stride=2)
        self.stage2_block1 = BasicBlock(32)
        
        # Stage 3: 32 -> 64, stride=2 at first block, 2 blocks
        self.stage3_block0 = BasicBlockDown(32, 64, stride=2)
        self.stage3_block1 = BasicBlock(64)
        
        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 64)
        self.fc1_relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
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
        # x = x.flatten(1)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc1_relu(x)
        x = self.fc2(x)
        
        return x