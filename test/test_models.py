"""
Test model definitions for the compiler test suite
"""

import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    """
    Simple MLP: Linear -> ReLU -> Linear
    Tests basic flow without complications.
    """
    
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ResNetBlock(nn.Module):
    """
    Simplified ResNet block: Conv -> BatchNorm -> ReLU -> Add (skip connection)
    Tests skip connections and topological traversing.
    """
    
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
        out = out + identity  # Skip connection
        return out


class MixedNet(nn.Module):
    """
    Mixed network: Conv -> ReLU -> Linear -> Softmax
    Tests different operation types and datatype bridging (future quantization).
    """
    
    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32 * 16 * 16, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Input: [B, C, H, W]
        x = self.conv(x)  # [B, 32, 16, 16]
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        x = self.softmax(x)
        return x


def get_test_models():
    """
    Get all test models with example inputs.
    
    Returns:
        List of tuples (model_name, model, example_input)
    """
    return [
        ("TinyMLP", TinyMLP(), torch.randn(1, 784)),
        ("ResNetBlock", ResNetBlock(), torch.randn(1, 64, 32, 32)),
        ("MixedNet", MixedNet(), torch.randn(1, 3, 32, 32)),
    ]

