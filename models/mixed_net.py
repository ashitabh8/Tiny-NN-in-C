"""MixedNet -- Conv + ReLU + Linear + Softmax."""

import torch
import torch.nn as nn


class MixedNet(nn.Module):
    """Conv -> ReLU -> flatten -> Linear -> Softmax."""

    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32 * 16 * 16, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x
