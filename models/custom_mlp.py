"""CustomMLP -- MLP with named layers for fine-grained quantization control."""

import torch
import torch.nn as nn


class CustomMLP(nn.Module):
    """MLP with encoder/precision/output naming for selective quantization.

    Layer naming strategy:
    - encoder_*    : can be aggressively quantized (int8)
    - precision_*  : keep in float32
    - output_*     : use higher precision (int16) or float
    """

    def __init__(self):
        super().__init__()
        self.encoder_fc1 = nn.Linear(784, 256)
        self.encoder_fc2 = nn.Linear(256, 128)
        self.precision_layer = nn.Linear(128, 64)
        self.output_fc1 = nn.Linear(64, 32)
        self.output_fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.encoder_fc1(x)
        x = self.encoder_fc2(x)
        x = torch.relu(self.precision_layer(x))
        x = self.output_fc1(x)
        x = self.output_fc2(x)
        return x
