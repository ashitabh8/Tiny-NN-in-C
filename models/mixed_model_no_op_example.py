import torch
import torch.nn as nn

class CustomMLP(nn.Module):
    """
    MLP with custom layer names for fine-grained quantization control.
    
    Layer naming strategy:
    - encoder_* : Can be aggressively quantized (int8)
    - precision_* : Keep in float32 (no quantization)
    - output_* : Use higher precision (int16)
    """
    
    def __init__(self):
        super().__init__()
        # Encoder layers - can use aggressive int8 quantization
        self.encoder_fc1 = nn.Linear(784, 256)
        self.encoder_fc2 = nn.Linear(256, 128)
        
        # Precision-critical layer - keep in float32
        self.precision_layer = nn.Linear(128, 64)
        
        # Output layers - use int16 for better accuracy
        self.output_fc1 = nn.Linear(64, 32)
        self.output_fc2 = nn.Linear(32, 10)
    
    def forward(self, x):
        # Encoder
        x = self.encoder_fc1(x)
        x = self.encoder_fc2(x)
        
        # Precision layer (stays float32)
        x = torch.relu(self.precision_layer(x))
        

        # Output
        x = self.output_fc1(x)
        x = self.output_fc2(x)
        return x