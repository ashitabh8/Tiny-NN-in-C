"""
Tiny ResNet for Embedded Devices
================================

A ~100KB int8-quantized ResNet-style model designed for microcontrollers.

Input:  (1, 10, 200) - 10 channels, 200 time steps (1D signal)
Output: (1, 10)      - 10-class classification

Model size:
- Float32: ~400KB
- Int8:    ~100KB (with static quantization)

Usage:
    python examples/tiny_resnet.py
    
Generated files will be in: tmp/tiny_resnet_for_embedded_device/
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pytorch_to_c.compiler import compile_model
from src.pytorch_to_c.codegen.c_printer import CPrinter
from src.pytorch_to_c.quantization import (
    StaticQuantRule,
    QuantizationTransform,
)


class ResBlock1D(nn.Module):
    """
    1D Residual block using Conv2d with kernel (1, k).
    
    Input shape: (B, C, 1, W) where W is the sequence length.
    This is equivalent to Conv1d but uses Conv2d for compatibility.
    """
    
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = (0, kernel_size // 2)  # Only pad width dimension
        
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
        
        out = self.conv2(x)
        out = self.bn2(out)
        
        out = out + identity  # Skip connection
        out = self.relu2(out)
        return out


class TinyResNet1D(nn.Module):
    """
    Tiny ResNet for 1D signals (e.g., sensor data, audio features).
    
    Architecture:
    - Input reshape: (B, C, W) -> (B, C, 1, W) for Conv2d compatibility
    - Initial conv: expand channels
    - 4 ResNet blocks with skip connections
    - Global average pool
    - FC classifier
    
    Target size: ~100KB with int8 quantization
    
    Parameters breakdown (for channels=64, 4 blocks):
    - conv_init:    10 * 64 * 1 * 7 = 4,480 + 64 bias = 4,544
    - res blocks:   4 blocks * 2 convs * (64 * 64 * 1 * 3) = 98,304 + biases
    - fc:           64 * 10 = 640 + 10 bias = 650
    - Total:        ~103,500 params ≈ 100KB int8
    """
    
    def __init__(self, in_channels: int = 10, num_classes: int = 10, hidden_channels: int = 64):
        super().__init__()
        
        self.in_channels = in_channels
        
        # Initial convolution to expand channel dimension
        # Kernel (1, 7) for wider receptive field at input
        self.conv_init = nn.Conv2d(in_channels, hidden_channels, kernel_size=(1, 7), padding=(0, 3))
        self.bn_init = nn.BatchNorm2d(hidden_channels)
        self.relu_init = nn.ReLU()
        
        # Residual blocks
        self.block1 = ResBlock1D(hidden_channels)
        self.block2 = ResBlock1D(hidden_channels)
        self.block3 = ResBlock1D(hidden_channels)
        self.block4 = ResBlock1D(hidden_channels)
        
        # Classifier
        self.fc = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x):
        # Input: (B, C, W) -> reshape to (B, C, 1, W) for Conv2d
        # Note: The compiler traces with the actual input shape, so we
        # expect input already in (B, C, 1, W) format
        
        # Initial conv
        x = self.conv_init(x)
        x = self.bn_init(x)
        x = self.relu_init(x)
        
        # ResNet blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        # Global average pool: (B, C, 1, W) -> (B, C)
        x = x.mean(dim=[2, 3])
        
        # Classifier
        x = self.fc(x)
        return x


def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_size_kb(model, dtype='float32'):
    """Estimate model size in KB."""
    total_params = count_parameters(model)
    bytes_per_param = {'float32': 4, 'int8': 1, 'int16': 2}[dtype]
    return (total_params * bytes_per_param) / 1024


def main():
    print("=" * 70)
    print("Tiny ResNet for Embedded Devices")
    print("=" * 70)
    
    # Configuration
    OUTPUT_DIR = "tmp/tiny_resnet_for_embedded_device"
    OUTPUT_DIR_FLOAT = "tmp/tiny_resnet_for_embedded_device_float"
    INPUT_SHAPE = (1, 10, 1, 200)  # (batch, channels, height=1, width=200)
    NUM_CLASSES = 10
    HIDDEN_CHANNELS = 64
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # =========================================================================
    # Step 1: Create model
    # =========================================================================
    print("\n[1/5] Creating TinyResNet1D model...")
    model = TinyResNet1D(
        in_channels=INPUT_SHAPE[1],  # 10
        num_classes=NUM_CLASSES,
        hidden_channels=HIDDEN_CHANNELS
    )
    model.eval()
    
    total_params = count_parameters(model)
    size_float = estimate_size_kb(model, 'float32')
    size_int8 = estimate_size_kb(model, 'int8')
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Size (float32):   {size_float:.1f} KB")
    print(f"   Size (int8):      {size_int8:.1f} KB")
    
    # =========================================================================
    # Step 2: Create example input and run PyTorch inference
    # =========================================================================
    print(f"\n[2/5] Creating example input {INPUT_SHAPE}...")
    example_input = torch.randn(*INPUT_SHAPE)
    
    print("   Running PyTorch inference...")
    with torch.no_grad():
        pytorch_output = model(example_input)
    print(f"   Output shape: {pytorch_output.shape}")
    print(f"   Output: {pytorch_output[0].detach().numpy().round(4)}")
    
    # =========================================================================
    # Step 3: Compile to IR
    # =========================================================================
    print("\n[3/5] Compiling PyTorch model to IR...")
    ir_graph = compile_model(
        model=model,
        example_input=example_input,
        return_ir=True
    )
    print(f"   IR nodes: {len(ir_graph.nodes)}")
    
    # =========================================================================
    # Step 4: Apply static quantization to conv layers
    # =========================================================================
    print("\n[4/5] Applying static quantization (int8) to conv layers...")
    
    # Static quantization rules for all conv layers
    # Using calibrated scales (in practice, these come from calibration data)
    rules = [
        StaticQuantRule(
            pattern=r'.*conv.*',
            dtype='int8',
            input_scale=0.05,
            input_offset=0,
            weight_scale=0.02,
            weight_offset=0,
            output_scale=0.05,
            output_offset=0
        ),
    ]
    
    print("   Quantization rules:")
    for rule in rules:
        print(f"     - pattern='{rule.pattern}', dtype={rule.dtype}")
        print(f"       input_scale={rule.input_scale}, weight_scale={rule.weight_scale}")

    printer_unoptimized = CPrinter(ir_graph)
    printer_unoptimized.generate_all(OUTPUT_DIR_FLOAT)

    print("GEenrated float code to ", OUTPUT_DIR_FLOAT)
    
    transform = QuantizationTransform(rules)
    quant_ir = transform.apply(ir_graph)
    
    # Count nodes by dtype
    dtype_counts = {}
    for node in quant_ir.nodes:
        dtype_counts[node.dtype] = dtype_counts.get(node.dtype, 0) + 1
    
    print(f"   Quantized IR nodes by dtype:")
    for dtype, count in sorted(dtype_counts.items()):
        print(f"     {dtype}: {count}")
    
    # =========================================================================
    # Step 5: Generate C code
    # =========================================================================
    print(f"\n[5/5] Generating C code to {OUTPUT_DIR}/...")
    
    printer = CPrinter(quant_ir)
    printer.generate_all(OUTPUT_DIR)
    
    print(f"\n{'=' * 70}")
    print("SUCCESS! Generated files:")
    print(f"{'=' * 70}")
    print(f"   {OUTPUT_DIR}/model.h      - Function declarations")
    print(f"   {OUTPUT_DIR}/model.c      - Model implementation")
    print(f"   {OUTPUT_DIR}/weights.h    - Quantized weights (int8)")
    print(f"   {OUTPUT_DIR}/nn_ops_*.h   - C operation kernels")
    
    print(f"\n{'=' * 70}")
    print("HOW TO USE THE GENERATED C CODE:")
    print(f"{'=' * 70}")
    print("""
1. Include the header in your embedded project:
   #include "model.h"

2. Prepare your input data (NHWC layout):
   - Input shape: (1, 1, 200, 10) in NHWC = 2000 floats
   - PyTorch uses NCHW (1, 10, 1, 200), C code uses NHWC (1, 1, 200, 10)
   
   float input_data[1 * 1 * 200 * 10];  // 2000 floats
   // Fill input_data with your sensor readings...

3. Allocate output buffer:
   float output[10];  // 10 classes

4. Run inference:
   model_forward(input_data, output);

5. Read output:
   // output[0..9] contains class scores
   int predicted_class = argmax(output, 10);

6. Compile for your target:
   # For testing on host:
   gcc -O2 -o model_test main.c model.c -lm
   
   # For ARM Cortex-M:
   arm-none-eabi-gcc -mcpu=cortex-m4 -O2 -c model.c -o model.o
""")
    
    # Print IR summary
    print(f"\n{'=' * 70}")
    print("IR GRAPH SUMMARY:")
    print(f"{'=' * 70}")
    print(quant_ir.print_graph())


if __name__ == "__main__":
    main()
