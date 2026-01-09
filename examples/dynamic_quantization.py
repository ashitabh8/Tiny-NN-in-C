"""
Example: Dynamic Quantization with automatic scale/offset computation

This example demonstrates DynamicQuantRuleMinMaxPerTensor which
automatically computes scale/offset from weight statistics - no
calibration data needed!
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pytorch_to_c.compiler import compile_model
from src.pytorch_to_c.codegen.c_printer import CPrinter
from src.pytorch_to_c.quantization import (
    DynamicQuantRuleMinMaxPerTensor,
    QuantizationTransform,
)


class TinyNet(nn.Module):
    """Simple 2-layer network for testing."""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 4)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    print("=" * 60)
    print(" Dynamic Quantization Example")
    print("=" * 60)
    print()
    
    # Create model
    model = TinyNet()
    model.eval()
    
    # Compile to float IR
    print("[1/4] Compiling to float IR...")
    example_input = torch.randn(1, 16)
    ir_graph = compile_model(model, example_input, return_ir=True)
    
    # Show float weights
    print("\n[2/4] Original float weights:")
    for name, param in ir_graph.parameters.items():
        if 'weight' in name:
            print(f"  {name}:")
            print(f"    shape: {param.shape}")
            print(f"    range: [{param.min():.4f}, {param.max():.4f}]")
    
    # Define dynamic quantization rules
    print("\n[3/4] Applying dynamic quantization...")
    rules = [
        # DynamicQuantRule computes scale from weight statistics
        DynamicQuantRuleMinMaxPerTensor(
            pattern=r'fc.*',  # Match all fc layers
            dtype='int8'
        ),
    ]
    
    transform = QuantizationTransform(rules)
    quant_ir = transform.apply(ir_graph)
    
    # Show computed scales
    print("\n[4/4] Quantized weights (scale computed from weights):")
    for name, param in quant_ir.parameters.items():
        if 'weight' in name:
            print(f"  {name}:")
            print(f"    dtype: {param.dtype}")
            print(f"    shape: {param.shape}")
            if hasattr(param, 'tolist'):
                flat = param.flatten()
                print(f"    range: [{flat.min()}, {flat.max()}]")
    
    # Show quantized nodes
    print("\n" + "=" * 60)
    print(" IR Graph (quantized)")
    print("=" * 60)
    for node in quant_ir.nodes:
        scale_info = ""
        if hasattr(node, 'scale') and node.scale is not None:
            scale_info = f", scale={node.scale:.6f}"
        elif node.op_type == 'dynamic_quantize':
            scale_info = ", scale=<computed at runtime>"
        elif node.op_type in ['quantize', 'dequantize']:
            scale_val = node.metadata.get('scale')
            scale_info = f", scale={scale_val:.6f}" if scale_val else ""
        print(f"  {node.name}: {node.op_type} [{node.dtype}{scale_info}]")
    
    # Generate C code
    print("\n" + "=" * 60)
    print(" Generating C code...")
    print("=" * 60)
    output_dir = "generated_dynamic"
    os.makedirs(output_dir, exist_ok=True)
    printer = CPrinter(quant_ir)
    printer.generate_all(output_dir)
    
    print(f"\n  Generated files in '{output_dir}/':")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith(('.c', '.h')):
            print(f"    - {f}")
    
    print("\n✓ Dynamic quantization complete!")
    print("  No calibration data was needed - scales computed from weights.")
    print()


if __name__ == "__main__":
    main()

