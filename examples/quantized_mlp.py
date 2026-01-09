"""
Example: Quantizing and Compiling an MLP to C code

This example demonstrates how to:
1. Compile a PyTorch model to float IR
2. Apply quantization rules (int8/int16)
3. Generate quantized C code

Shows fine-grained control with different quantization for different layers.
"""

import torch
import torch.nn as nn
import sys
import os

# Add src to path so we can import the compiler
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pytorch_to_c.compiler import compile_model
from src.pytorch_to_c.codegen.c_printer import CPrinter
from src.pytorch_to_c.quantization import (
    StaticQuantRule, 
    DynamicQuantRuleMinMaxPerTensor,
    QuantizationTransform,
    StaticQuantLinearNode,
    DynamicQuantLinearNode
)


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
        x = torch.relu(self.encoder_fc1(x))
        x = torch.relu(self.encoder_fc2(x))
        
        # Precision layer (stays float32)
        x = torch.relu(self.precision_layer(x))
        
        # Output
        x = torch.relu(self.output_fc1(x))
        x = self.output_fc2(x)
        return x


def print_ir_graph(ir_graph, title="IR Graph"):
    """Pretty print the IR graph with dtype info."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    
    for node in ir_graph.nodes:
        # Get dtype info
        dtype = node.dtype
        if isinstance(node, (StaticQuantLinearNode, DynamicQuantLinearNode)):
            dtype_info = f"{dtype} (scale={node.scale:.4f}, offset={node.offset})"
        elif node.op_type in ['quantize', 'dequantize', 'dynamic_quantize']:
            scale = node.metadata.get('scale', 'runtime') if node.op_type == 'dynamic_quantize' else node.metadata.get('scale', 'N/A')
            dtype_info = f"{dtype} (scale={scale})"
        else:
            dtype_info = dtype
        
        # Get shape info
        shape_str = str(node.output_shape) if node.output_shape else "unknown"
        
        # Print node info
        inputs_str = ", ".join([inp.name for inp in node.inputs]) if node.inputs else "input"
        print(f"  {node.name}")
        print(f"    op: {node.op_type}, dtype: {dtype_info}")
        print(f"    shape: {shape_str}")
        print(f"    inputs: [{inputs_str}]")
        print()


def main():
    """Main example function."""
    print("=" * 60)
    print(" Quantization Example - Mixed Precision MLP")
    print("=" * 60)
    
    # =========================================================================
    # Step 1: Create model
    # =========================================================================
    print("\n[1/5] Creating model with custom layer names...")
    model = CustomMLP()
    model.eval()
    
    print("  Layer names in model:")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"    - {name}: Linear({module.in_features} -> {module.out_features})")
    
    # =========================================================================
    # Step 2: Compile to float IR
    # =========================================================================
    print("\n[2/5] Compiling to float IR...")
    example_input = torch.randn(1, 784)
    
    ir_graph = compile_model(
        model=model,
        example_input=example_input,
        return_ir=True  # Don't generate C yet, just get IR
    )
    
    print(f"  Created {len(ir_graph.nodes)} IR nodes")
    print(f"  Extracted {len(ir_graph.parameters)} parameters")
    
    # Print float IR
    print_ir_graph(ir_graph, "Float32 IR Graph (Before Quantization)")
    
    # =========================================================================
    # Step 3: Define quantization rules
    # =========================================================================
    print("\n[3/5] Defining quantization rules...")
    
    rules = [
        # Rule 1: Encoder layers -> int8 (aggressive quantization)
        StaticQuantRule(
            pattern=r'.*encoder.*',
            dtype='int8',
            input_scale=0.05, input_offset=0,
            weight_scale=0.05, weight_offset=0,
            output_scale=0.05, output_offset=0
        ),
        
        # Rule 2: Output layers -> int16 (higher precision)
        StaticQuantRule(
            pattern=r'.*output.*',
            dtype='int16',
            input_scale=0.001, input_offset=0,
            weight_scale=0.001, weight_offset=0,
            output_scale=0.001, output_offset=0
        ),
        
        # Note: precision_layer doesn't match any rule -> stays float32!
    ]
    
    print("  Quantization rules:")
    for i, rule in enumerate(rules, 1):
        print(f"    {i}. pattern='{rule.pattern}', dtype={rule.dtype}, scale={rule.scale}")
    print("    → 'precision_layer' has no matching rule → stays float32")
    
    # =========================================================================
    # Step 4: Apply quantization
    # =========================================================================
    print("\n[4/5] Applying quantization transform...")
    
    transform = QuantizationTransform(rules)
    quant_ir = transform.apply(ir_graph)
    
    # Count quantized nodes
    int8_count = sum(1 for n in quant_ir.nodes if n.dtype == 'int8')
    int16_count = sum(1 for n in quant_ir.nodes if n.dtype == 'int16')
    float_count = sum(1 for n in quant_ir.nodes if n.dtype == 'float32')
    
    print(f"  Nodes by dtype:")
    print(f"    int8:    {int8_count}")
    print(f"    int16:   {int16_count}")
    print(f"    float32: {float_count}")
    
    # Print quantized IR
    print_ir_graph(quant_ir, "Quantized IR Graph (After Quantization)")
    
    # =========================================================================
    # Step 5: Generate C code
    # =========================================================================
    print("\n[5/5] Generating quantized C code...")
    
    output_dir = "generated_quant"
    os.makedirs(output_dir, exist_ok=True)
    
    printer = CPrinter(quant_ir)
    printer.generate_all(output_dir)
    
    print(f"  Generated files in '{output_dir}/':")
    for f in os.listdir(output_dir):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"    - {f} ({size} bytes)")
    
    # =========================================================================
    # Show generated code snippets
    # =========================================================================
    print("\n" + "=" * 60)
    print(" Generated Code Snippets")
    print("=" * 60)
    
    # Show buffer declarations from model.c
    with open(os.path.join(output_dir, 'model.c')) as f:
        model_c = f.read()
    
    print("\n--- Buffer Declarations (model.c) ---")
    in_buffers = False
    for line in model_c.split('\n'):
        if 'Intermediate buffers' in line:
            in_buffers = True
        if in_buffers:
            print(line)
            if line.strip() == '' and in_buffers:
                break
    
    print("\n--- Quantized Operations (model.c) ---")
    for line in model_c.split('\n'):
        if 'dense_int8' in line or 'dense_int16' in line:
            print(f"  {line.strip()}")
        if 'quantize_float_to_int' in line or 'dequantize_int' in line:
            print(f"  {line.strip()}")
    
    # Show weight types from weights.h
    print("\n--- Weight Types (weights.h) ---")
    with open(os.path.join(output_dir, 'weights.h')) as f:
        for line in f:
            if line.startswith('// Shape:') or line.startswith('static const'):
                print(f"  {line.rstrip()}")
                if 'static const' in line:
                    print()
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print(" Summary")
    print("=" * 60)
    print(f"""
  ✓ Model compiled with mixed precision:
    - encoder layers:    int8  (aggressive, ~4x memory savings)
    - precision_layer:   float32 (kept precise)
    - output layers:     int16 (2x memory savings, higher accuracy)
    
  ✓ Generated C files in '{output_dir}/'
  
  ✓ C code includes:
    - int8_t buffers for encoder layers
    - int16_t buffers for output layers
    - float buffers for precision layer
    - Quantize/Dequantize operations at type boundaries
    
  Next steps:
    1. Review generated code
    2. Compile with: gcc -c model.c -I.
    3. Link with your embedded project
    4. Call model_forward(input, output)
""")


if __name__ == "__main__":
    main()

