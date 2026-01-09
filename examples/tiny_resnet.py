"""
Example: Compiling a Quantized ResNet-style block to C code

This example demonstrates compiling a model with:
- Skip connections (residual connections)
- Conv2D + BatchNorm2D + ReLU pattern
- Element-wise addition
- Dynamic quantization (int8) for Conv2D and Linear layers

Tests proper handling of multi-input nodes, topological ordering, and quantization.
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
    DynamicQuantRuleMinMaxPerTensor,
    QuantizationTransform,
)


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


class TinyResNet(nn.Module):
    """
    A tiny ResNet-style model for embedded deployment.
    
    Architecture:
    - Initial Conv to set channel dimension
    - Two ResNet blocks with skip connections
    - Global Average Pool
    - FC classifier
    """
    
    def __init__(self, in_channels=3, num_classes=10, channels=32):
        super().__init__()
        
        # Initial convolution
        self.conv_init = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.bn_init = nn.BatchNorm2d(channels)
        self.relu_init = nn.ReLU()
        
        # ResNet blocks
        self.block1 = ResNetBlock(channels)
        self.block2 = ResNetBlock(channels)
        
        # Global average pooling will be done manually
        # Classifier
        self.fc = nn.Linear(channels, num_classes)
    
    def forward(self, x):
        # Initial conv
        x = self.conv_init(x)
        x = self.bn_init(x)
        x = self.relu_init(x)
        
        # ResNet blocks
        x = self.block1(x)
        x = self.block2(x)
        
        # Global average pool: [B, C, H, W] -> [B, C]
        x = x.mean(dim=[2, 3])
        
        # Classifier
        x = self.fc(x)
        return x


def print_ir_graph(ir_graph):
    """Print detailed IR graph information."""
    print("\n" + "=" * 60)
    print("IR Graph Details")
    print("=" * 60)
    
    print("\n📊 Graph Summary:")
    print(f"  Total Nodes: {len(ir_graph.nodes)}")
    print(f"  Parameters:  {len(ir_graph.parameters)}")
    print(f"  Inputs:      {len(ir_graph.inputs)}")
    print(f"  Outputs:     {len(ir_graph.outputs)}")
    
    print("\n📋 Node Types:")
    node_types = {}
    for node in ir_graph.nodes:
        node_types[node.op_type] = node_types.get(node.op_type, 0) + 1
    for op_type, count in sorted(node_types.items()):
        print(f"  {op_type}: {count}")
    
    print("\n🔗 Node List:")
    print("-" * 60)
    for i, node in enumerate(ir_graph.nodes):
        inputs_str = ", ".join(inp.name for inp in node.inputs) if node.inputs else "none"
        shape_str = str(node.output_shape) if node.output_shape else "?"
        print(f"  [{i:2d}] {node.name}")
        print(f"       op: {node.op_type}, dtype: {node.dtype}")
        print(f"       shape: {shape_str}")
        print(f"       inputs: [{inputs_str}]")
    
    print("-" * 60)
    
    print("\n📦 Parameters:")
    for name, param in list(ir_graph.parameters.items())[:5]:
        print(f"  {name}: shape={param.shape}, dtype={param.dtype}")
    if len(ir_graph.parameters) > 5:
        print(f"  ... and {len(ir_graph.parameters) - 5} more")


def main():
    """Main example function."""
    print("=" * 60)
    print("PyTorch to C Compiler - Quantized ResNet Block Example")
    print("=" * 60)
    print()
    
    # Create output directory
    output_dir = "tmp/generated_quant_resnet"
    os.makedirs(output_dir, exist_ok=True)
    
    # =========================================================================
    # Step 1: Create model
    # =========================================================================
    print("[1/6] Creating TinyResNet model...")
    model = TinyResNet(in_channels=3, num_classes=10, channels=32)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Create example input: [batch=1, channels=3, height=16, width=16]
    print("\n📥 Creating example input [1, 3, 16, 16]...")
    example_input = torch.randn(1, 3, 16, 16)
    
    print("\n[3/6] Running PyTorch inference...")
    with torch.no_grad():
        pytorch_output = model(example_input)
    print(f"   Output shape: {pytorch_output.shape}")
    print(f"   Output (first 5): {pytorch_output[0, :5].numpy().round(4)}")
    
    # =========================================================================
    # Step 3: Compile to float IR
    # =========================================================================
    print("\n[4/6] Compiling to float IR...")
    
    ir_graph = compile_model(
        model=model,
        example_input=example_input,
        return_ir=True  # Don't generate C yet, just get IR
    )
    
    
    # =========================================================================
    # Step 4: Define quantization rules for Conv2D and Linear layers
    # =========================================================================
    print("\n[5/6] Defining quantization rules...")
    
    rules = [
        # Rule 1: All Conv2D layers -> int8 dynamic quantization
        # Matches: conv_init, block1.conv1, block2.conv1
        DynamicQuantRuleMinMaxPerTensor(
            pattern=r'.*conv.*',  # Match all conv layers
            dtype='int8'
        ),
        
        # Rule 2: FC (classifier) layer -> int8 dynamic quantization
        DynamicQuantRuleMinMaxPerTensor(
            pattern=r'.*fc.*',  # Match fc layer
            dtype='int8'
        ),
    ]
    
    print("   Quantization rules:")
    for i, rule in enumerate(rules, 1):
        print(f"    {i}. pattern='{rule.pattern}', dtype={rule.dtype}")
    print("   -> Scale/offset computed automatically from weight statistics")
    
    # =========================================================================
    # Step 5: Apply quantization transform
    # =========================================================================
    print("\n[6/6] Applying quantization transform...")
    
    transform = QuantizationTransform(rules)
    quant_ir = transform.apply(ir_graph)
    
    # Count quantized nodes by type
    int8_count = sum(1 for n in quant_ir.nodes if n.dtype == 'int8')
    float_count = sum(1 for n in quant_ir.nodes if n.dtype == 'float32')
    
    print(f"   Nodes by dtype:")
    print(f"     int8:    {int8_count}")
    print(f"     float32: {float_count}")
    
    
    # Print detailed IR
    ir_str = quant_ir.print_graph()
    print(ir_str)
    print("\n" + "=" * 60)
    print("Generating quantized C code...")
    print("=" * 60)
    
    printer = CPrinter(quant_ir)
    printer.generate_all(output_dir)
    
    print(f"\n📁 Generated files in: {output_dir}/")
    print("   - model.h         : Function declarations")
    print("   - model.c         : Model implementation (quantized ops)")
    print("   - weights.h       : Trained weights (int8 quantized)")
    print("   - nn_ops_*.h      : Operations library")
    


if __name__ == "__main__":
    main()

