"""
Example demonstrating the FuseDequantQuantPass optimization.

This example shows how consecutive dequantize→quantize pairs can be
eliminated when they operate on the same dtype AND same scale/offset.

The CustomMLP model has:
- encoder_fc1, encoder_fc2: quantized (int8) with SAME output→input scale
- precision_layer: float32 (no quantization)
- output_fc1, output_fc2: float32 (no quantization)

Between encoder_fc1 and encoder_fc2, there's a redundant:
    dequantize(int8→float, scale=S) → quantize(float→int8, scale=S)

Because the scales match, this is a TRUE NO-OP and can be safely eliminated.
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pytorch_to_c.compiler import compile_model
from src.pytorch_to_c.codegen.c_printer import CPrinter
from src.pytorch_to_c.quantization import (
    StaticQuantRule,
    QuantizationTransform,
    DynamicQuantRuleMinMaxPerTensor,
)
from src.passes import FuseDequantQuantPass

from models.mixed_model_no_op_example import CustomMLP


def count_nodes_by_type(ir_graph):
    """Count nodes by their op_type."""
    counts = {}
    for node in ir_graph.nodes:
        op = node.op_type
        counts[op] = counts.get(op, 0) + 1
    return counts


def main():
    print("=" * 70)
    print("FuseDequantQuantPass Optimization Demo (Static Quantization)")
    print("=" * 70)
    
    # Create model and compile
    model = CustomMLP()
    # Load the weights into model
    model.eval()
    example_input = torch.randn(1, 784)
    
    print("\n📥 Model: CustomMLP")
    print("   - encoder_fc1, encoder_fc2: will be quantized (int8)")
    print("   - precision_layer, output_*: stay float32")
    
    # Define a SHARED scale for the connection between encoder_fc1 and encoder_fc2
    # This is key: output_scale of fc1 == input_scale of fc2 → fusable!
    shared_scale = 0.01  # Common scale for the int8 activations
    weight_scale = 0.005  # Scale for weights
    
    print(f"\n🔧 Using SHARED scale between layers: {shared_scale}")
    print(f"   encoder_fc1 output_scale = {shared_scale}")
    print(f"   encoder_fc2 input_scale  = {shared_scale}")
    print("   → These can be fused because scales match!")
    
    # Compile to IR (for unoptimized version)
    ir_graph_unopt = compile_model(model, example_input, return_ir=True, verbose=False)
    
    # Apply quantization with static rules
    # Key: encoder_fc1's output_scale matches encoder_fc2's input_scale
    rules = [
        StaticQuantRule(
            pattern=r'.*encoder_fc1.*',
            dtype='int8',
            input_scale=0.02,       # Input activation scale
            input_offset=0,
            weight_scale=weight_scale,
            weight_offset=0,
            output_scale=shared_scale,  # <-- This matches encoder_fc2's input
            output_offset=0
        ),
        StaticQuantRule(
            pattern=r'.*encoder_fc2.*',
            dtype='int16',
            input_scale=shared_scale,   # <-- This matches encoder_fc1's output
            input_offset=0,
            weight_scale=weight_scale,
            weight_offset=0,
            output_scale=0.015,     # Output to precision_layer
            output_offset=0
        ),
        DynamicQuantRuleMinMaxPerTensor(
            pattern=r'.*precision_layer.*',
            dtype='int8'
        ),
    ]
    
    transform = QuantizationTransform(rules)
    quant_ir_unopt = transform.apply(ir_graph_unopt)
    
    print("\n" + "=" * 70)
    print("BEFORE Optimization")
    print("=" * 70)
    
    print("\n📊 Node counts:")
    before_counts = count_nodes_by_type(quant_ir_unopt)
    for op, count in sorted(before_counts.items()):
        print(f"   {op}: {count}")
    print(f"   TOTAL: {len(quant_ir_unopt.nodes)} nodes")
    
    # Generate unoptimized code
    os.makedirs("tmp/generated_unoptimized", exist_ok=True)
    printer_before = CPrinter(quant_ir_unopt)
    printer_before.generate_all("tmp/generated_unoptimized_123")
    print("\n📁 Generated unoptimized code: tmp/generated_unoptimized/")
    
    
    # Now create optimized version
    ir_graph_opt = compile_model(model, example_input, return_ir=True, verbose=False)
    quant_ir_opt = QuantizationTransform(rules).apply(ir_graph_opt)
    
    # Apply the optimization pass
    print("\n" + "=" * 70)
    print("Applying FuseDequantQuantPass...")
    print("=" * 70)
    
    fuse_pass = FuseDequantQuantPass(verbose=True)
    optimized_ir = fuse_pass.apply(quant_ir_opt)
    
    stats = fuse_pass.get_stats()
    print(f"\n📈 Pass Statistics:")
    print(f"   Pairs found: {stats['pairs_found']}")
    print(f"   Pairs fused: {stats['pairs_fused']}")
    print(f"   Nodes removed: {stats['nodes_removed']}")
    
    print("\n" + "=" * 70)
    print("AFTER Optimization")
    print("=" * 70)
    
    print("\n📊 Node counts:")
    after_counts = count_nodes_by_type(optimized_ir)
    for op, count in sorted(after_counts.items()):
        print(f"   {op}: {count}")
    print(f"   TOTAL: {len(optimized_ir.nodes)} nodes")
    
    # Generate optimized code
    os.makedirs("tmp/generated_optimized", exist_ok=True)
    printer_after = CPrinter(optimized_ir)
    printer_after.generate_all("tmp/generated_optimized_123")
    print("\n📁 Generated optimized code: tmp/generated_optimized/")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    nodes_saved = len(quant_ir_unopt.nodes) - len(optimized_ir.nodes)
    print(f"   Nodes before: {len(quant_ir_unopt.nodes)}")
    print(f"   Nodes after:  {len(optimized_ir.nodes)}")
    print(f"   Nodes saved:  {nodes_saved} ({100*nodes_saved/len(quant_ir_unopt.nodes):.1f}%)")
    
    # Show IR graphs side by side
    print("\n" + "=" * 70)
    print("IR Graph Comparison (encoder layers)")
    print("=" * 70)
    
    print("\n--- BEFORE ---")
    for node in quant_ir_unopt.nodes:
        if 'encoder' in node.name or node.name == 'x':
            inputs = [n.name for n in node.inputs] if node.inputs else []
            scale_info = f", scale={node.scale}" if hasattr(node, 'scale') else ""
            print(f"  {node.name} [{node.op_type}] dtype={node.dtype}{scale_info}")
            print(f"      inputs: {inputs}")
    
    print("\n--- AFTER ---")
    for node in optimized_ir.nodes:
        if 'encoder' in node.name or node.name == 'x':
            inputs = [n.name for n in node.inputs] if node.inputs else []
            scale_info = f", scale={node.scale}" if hasattr(node, 'scale') else ""
            print(f"  {node.name} [{node.op_type}] dtype={node.dtype}{scale_info}")
            print(f"      inputs: {inputs}")
    
    print("\n" + "=" * 70)
    print("Key Insight")
    print("=" * 70)
    print("""
    Because encoder_fc1's OUTPUT scale (0.01) equals encoder_fc2's INPUT scale (0.01),
    the dequantize→quantize pair is a mathematical no-op:
    
        int8 → float32 → int8  (with same scale)
        
    Is equivalent to just passing the int8 values directly!
    
    This optimization saves:
    - 2 function calls (dequantize + quantize)
    - 1 intermediate float32 buffer
    - CPU cycles
    """)


if __name__ == "__main__":
    main()
