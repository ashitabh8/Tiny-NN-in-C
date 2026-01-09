"""
Tests for compiler optimization passes.

These tests verify that optimization passes produce numerically correct results.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import subprocess
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pytorch_to_c.compiler import compile_model
from src.pytorch_to_c.codegen.c_printer import CPrinter
from src.pytorch_to_c.quantization import (
    StaticQuantRule,
    QuantizationTransform,
)
from src.passes import FuseDequantQuantPass


# ============================================================================
# Test Model (copied from models/mixed_model_no_op_example.py)
# ============================================================================

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


# ============================================================================
# Test Helpers
# ============================================================================

class TestFuseDequantQuantPass:
    """Tests for FuseDequantQuantPass optimization."""
    
    def _compile_and_run_c_model(self, tmpdir, input_np, input_size, output_size):
        """Compile and run the C model, return output."""
        # Create test harness
        test_c = f'''
#include <stdio.h>
#include <stdlib.h>
#include "model.h"

int main() {{
    float input[{input_size}];
    float output[{output_size}];
    
    // Read input from stdin
    for (int i = 0; i < {input_size}; i++) {{
        scanf("%f", &input[i]);
    }}
    
    // Run model
    model_forward(input, output);
    
    // Print output
    for (int i = 0; i < {output_size}; i++) {{
        printf("%.10f\\n", output[i]);
    }}
    
    return 0;
}}
'''
        
        test_c_path = os.path.join(tmpdir, 'test_main.c')
        with open(test_c_path, 'w') as f:
            f.write(test_c)
        
        # Compile
        exe_path = os.path.join(tmpdir, 'test_model')
        compile_cmd = [
            'gcc', '-O2', '-o', exe_path,
            test_c_path,
            os.path.join(tmpdir, 'model.c'),
            '-I', tmpdir,
            '-lm'
        ]
        
        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Compilation failed:\n{result.stderr}")
            raise RuntimeError(f"Compilation failed: {result.stderr}")
        
        # Run
        input_str = '\n'.join(f'{x:.10f}' for x in input_np)
        result = subprocess.run(
            [exe_path],
            input=input_str,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Execution failed: {result.stderr}")
        
        # Parse output
        output_lines = result.stdout.strip().split('\n')
        return np.array([float(x) for x in output_lines])
    
    def test_fuse_pass_numerical_correctness(self):
        """
        Test that FuseDequantQuantPass produces numerically identical results.
        
        When dequantize→quantize pairs have matching scales, removing them
        should not change the output at all.
        """
        print("\n" + "=" * 60)
        print("Test: FuseDequantQuantPass Numerical Correctness")
        print("=" * 60)
        
        # Create model
        torch.manual_seed(42)
        model = CustomMLP()
        model.eval()
        
        example_input = torch.randn(1, 784)
        input_size = 784
        output_size = 10
        
        # Define quantization rules with MATCHING scales at the boundary
        shared_scale = 0.01
        weight_scale = 0.005
        
        rules = [
            StaticQuantRule(
                pattern=r'.*encoder_fc1.*',
                dtype='int8',
                input_scale=0.02,
                input_offset=0,
                weight_scale=weight_scale,
                weight_offset=0,
                output_scale=shared_scale,  # Matches encoder_fc2's input
                output_offset=0
            ),
            StaticQuantRule(
                pattern=r'.*encoder_fc2.*',
                dtype='int8',
                input_scale=shared_scale,   # Matches encoder_fc1's output
                input_offset=0,
                weight_scale=weight_scale,
                weight_offset=0,
                output_scale=0.015,
                output_offset=0
            ),
        ]
        
        # ================================================================
        # UNOPTIMIZED: Compile without the pass
        # ================================================================
        print("\n[1] Building UNOPTIMIZED version...")
        
        ir_graph_unopt = compile_model(model, example_input, return_ir=True, verbose=False)
        quant_ir_unopt = QuantizationTransform(rules).apply(ir_graph_unopt)
        
        unopt_node_count = len(quant_ir_unopt.nodes)
        print(f"    Nodes: {unopt_node_count}")
        
        with tempfile.TemporaryDirectory() as tmpdir_unopt:
            printer_unopt = CPrinter(quant_ir_unopt)
            printer_unopt.generate_all(tmpdir_unopt)
            
            input_np = example_input.numpy().flatten()
            output_unopt = self._compile_and_run_c_model(
                tmpdir_unopt, input_np, input_size, output_size
            )
        
        print(f"    Output: {output_unopt[:5]}...")
        
        # ================================================================
        # OPTIMIZED: Compile with the pass
        # ================================================================
        print("\n[2] Building OPTIMIZED version...")
        
        ir_graph_opt = compile_model(model, example_input, return_ir=True, verbose=False)
        quant_ir_opt = QuantizationTransform(rules).apply(ir_graph_opt)
        
        # Apply the optimization pass
        fuse_pass = FuseDequantQuantPass(verbose=False)
        optimized_ir = fuse_pass.apply(quant_ir_opt)
        
        opt_node_count = len(optimized_ir.nodes)
        stats = fuse_pass.get_stats()
        print(f"    Nodes: {opt_node_count} (removed {stats['nodes_removed']})")
        print(f"    Pairs fused: {stats['pairs_fused']}")
        
        with tempfile.TemporaryDirectory() as tmpdir_opt:
            printer_opt = CPrinter(optimized_ir)
            printer_opt.generate_all(tmpdir_opt)
            
            output_opt = self._compile_and_run_c_model(
                tmpdir_opt, input_np, input_size, output_size
            )
        
        print(f"    Output: {output_opt[:5]}...")
        
        # ================================================================
        # Compare outputs
        # ================================================================
        print("\n[3] Comparing outputs...")
        
        abs_error = np.abs(output_unopt - output_opt)
        max_error = np.max(abs_error)
        mean_error = np.mean(abs_error)
        
        print(f"    Max absolute error:  {max_error:.2e}")
        print(f"    Mean absolute error: {mean_error:.2e}")
        
        # The outputs should be EXACTLY the same (or very close due to float precision)
        tolerance = 1e-5
        
        if max_error < tolerance:
            print(f"\n    ✓ PASSED: max_error ({max_error:.2e}) < tolerance ({tolerance})")
        else:
            print(f"\n    ✗ FAILED: max_error ({max_error:.2e}) >= tolerance ({tolerance})")
        
        assert max_error < tolerance, \
            f"Optimized output differs from unoptimized: max_error={max_error:.2e} >= {tolerance}"
        
        # Verify that optimization actually removed nodes
        assert stats['pairs_fused'] > 0, "Pass should have fused at least one pair"
        assert opt_node_count < unopt_node_count, "Optimized graph should have fewer nodes"
        
        print("\n" + "=" * 60)
        print("Test PASSED: Optimization is numerically correct!")
        print("=" * 60)
    
    def test_fuse_pass_skips_mismatched_scales(self):
        """
        Test that the pass correctly skips pairs with mismatched scales.
        """
        print("\n" + "=" * 60)
        print("Test: FuseDequantQuantPass Skips Mismatched Scales")
        print("=" * 60)
        
        torch.manual_seed(42)
        model = CustomMLP()
        model.eval()
        example_input = torch.randn(1, 784)
        
        # Define rules with DIFFERENT scales at the boundary
        rules = [
            StaticQuantRule(
                pattern=r'.*encoder_fc1.*',
                dtype='int8',
                input_scale=0.02,
                input_offset=0,
                weight_scale=0.005,
                weight_offset=0,
                output_scale=0.01,      # Different from encoder_fc2's input!
                output_offset=0
            ),
            StaticQuantRule(
                pattern=r'.*encoder_fc2.*',
                dtype='int8',
                input_scale=0.02,       # Different from encoder_fc1's output!
                input_offset=0,
                weight_scale=0.005,
                weight_offset=0,
                output_scale=0.015,
                output_offset=0
            ),
        ]
        
        ir_graph = compile_model(model, example_input, return_ir=True, verbose=False)
        quant_ir = QuantizationTransform(rules).apply(ir_graph)
        
        original_count = len(quant_ir.nodes)
        
        # Apply pass
        fuse_pass = FuseDequantQuantPass(verbose=True)
        optimized_ir = fuse_pass.apply(quant_ir)
        
        stats = fuse_pass.get_stats()
        
        print(f"\n    Pairs found (but not fusable): {stats['pairs_found']}")
        print(f"    Pairs fused: {stats['pairs_fused']}")
        print(f"    Nodes before: {original_count}")
        print(f"    Nodes after:  {len(optimized_ir.nodes)}")
        
        # With mismatched scales, nothing should be fused
        assert stats['pairs_fused'] == 0, \
            "Pass should not fuse pairs with mismatched scales"
        assert len(optimized_ir.nodes) == original_count, \
            "Node count should not change when scales don't match"
        
        print("\n    ✓ PASSED: Pass correctly skipped mismatched scales")
    
    def test_fuse_pass_preserves_non_quantized_layers(self):
        """
        Test that the pass doesn't affect non-quantized layers.
        """
        print("\n" + "=" * 60)
        print("Test: FuseDequantQuantPass Preserves Float Layers")
        print("=" * 60)
        
        torch.manual_seed(42)
        model = CustomMLP()
        model.eval()
        example_input = torch.randn(1, 784)
        
        # Only quantize encoder layers
        shared_scale = 0.01
        rules = [
            StaticQuantRule(
                pattern=r'.*encoder_fc1.*',
                dtype='int8',
                input_scale=0.02,
                input_offset=0,
                weight_scale=0.005,
                weight_offset=0,
                output_scale=shared_scale,
                output_offset=0
            ),
            StaticQuantRule(
                pattern=r'.*encoder_fc2.*',
                dtype='int8',
                input_scale=shared_scale,
                input_offset=0,
                weight_scale=0.005,
                weight_offset=0,
                output_scale=0.015,
                output_offset=0
            ),
        ]
        
        ir_graph = compile_model(model, example_input, return_ir=True, verbose=False)
        quant_ir = QuantizationTransform(rules).apply(ir_graph)
        
        # Count float linear layers before
        float_layers_before = sum(
            1 for n in quant_ir.nodes 
            if n.op_type == 'linear' and n.dtype == 'float32'
        )
        
        # Apply pass
        fuse_pass = FuseDequantQuantPass(verbose=False)
        optimized_ir = fuse_pass.apply(quant_ir)
        
        # Count float linear layers after
        float_layers_after = sum(
            1 for n in optimized_ir.nodes 
            if n.op_type == 'linear' and n.dtype == 'float32'
        )
        
        print(f"    Float linear layers before: {float_layers_before}")
        print(f"    Float linear layers after:  {float_layers_after}")
        
        assert float_layers_before == float_layers_after == 3, \
            "Float layers (precision_layer, output_fc1, output_fc2) should be preserved"
        
        print("\n    ✓ PASSED: Float layers preserved")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

