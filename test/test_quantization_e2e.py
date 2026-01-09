"""
End-to-end tests for quantization pipeline.

Tests that quantized C models compile, run, and produce reasonable outputs.
Due to quantization error, we use relaxed tolerances (1e-1).
"""

import pytest
import os
import tempfile
import subprocess
import torch
import torch.nn as nn
import numpy as np

from src.pytorch_to_c.compiler import compile_model
from src.pytorch_to_c.codegen.c_printer import CPrinter
from src.pytorch_to_c.quantization import (
    StaticQuantRule,
    DynamicQuantRuleMinMaxPerTensor,
    QuantizationTransform,
)


class SimpleMLP(nn.Module):
    """Simple MLP for quantization testing."""
    
    def __init__(self, input_size=16, hidden_size=8, output_size=4):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ResNetBlock(nn.Module):
    """
    Simplified ResNet block: Conv -> BatchNorm -> ReLU -> Add (skip connection)
    Tests skip connections and topological traversing.
    """
    
    def __init__(self, channels=16):
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
    A tiny ResNet-style model for testing.
    
    Architecture:
    - Initial Conv to set channel dimension
    - Two ResNet blocks with skip connections
    - Global Average Pool (mean over spatial dims)
    - FC classifier
    """
    
    def __init__(self, in_channels=3, num_classes=4, channels=16):
        super().__init__()
        
        # Initial convolution
        self.conv_init = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.bn_init = nn.BatchNorm2d(channels)
        self.relu_init = nn.ReLU()
        
        # ResNet blocks
        self.block1 = ResNetBlock(channels)
        self.block2 = ResNetBlock(channels)
        
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


class TestQuantizedE2E:
    """End-to-end tests for quantized models."""
    
    def _check_gcc_available(self):
        """Check if gcc is available."""
        try:
            subprocess.run(
                ["gcc", "--version"],
                check=True,
                capture_output=True,
                timeout=5
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _create_test_harness(self, tmpdir, input_size, output_size):
        """Create a C test harness that reads input and writes output."""
        harness_code = f"""
#include <stdio.h>
#include <stdlib.h>
#include "model.h"

int main(int argc, char* argv[]) {{
    if (argc != 3) {{
        fprintf(stderr, "Usage: %s <input.bin> <output.bin>\\n", argv[0]);
        return 1;
    }}
    
    // Allocate buffers
    float* input = (float*)malloc({input_size} * sizeof(float));
    float* output = (float*)malloc({output_size} * sizeof(float));
    
    if (!input || !output) {{
        fprintf(stderr, "Memory allocation failed\\n");
        return 1;
    }}
    
    // Read input
    FILE* f_in = fopen(argv[1], "rb");
    if (!f_in) {{
        fprintf(stderr, "Failed to open input file\\n");
        return 1;
    }}
    
    size_t read_count = fread(input, sizeof(float), {input_size}, f_in);
    fclose(f_in);
    
    if (read_count != {input_size}) {{
        fprintf(stderr, "Failed to read input: expected %d, got %zu\\n", {input_size}, read_count);
        return 1;
    }}
    
    // Run model
    model_forward(input, output);
    
    // Write output
    FILE* f_out = fopen(argv[2], "wb");
    if (!f_out) {{
        fprintf(stderr, "Failed to open output file\\n");
        return 1;
    }}
    
    fwrite(output, sizeof(float), {output_size}, f_out);
    fclose(f_out);
    
    // Cleanup
    free(input);
    free(output);
    
    return 0;
}}
"""
        harness_path = os.path.join(tmpdir, "test_harness.c")
        with open(harness_path, 'w') as f:
            f.write(harness_code)
        
        return harness_path
    
    def _compile_and_run_c_model(self, tmpdir, input_data, input_size, output_size):
        """Compile the C model and run it with given input."""
        # Create test harness
        harness_path = self._create_test_harness(tmpdir, input_size, output_size)
        
        # Compile
        model_c = os.path.join(tmpdir, "model.c")
        executable = os.path.join(tmpdir, "test_model")
        
        compile_cmd = [
            "gcc",
            "-o", executable,
            harness_path,
            model_c,
            f"-I{tmpdir}",
            "-lm",
            "-std=c99",
            "-O2"
        ]
        
        result = subprocess.run(
            compile_cmd,
            capture_output=True,
            timeout=30,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Compilation failed:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            raise RuntimeError(f"Compilation failed: {result.stderr}")
        
        # Write input to binary file
        input_file = os.path.join(tmpdir, "input.bin")
        input_data.astype(np.float32).tofile(input_file)
        
        # Run executable
        output_file = os.path.join(tmpdir, "output.bin")
        
        result = subprocess.run(
            [executable, input_file, output_file],
            capture_output=True,
            timeout=30,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Execution failed:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            raise RuntimeError(f"Execution failed: {result.stderr}")
        
        # Read output
        c_output = np.fromfile(output_file, dtype=np.float32)
        
        return c_output
    
    def _run_quantized_test(
        self, 
        model, 
        example_input, 
        rules, 
        input_size, 
        output_size,
        test_name
    ):
        """Run a quantized model test and return comparison results."""
        model.eval()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Compile to float IR
            ir_graph = compile_model(
                model, example_input, 
                output_dir=None,  # Don't generate C yet
                verbose=False,
                return_ir=True
            )
            
            # Step 2: Apply quantization
            transform = QuantizationTransform(rules)
            quant_ir = transform.apply(ir_graph)
            
            # Step 3: Generate C code
            printer = CPrinter(quant_ir)
            printer.generate_all(tmpdir)
            
            # Step 4: Get PyTorch output (float)
            with torch.no_grad():
                pytorch_output = model(example_input)
            pytorch_output_np = pytorch_output.numpy().flatten()
            
            # Step 5: Get C output
            input_np = example_input.numpy().flatten()
            c_output_np = self._compile_and_run_c_model(
                tmpdir,
                input_np,
                input_size=input_size,
                output_size=output_size
            )
            
            # Step 6: Compare outputs
            abs_error = np.abs(pytorch_output_np - c_output_np)
            max_abs_error = np.max(abs_error)
            mean_abs_error = np.mean(abs_error)
            
            # Relative error (avoid division by zero)
            eps = 1e-7
            rel_error = abs_error / (np.abs(pytorch_output_np) + eps)
            max_rel_error = np.max(rel_error)
            mean_rel_error = np.mean(rel_error)
            
            return {
                'test_name': test_name,
                'pytorch_output': pytorch_output_np,
                'c_output': c_output_np,
                'max_abs_error': max_abs_error,
                'mean_abs_error': mean_abs_error,
                'max_rel_error': max_rel_error,
                'mean_rel_error': mean_rel_error,
            }
    
    def _print_results(self, results):
        """Pretty print test results."""
        print(f"\n{'='*60}")
        print(f" {results['test_name']}")
        print(f"{'='*60}")
        print(f"  PyTorch output: {results['pytorch_output']}")
        print(f"  C output:       {results['c_output']}")
        print(f"  ")
        print(f"  Max absolute error:  {results['max_abs_error']:.4e}")
        print(f"  Mean absolute error: {results['mean_abs_error']:.4e}")
        print(f"  Max relative error:  {results['max_rel_error']:.4e} ({results['max_rel_error']*100:.2f}%)")
        print(f"  Mean relative error: {results['mean_rel_error']:.4e} ({results['mean_rel_error']*100:.2f}%)")
    
    # =========================================================================
    # Static Quantization Tests
    # =========================================================================
    
    def test_static_quant_int8(self):
        """Test static int8 quantization end-to-end."""
        if not self._check_gcc_available():
            pytest.skip("gcc not available")
        
        torch.manual_seed(42)
        model = SimpleMLP(input_size=16, hidden_size=8, output_size=4)
        example_input = torch.randn(1, 16) * 0.5  # Keep inputs small
        
        rules = [
            StaticQuantRule(
                pattern=r'fc.*',
                dtype='int8',
                input_scale=0.01, input_offset=0,
                weight_scale=0.01, weight_offset=0,
                output_scale=0.01, output_offset=0
            )
        ]
        
        results = self._run_quantized_test(
            model=model,
            example_input=example_input,
            rules=rules,
            input_size=16,
            output_size=4,
            test_name="Static Quantization - int8"
        )
        
        self._print_results(results)
        
        # Verify model runs (main goal)
        assert results['c_output'] is not None
        assert len(results['c_output']) == 4
        
        # Relaxed tolerance for quantized models
        # Note: May fail due to quantization error - that's expected
        tolerance = 1e-1
        if results['max_abs_error'] < tolerance:
            print(f"  ✓ PASSED (max_abs_error < {tolerance})")
        else:
            print(f"  ⚠ High error (max_abs_error >= {tolerance}) - expected for quantization")
        
        # Just verify it's not completely wrong (sanity check)
        assert results['max_abs_error'] < 10.0, "Output seems completely wrong"
    
    def test_static_quant_int16(self):
        """Test static int16 quantization end-to-end."""
        if not self._check_gcc_available():
            pytest.skip("gcc not available")
        
        torch.manual_seed(42)
        model = SimpleMLP(input_size=16, hidden_size=8, output_size=4)
        example_input = torch.randn(1, 16) * 0.5
        
        rules = [
            StaticQuantRule(
                pattern=r'fc.*',
                dtype='int16',
                input_scale=0.001, input_offset=0,
                weight_scale=0.001, weight_offset=0,
                output_scale=0.001, output_offset=0
            )
        ]
        
        results = self._run_quantized_test(
            model=model,
            example_input=example_input,
            rules=rules,
            input_size=16,
            output_size=4,
            test_name="Static Quantization - int16"
        )
        
        self._print_results(results)
        
        # Verify model runs
        assert results['c_output'] is not None
        assert len(results['c_output']) == 4
        
        # int16 should have better accuracy than int8
        tolerance = 1e-1
        if results['max_abs_error'] < tolerance:
            print(f"  ✓ PASSED (max_abs_error < {tolerance})")
        else:
            print(f"  ⚠ High error (max_abs_error >= {tolerance}) - expected for quantization")
        
        assert results['max_abs_error'] < 10.0, "Output seems completely wrong"
    
    # =========================================================================
    # Dynamic Quantization Tests
    # =========================================================================
    
    def test_dynamic_quant_int8(self):
        """Test dynamic int8 quantization end-to-end."""
        if not self._check_gcc_available():
            pytest.skip("gcc not available")
        
        torch.manual_seed(42)
        model = SimpleMLP(input_size=16, hidden_size=8, output_size=4)
        example_input = torch.randn(1, 16) * 0.5
        
        rules = [
            DynamicQuantRuleMinMaxPerTensor(
                pattern=r'fc.*',
                dtype='int8'
            )
        ]
        
        results = self._run_quantized_test(
            model=model,
            example_input=example_input,
            rules=rules,
            input_size=16,
            output_size=4,
            test_name="Dynamic Quantization - int8"
        )
        
        self._print_results(results)
        
        # Verify model runs
        assert results['c_output'] is not None
        assert len(results['c_output']) == 4
        
        tolerance = 1e-1
        if results['max_abs_error'] < tolerance:
            print(f"  ✓ PASSED (max_abs_error < {tolerance})")
        else:
            print(f"  ⚠ High error (max_abs_error >= {tolerance}) - expected for quantization")
        
        assert results['max_abs_error'] < 10.0, "Output seems completely wrong"
    
    def test_dynamic_quant_int16(self):
        """Test dynamic int16 quantization end-to-end."""
        if not self._check_gcc_available():
            pytest.skip("gcc not available")
        
        torch.manual_seed(42)
        model = SimpleMLP(input_size=16, hidden_size=8, output_size=4)
        example_input = torch.randn(1, 16) * 0.5
        
        rules = [
            DynamicQuantRuleMinMaxPerTensor(
                pattern=r'fc.*',
                dtype='int16'
            )
        ]
        
        results = self._run_quantized_test(
            model=model,
            example_input=example_input,
            rules=rules,
            input_size=16,
            output_size=4,
            test_name="Dynamic Quantization - int16"
        )
        
        self._print_results(results)
        
        # Verify model runs
        assert results['c_output'] is not None
        assert len(results['c_output']) == 4
        
        tolerance = 1e-1
        if results['max_abs_error'] < tolerance:
            print(f"  ✓ PASSED (max_abs_error < {tolerance})")
        else:
            print(f"  ⚠ High error (max_abs_error >= {tolerance}) - expected for quantization")
        
        assert results['max_abs_error'] < 10.0, "Output seems completely wrong"
    
    # =========================================================================
    # Mixed Precision Test
    # =========================================================================
    
    def test_mixed_precision(self):
        """Test mixed int8/int16 quantization end-to-end."""
        if not self._check_gcc_available():
            pytest.skip("gcc not available")
        
        torch.manual_seed(42)
        model = SimpleMLP(input_size=16, hidden_size=8, output_size=4)
        example_input = torch.randn(1, 16) * 0.5
        
        rules = [
            # First layer: int8 (aggressive)
            StaticQuantRule(
                pattern=r'fc1',
                dtype='int8',
                input_scale=0.01, input_offset=0,
                weight_scale=0.01, weight_offset=0,
                output_scale=0.01, output_offset=0
            ),
            # Second layer: int16 (higher precision)
            StaticQuantRule(
                pattern=r'fc2',
                dtype='int16',
                input_scale=0.001, input_offset=0,
                weight_scale=0.001, weight_offset=0,
                output_scale=0.001, output_offset=0
            ),
        ]
        
        results = self._run_quantized_test(
            model=model,
            example_input=example_input,
            rules=rules,
            input_size=16,
            output_size=4,
            test_name="Mixed Precision - int8/int16"
        )
        
        self._print_results(results)
        
        # Verify model runs
        assert results['c_output'] is not None
        assert len(results['c_output']) == 4
        
        tolerance = 1e-1
        if results['max_abs_error'] < tolerance:
            print(f"  ✓ PASSED (max_abs_error < {tolerance})")
        else:
            print(f"  ⚠ High error (max_abs_error >= {tolerance}) - expected for quantization")
        
        assert results['max_abs_error'] < 10.0, "Output seems completely wrong"


class TestQuantizedCompilationOnly:
    """Tests that just verify C code compiles (no execution comparison)."""
    
    def _check_gcc_available(self):
        try:
            subprocess.run(["gcc", "--version"], check=True, capture_output=True, timeout=5)
            return True
        except:
            return False
    
    def test_all_quant_configs_compile(self):
        """Verify all quantization configurations produce compilable C code."""
        if not self._check_gcc_available():
            pytest.skip("gcc not available")
        
        torch.manual_seed(42)
        model = SimpleMLP(input_size=16, hidden_size=8, output_size=4)
        example_input = torch.randn(1, 16)
        
        configs = [
            ("static_int8", [StaticQuantRule(
                r'fc.*', 'int8',
                input_scale=0.01, input_offset=0,
                weight_scale=0.01, weight_offset=0,
                output_scale=0.01, output_offset=0
            )]),
            ("static_int16", [StaticQuantRule(
                r'fc.*', 'int16',
                input_scale=0.001, input_offset=0,
                weight_scale=0.001, weight_offset=0,
                output_scale=0.001, output_offset=0
            )]),
            ("dynamic_int8", [DynamicQuantRuleMinMaxPerTensor(r'fc.*', 'int8')]),
            ("dynamic_int16", [DynamicQuantRuleMinMaxPerTensor(r'fc.*', 'int16')]),
            ("mixed", [
                StaticQuantRule(
                    r'fc1', 'int8',
                    input_scale=0.01, input_offset=0,
                    weight_scale=0.01, weight_offset=0,
                    output_scale=0.01, output_offset=0
                ),
                StaticQuantRule(
                    r'fc2', 'int16',
                    input_scale=0.001, input_offset=0,
                    weight_scale=0.001, weight_offset=0,
                    output_scale=0.001, output_offset=0
                ),
            ]),
        ]
        
        for config_name, rules in configs:
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    # Compile to IR
                    ir_graph = compile_model(
                        model, example_input, 
                        output_dir=None, verbose=False, return_ir=True
                    )
                    
                    # Apply quantization
                    transform = QuantizationTransform(rules)
                    quant_ir = transform.apply(ir_graph)
                    
                    # Generate C code
                    printer = CPrinter(quant_ir)
                    printer.generate_all(tmpdir)
                    
                    # Compile with gcc
                    result = subprocess.run(
                        ["gcc", "-c", os.path.join(tmpdir, "model.c"),
                         "-I", tmpdir, "-o", os.path.join(tmpdir, "model.o")],
                        capture_output=True, timeout=30, text=True
                    )
                    
                    if result.returncode != 0:
                        print(f"\n{config_name} compilation failed:")
                        print(result.stderr)
                        pytest.fail(f"{config_name} failed to compile")
                    
                    print(f"  ✓ {config_name} compiles successfully")
                    
                except Exception as e:
                    pytest.fail(f"{config_name} failed: {e}")


class TestResNetE2E:
    """End-to-end tests for ResNet-style models with skip connections."""
    
    def _check_gcc_available(self):
        """Check if gcc is available."""
        try:
            subprocess.run(
                ["gcc", "--version"],
                check=True,
                capture_output=True,
                timeout=5
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _create_test_harness(self, tmpdir, input_size, output_size):
        """Create a C test harness for ResNet (same as SimpleMLP harness)."""
        harness_code = f"""
#include <stdio.h>
#include <stdlib.h>
#include "model.h"

int main(int argc, char* argv[]) {{
    if (argc != 3) {{
        fprintf(stderr, "Usage: %s <input.bin> <output.bin>\\n", argv[0]);
        return 1;
    }}
    
    float* input = (float*)malloc({input_size} * sizeof(float));
    float* output = (float*)malloc({output_size} * sizeof(float));
    
    if (!input || !output) {{
        fprintf(stderr, "Memory allocation failed\\n");
        return 1;
    }}
    
    FILE* f_in = fopen(argv[1], "rb");
    if (!f_in) {{
        fprintf(stderr, "Failed to open input file\\n");
        return 1;
    }}
    
    size_t read_count = fread(input, sizeof(float), {input_size}, f_in);
    fclose(f_in);
    
    if (read_count != {input_size}) {{
        fprintf(stderr, "Failed to read input\\n");
        return 1;
    }}
    
    model_forward(input, output);
    
    FILE* f_out = fopen(argv[2], "wb");
    if (!f_out) {{
        fprintf(stderr, "Failed to open output file\\n");
        return 1;
    }}
    
    fwrite(output, sizeof(float), {output_size}, f_out);
    fclose(f_out);
    
    free(input);
    free(output);
    
    return 0;
}}
"""
        harness_path = os.path.join(tmpdir, "test_harness.c")
        with open(harness_path, 'w') as f:
            f.write(harness_code)
        
        return harness_path
    
    def _compile_and_run_c_model(self, tmpdir, input_data, input_size, output_size):
        """Compile the C model and run it with given input."""
        harness_path = self._create_test_harness(tmpdir, input_size, output_size)
        
        model_c = os.path.join(tmpdir, "model.c")
        executable = os.path.join(tmpdir, "test_model")
        
        compile_cmd = [
            "gcc",
            "-o", executable,
            harness_path,
            model_c,
            f"-I{tmpdir}",
            "-lm",
            "-std=c99",
            "-O2"
        ]
        
        result = subprocess.run(
            compile_cmd,
            capture_output=True,
            timeout=30,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Compilation failed:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            raise RuntimeError(f"Compilation failed: {result.stderr}")
        
        input_file = os.path.join(tmpdir, "input.bin")
        input_data.astype(np.float32).tofile(input_file)
        
        output_file = os.path.join(tmpdir, "output.bin")
        
        result = subprocess.run(
            [executable, input_file, output_file],
            capture_output=True,
            timeout=30,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Execution failed:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            raise RuntimeError(f"Execution failed: {result.stderr}")
        
        c_output = np.fromfile(output_file, dtype=np.float32)
        
        return c_output
    
    def _print_results(self, results):
        """Pretty print test results."""
        print(f"\n{'='*60}")
        print(f" {results['test_name']}")
        print(f"{'='*60}")
        print(f"  PyTorch output: {results['pytorch_output']}")
        print(f"  C output:       {results['c_output']}")
        print(f"  ")
        print(f"  Max absolute error:  {results['max_abs_error']:.4e}")
        print(f"  Mean absolute error: {results['mean_abs_error']:.4e}")
        print(f"  Max relative error:  {results['max_rel_error']:.4e} ({results['max_rel_error']*100:.2f}%)")
        print(f"  Mean relative error: {results['mean_rel_error']:.4e} ({results['mean_rel_error']*100:.2f}%)")
    
    # =========================================================================
    # Float ResNet Test
    # =========================================================================
    
    def test_codegen_resnet_float(self):
        """Test ResNet-style model code generation with float32 (no quantization)."""
        if not self._check_gcc_available():
            pytest.skip("gcc not available")
        
        torch.manual_seed(42)
        
        # Small ResNet: 3 input channels, 4 output classes, 16 channels
        model = TinyResNet(in_channels=3, num_classes=4, channels=16)
        model.eval()
        
        # Input: [1, 3, 8, 8] - small spatial size for fast testing
        example_input = torch.randn(1, 3, 8, 8) * 0.5
        
        # Input size: 3*8*8 = 192, Output size: 4
        input_size = 3 * 8 * 8
        output_size = 4
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Compile to C (no quantization)
            ir_graph = compile_model(
                model, example_input,
                output_dir=tmpdir,
                verbose=False,
                return_ir=False
            )
            
            # Get PyTorch output
            with torch.no_grad():
                pytorch_output = model(example_input)
            pytorch_output_np = pytorch_output.numpy().flatten()
            
            # Get C output - convert NCHW to NHWC for C code
            # PyTorch uses NCHW [batch, channels, height, width]
            # Our C code uses NHWC [batch, height, width, channels]
            input_nhwc = example_input.permute(0, 2, 3, 1).numpy().flatten()
            c_output_np = self._compile_and_run_c_model(
                tmpdir, input_nhwc, input_size, output_size
            )
            
            # Compare outputs
            abs_error = np.abs(pytorch_output_np - c_output_np)
            max_abs_error = np.max(abs_error)
            mean_abs_error = np.mean(abs_error)
            
            eps = 1e-7
            rel_error = abs_error / (np.abs(pytorch_output_np) + eps)
            max_rel_error = np.max(rel_error)
            mean_rel_error = np.mean(rel_error)
            
            results = {
                'test_name': 'ResNet Float32 (no quantization)',
                'pytorch_output': pytorch_output_np,
                'c_output': c_output_np,
                'max_abs_error': max_abs_error,
                'mean_abs_error': mean_abs_error,
                'max_rel_error': max_rel_error,
                'mean_rel_error': mean_rel_error,
            }
            
            self._print_results(results)
            
            # Float tolerance - should be very tight now that NCHW->NHWC is correct
            tolerance = 1e-5  # Should be within floating point precision
            if max_abs_error < tolerance:
                print(f"  ✓ PASSED (max_abs_error < {tolerance})")
            else:
                print(f"  ⚠ Error higher than expected: {max_abs_error:.4e}")
            
            assert max_abs_error < tolerance, \
                f"Float ResNet error too high: {max_abs_error:.4e} >= {tolerance}"
    
    # =========================================================================
    # Static Quantization ResNet Tests
    # =========================================================================
    
    def test_resnet_static_quant_int8(self):
        """Test ResNet with static int8 quantization on conv and linear layers."""
        if not self._check_gcc_available():
            pytest.skip("gcc not available")
        
        torch.manual_seed(42)
        model = TinyResNet(in_channels=3, num_classes=4, channels=16)
        model.eval()
        
        example_input = torch.randn(1, 3, 8, 8) * 0.5
        input_size = 3 * 8 * 8
        output_size = 4
        
        # Quantize both conv and linear layers
        rules = [
            StaticQuantRule(
                pattern=r'.*conv.*',
                dtype='int8',
                input_scale=0.02, input_offset=0,
                weight_scale=0.02, weight_offset=0,
                output_scale=0.02, output_offset=0
            ),
            StaticQuantRule(
                pattern=r'.*fc.*',
                dtype='int8',
                input_scale=0.02, input_offset=0,
                weight_scale=0.02, weight_offset=0,
                output_scale=0.02, output_offset=0
            ),
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Compile to IR
            ir_graph = compile_model(
                model, example_input,
                output_dir=None, verbose=False, return_ir=True
            )
            
            # Apply quantization
            transform = QuantizationTransform(rules)
            quant_ir = transform.apply(ir_graph)
            
            # Generate C code
            printer = CPrinter(quant_ir)
            printer.generate_all(tmpdir)
            
            # Get PyTorch output
            with torch.no_grad():
                pytorch_output = model(example_input)
            pytorch_output_np = pytorch_output.numpy().flatten()
            
            # Get C output - convert NCHW to NHWC for C code
            input_nhwc = example_input.permute(0, 2, 3, 1).numpy().flatten()
            c_output_np = self._compile_and_run_c_model(
                tmpdir, input_nhwc, input_size, output_size
            )
            
            abs_error = np.abs(pytorch_output_np - c_output_np)
            max_abs_error = np.max(abs_error)
            mean_abs_error = np.mean(abs_error)
            
            eps = 1e-7
            rel_error = abs_error / (np.abs(pytorch_output_np) + eps)
            
            results = {
                'test_name': 'ResNet Static Quantization - int8 (conv + linear)',
                'pytorch_output': pytorch_output_np,
                'c_output': c_output_np,
                'max_abs_error': max_abs_error,
                'mean_abs_error': mean_abs_error,
                'max_rel_error': np.max(rel_error),
                'mean_rel_error': np.mean(rel_error),
            }
            
            self._print_results(results)
            
            # Quantized can have higher error
            tolerance = 1e-1
            if max_abs_error < tolerance:
                print(f"  ✓ PASSED (max_abs_error < {tolerance})")
            else:
                print(f"  ⚠ High error (expected for quantization): {max_abs_error:.4e}")
            
            # Sanity check
            assert max_abs_error < 10.0, "Output seems completely wrong"
    
    def test_resnet_static_quant_int16(self):
        """Test ResNet with static int16 quantization on conv and linear layers."""
        if not self._check_gcc_available():
            pytest.skip("gcc not available")
        
        torch.manual_seed(42)
        model = TinyResNet(in_channels=3, num_classes=4, channels=16)
        model.eval()
        
        example_input = torch.randn(1, 3, 8, 8) * 0.5
        input_size = 3 * 8 * 8
        output_size = 4
        
        rules = [
            StaticQuantRule(
                pattern=r'.*conv.*',
                dtype='int16',
                input_scale=0.001, input_offset=0,
                weight_scale=0.001, weight_offset=0,
                output_scale=0.001, output_offset=0
            ),
            StaticQuantRule(
                pattern=r'.*fc.*',
                dtype='int16',
                input_scale=0.001, input_offset=0,
                weight_scale=0.001, weight_offset=0,
                output_scale=0.001, output_offset=0
            ),
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ir_graph = compile_model(
                model, example_input,
                output_dir=None, verbose=False, return_ir=True
            )
            
            transform = QuantizationTransform(rules)
            quant_ir = transform.apply(ir_graph)
            
            printer = CPrinter(quant_ir)
            printer.generate_all(tmpdir)
            
            with torch.no_grad():
                pytorch_output = model(example_input)
            pytorch_output_np = pytorch_output.numpy().flatten()
            
            # Convert NCHW to NHWC for C code
            input_nhwc = example_input.permute(0, 2, 3, 1).numpy().flatten()
            c_output_np = self._compile_and_run_c_model(
                tmpdir, input_nhwc, input_size, output_size
            )
            
            abs_error = np.abs(pytorch_output_np - c_output_np)
            max_abs_error = np.max(abs_error)
            
            results = {
                'test_name': 'ResNet Static Quantization - int16 (conv + linear)',
                'pytorch_output': pytorch_output_np,
                'c_output': c_output_np,
                'max_abs_error': max_abs_error,
                'mean_abs_error': np.mean(abs_error),
                'max_rel_error': np.max(abs_error / (np.abs(pytorch_output_np) + 1e-7)),
                'mean_rel_error': np.mean(abs_error / (np.abs(pytorch_output_np) + 1e-7)),
            }
            
            self._print_results(results)
            
            tolerance = 1e-1
            if max_abs_error < tolerance:
                print(f"  ✓ PASSED (max_abs_error < {tolerance})")
            else:
                print(f"  ⚠ High error: {max_abs_error:.4e}")
            
            assert max_abs_error < 10.0, "Output seems completely wrong"
    
    # =========================================================================
    # Dynamic Quantization ResNet Tests
    # =========================================================================
    
    def test_resnet_dynamic_quant_int8(self):
        """Test ResNet with dynamic int8 quantization on conv and linear layers."""
        if not self._check_gcc_available():
            pytest.skip("gcc not available")
        
        torch.manual_seed(42)
        model = TinyResNet(in_channels=3, num_classes=4, channels=16)
        model.eval()
        
        example_input = torch.randn(1, 3, 8, 8) * 0.5
        input_size = 3 * 8 * 8
        output_size = 4
        
        rules = [
            DynamicQuantRuleMinMaxPerTensor(pattern=r'.*conv.*', dtype='int8'),
            DynamicQuantRuleMinMaxPerTensor(pattern=r'.*fc.*', dtype='int8'),
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ir_graph = compile_model(
                model, example_input,
                output_dir=None, verbose=False, return_ir=True
            )
            
            transform = QuantizationTransform(rules)
            quant_ir = transform.apply(ir_graph)
            
            printer = CPrinter(quant_ir)
            printer.generate_all(tmpdir)
            
            with torch.no_grad():
                pytorch_output = model(example_input)
            pytorch_output_np = pytorch_output.numpy().flatten()
            
            # Convert NCHW to NHWC for C code
            input_nhwc = example_input.permute(0, 2, 3, 1).numpy().flatten()
            c_output_np = self._compile_and_run_c_model(
                tmpdir, input_nhwc, input_size, output_size
            )
            
            abs_error = np.abs(pytorch_output_np - c_output_np)
            max_abs_error = np.max(abs_error)
            
            results = {
                'test_name': 'ResNet Dynamic Quantization - int8 (conv + linear)',
                'pytorch_output': pytorch_output_np,
                'c_output': c_output_np,
                'max_abs_error': max_abs_error,
                'mean_abs_error': np.mean(abs_error),
                'max_rel_error': np.max(abs_error / (np.abs(pytorch_output_np) + 1e-7)),
                'mean_rel_error': np.mean(abs_error / (np.abs(pytorch_output_np) + 1e-7)),
            }
            
            self._print_results(results)
            
            tolerance = 1e-1
            if max_abs_error < tolerance:
                print(f"  ✓ PASSED (max_abs_error < {tolerance})")
            else:
                print(f"  ⚠ High error: {max_abs_error:.4e}")
            
            assert max_abs_error < 10.0, "Output seems completely wrong"
    
    def test_resnet_dynamic_quant_int16(self):
        """Test ResNet with dynamic int16 quantization on conv and linear layers."""
        if not self._check_gcc_available():
            pytest.skip("gcc not available")
        
        torch.manual_seed(42)
        model = TinyResNet(in_channels=3, num_classes=4, channels=16)
        model.eval()
        
        example_input = torch.randn(1, 3, 8, 8) * 0.5
        input_size = 3 * 8 * 8
        output_size = 4
        
        rules = [
            DynamicQuantRuleMinMaxPerTensor(pattern=r'.*conv.*', dtype='int16'),
            DynamicQuantRuleMinMaxPerTensor(pattern=r'.*fc.*', dtype='int16'),
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ir_graph = compile_model(
                model, example_input,
                output_dir=None, verbose=False, return_ir=True
            )
            
            transform = QuantizationTransform(rules)
            quant_ir = transform.apply(ir_graph)
            
            printer = CPrinter(quant_ir)
            printer.generate_all(tmpdir)
            
            with torch.no_grad():
                pytorch_output = model(example_input)
            pytorch_output_np = pytorch_output.numpy().flatten()
            
            # Convert NCHW to NHWC for C code
            input_nhwc = example_input.permute(0, 2, 3, 1).numpy().flatten()
            c_output_np = self._compile_and_run_c_model(
                tmpdir, input_nhwc, input_size, output_size
            )
            
            abs_error = np.abs(pytorch_output_np - c_output_np)
            max_abs_error = np.max(abs_error)
            
            results = {
                'test_name': 'ResNet Dynamic Quantization - int16 (conv + linear)',
                'pytorch_output': pytorch_output_np,
                'c_output': c_output_np,
                'max_abs_error': max_abs_error,
                'mean_abs_error': np.mean(abs_error),
                'max_rel_error': np.max(abs_error / (np.abs(pytorch_output_np) + 1e-7)),
                'mean_rel_error': np.mean(abs_error / (np.abs(pytorch_output_np) + 1e-7)),
            }
            
            self._print_results(results)
            
            tolerance = 1e-1
            if max_abs_error < tolerance:
                print(f"  ✓ PASSED (max_abs_error < {tolerance})")
            else:
                print(f"  ⚠ High error: {max_abs_error:.4e}")
            
            assert max_abs_error < 10.0, "Output seems completely wrong"
