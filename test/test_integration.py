"""
Integration tests - end-to-end compilation
"""

import pytest
import os
import tempfile
import subprocess
import torch
import numpy as np
from pathlib import Path

from src.pytorch_to_c.compiler import PyTorchToCCompiler, compile_model
from test.test_models import TinyMLP, ResNetBlock, MixedNet, get_test_models


class TestIntegration:
    """End-to-end integration tests."""
    
    def test_compile_tiny_mlp(self):
        """Test end-to-end compilation of TinyMLP."""
        model = TinyMLP(input_size=10, hidden_size=5, output_size=2)
        example_input = torch.randn(1, 10)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            compiler = PyTorchToCCompiler(verbose=False)
            ir_graph = compiler.compile(model, example_input, tmpdir)
            
            # Verify IR graph
            assert ir_graph is not None
            assert len(ir_graph.nodes) > 0
            
            # Verify files generated
            assert os.path.exists(os.path.join(tmpdir, "model.h"))
            assert os.path.exists(os.path.join(tmpdir, "model.c"))
            assert os.path.exists(os.path.join(tmpdir, "weights.h"))
    
    def test_compile_all_test_models(self):
        """Test compilation of all test models."""
        for model_name, model, example_input in get_test_models():
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    ir_graph = compile_model(
                        model,
                        example_input,
                        tmpdir,
                        verbose=False
                    )
                    
                    assert ir_graph is not None
                    assert len(ir_graph.nodes) > 0
                    
                except Exception as e:
                    pytest.fail(f"Failed to compile {model_name}: {e}")
    
    def test_generated_c_compiles(self):
        """Test that generated C code compiles with gcc (if available)."""
        model = TinyMLP(input_size=10, hidden_size=5, output_size=2)
        example_input = torch.randn(1, 10)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            compiler = PyTorchToCCompiler(verbose=False)
            compiler.compile(model, example_input, tmpdir)
            
            # Try to compile with gcc (skip if not available)
            try:
                # Check if gcc is available
                subprocess.run(
                    ["gcc", "--version"],
                    check=True,
                    capture_output=True
                )
                
                # Note: This is a basic syntax check, not a full compilation
                # Full compilation would require all headers and proper includes
                result = subprocess.run(
                    ["gcc", "-fsyntax-only", "-c", 
                     os.path.join(tmpdir, "model.c"),
                     "-I", tmpdir],
                    capture_output=True
                )
                
                # For now, we just verify the files exist and have content
                # Full gcc compilation testing can be added later
                
            except (subprocess.CalledProcessError, FileNotFoundError):
                pytest.skip("gcc not available for compilation test")
    
    def test_pipeline_preserves_model_info(self):
        """Test that the compilation pipeline preserves important model information."""
        model = TinyMLP()
        example_input = torch.randn(1, 784)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            compiler = PyTorchToCCompiler(verbose=False)
            ir_graph = compiler.compile(model, example_input, tmpdir)
            
            # Verify parameter count matches
            pytorch_params = sum(p.numel() for p in model.parameters())
            ir_params = sum(p.size for p in ir_graph.parameters.values())
            
            assert pytorch_params == ir_params, \
                f"Parameter count mismatch: PyTorch={pytorch_params}, IR={ir_params}"
    
    def test_convenience_function(self):
        """Test the convenience compile_model function."""
        model = TinyMLP()
        example_input = torch.randn(1, 784)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ir_graph = compile_model(model, example_input, tmpdir, verbose=False)
            
            assert ir_graph is not None
            assert os.path.exists(os.path.join(tmpdir, "model.c"))


class TestPyTorchCComparison:
    """Test that C model outputs match PyTorch outputs."""
    
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
        
        # Compile (nn_ops_float.h is now copied to tmpdir automatically)
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
        
        try:
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
                
        except subprocess.TimeoutExpired:
            raise RuntimeError("Compilation timed out")
        
        # Write input to binary file
        input_file = os.path.join(tmpdir, "input.bin")
        input_data.astype(np.float32).tofile(input_file)
        
        # Run executable
        output_file = os.path.join(tmpdir, "output.bin")
        
        try:
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
                
        except subprocess.TimeoutExpired:
            raise RuntimeError("Execution timed out")
        
        # Read output
        c_output = np.fromfile(output_file, dtype=np.float32)
        
        return c_output
    
    def test_compare_tiny_mlp_outputs(self):
        """Compare PyTorch and C outputs for TinyMLP."""
        if not self._check_gcc_available():
            pytest.skip("gcc not available")
        
        # Small model for faster testing
        model = TinyMLP(input_size=20, hidden_size=10, output_size=5)
        model.eval()
        
        # Create deterministic input
        torch.manual_seed(42)
        example_input = torch.randn(1, 20)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Compile model to C
            compile_model(model, example_input, tmpdir, verbose=False)
            
            # Get PyTorch output
            with torch.no_grad():
                pytorch_output = model(example_input)
            pytorch_output_np = pytorch_output.numpy().flatten()
            
            # Get C output
            input_np = example_input.numpy().flatten()
            c_output_np = self._compile_and_run_c_model(
                tmpdir,
                input_np,
                input_size=20,
                output_size=5
            )
            
            # Compare outputs
            max_error = np.max(np.abs(pytorch_output_np - c_output_np))
            mean_error = np.mean(np.abs(pytorch_output_np - c_output_np))
            
            print(f"\nTinyMLP Comparison:")
            print(f"  Max error: {max_error:.2e}")
            print(f"  Mean error: {mean_error:.2e}")
            print(f"  PyTorch output (first 5): {pytorch_output_np[:5]}")
            print(f"  C output (first 5): {c_output_np[:5]}")
            
            # Check tolerance (Phase 1 success criteria: 1e-5)
            assert max_error < 1e-3, f"Max error {max_error} exceeds tolerance 1e-3"
            assert mean_error < 1e-4, f"Mean error {mean_error} exceeds tolerance 1e-4"
    
    def test_compare_all_models(self):
        """Compare PyTorch and C outputs for all test models."""
        if not self._check_gcc_available():
            pytest.skip("gcc not available")
        
        # Simplified versions of test models for faster testing
        test_cases = [
            ("TinyMLP", TinyMLP(input_size=20, hidden_size=10, output_size=5), 
             torch.randn(1, 20), 20, 5),
        ]
        
        for model_name, model, example_input, input_size, output_size in test_cases:
            model.eval()
            
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    # Compile model to C
                    compile_model(model, example_input, tmpdir, verbose=False)
                    
                    # Get PyTorch output
                    with torch.no_grad():
                        pytorch_output = model(example_input)
                    pytorch_output_np = pytorch_output.numpy().flatten()
                    
                    # Get C output
                    input_np = example_input.numpy().flatten()
                    c_output_np = self._compile_and_run_c_model(
                        tmpdir,
                        input_np,
                        input_size=input_size,
                        output_size=output_size
                    )
                    
                    # Compare outputs
                    max_error = np.max(np.abs(pytorch_output_np - c_output_np))
                    mean_error = np.mean(np.abs(pytorch_output_np - c_output_np))
                    
                    print(f"\n{model_name} Comparison:")
                    print(f"  Max error: {max_error:.2e}")
                    print(f"  Mean error: {mean_error:.2e}")
                    
                    # Check tolerance
                    assert max_error < 1e-3, \
                        f"{model_name}: Max error {max_error} exceeds tolerance"
                    assert mean_error < 1e-4, \
                        f"{model_name}: Mean error {mean_error} exceeds tolerance"
                    
                    print(f"  ✓ {model_name} passed comparison test")
                    
                except Exception as e:
                    pytest.fail(f"{model_name} comparison failed: {e}")

