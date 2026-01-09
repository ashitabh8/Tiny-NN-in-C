"""
Integration tests for quantization pipeline (Phase 2.2)
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import os
import tempfile
import subprocess
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pytorch_to_c.compiler import compile_model
from src.pytorch_to_c.codegen.c_printer import CPrinter
from src.pytorch_to_c.quantization import (
    StaticQuantRule, DynamicQuantRuleMinMaxPerTensor,
    QuantizationTransform, StaticQuantLinearNode, DynamicQuantLinearNode,
    QuantizeNode, DequantizeNode
)


class TinyMLP(nn.Module):
    """Simple MLP for testing"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TestQuantizationTransform:
    """Test QuantizationTransform"""
    
    def test_transform_tiny_mlp(self):
        """Test applying quantization to TinyMLP"""
        model = TinyMLP()
        example_input = torch.randn(1, 784)
        
        # Compile to float IR
        ir_graph = compile_model(
            model=model,
            example_input=example_input,
            output_dir=None,
            verbose=False,
            return_ir=True
        )
        
        # Define quantization rules
        rules = [
            StaticQuantRule(
                r'.*fc.*', 'int8',
                input_scale=0.05, input_offset=0,
                weight_scale=0.05, weight_offset=0,
                output_scale=0.05, output_offset=0
            ),
        ]
        
        # Apply quantization
        transform = QuantizationTransform(rules)
        quant_ir = transform.apply(ir_graph)
        
        # Verify quantized nodes were created
        quant_linear_count = sum(
            1 for n in quant_ir.nodes 
            if isinstance(n, (StaticQuantLinearNode, DynamicQuantLinearNode))
        )
        assert quant_linear_count == 2, "Expected 2 quantized linear nodes (fc1, fc2)"
        
        # Verify conversion nodes were inserted
        quant_node_count = sum(
            1 for n in quant_ir.nodes 
            if n.op_type == 'quantize'
        )
        dequant_node_count = sum(
            1 for n in quant_ir.nodes 
            if n.op_type == 'dequantize'
        )
        assert quant_node_count > 0, "Expected at least one QuantizeNode"
        assert dequant_node_count > 0, "Expected at least one DequantizeNode"
    
    def test_weights_quantized(self):
        """Test that weights are quantized during transformation"""
        model = TinyMLP()
        example_input = torch.randn(1, 784)
        
        # Compile to float IR
        ir_graph = compile_model(
            model=model,
            example_input=example_input,
            output_dir=None,
            verbose=False,
            return_ir=True
        )
        
        # Get original weight dtypes
        original_dtypes = {}
        for name, param in ir_graph.parameters.items():
            original_dtypes[name] = param.dtype
            assert param.dtype == np.float32 or param.dtype == np.float64
        
        # Apply quantization
        rules = [StaticQuantRule(
            r'.*fc.*', 'int8',
            input_scale=0.05, input_offset=0,
            weight_scale=0.05, weight_offset=0,
            output_scale=0.05, output_offset=0
        )]
        transform = QuantizationTransform(rules)
        quant_ir = transform.apply(ir_graph)
        
        # Verify weights were quantized
        quantized_count = 0
        for name, param in quant_ir.parameters.items():
            if 'fc' in name and 'weight' in name:
                assert param.dtype == np.int8, f"Expected int8, got {param.dtype}"
                quantized_count += 1
        
        assert quantized_count >= 2, "Expected at least 2 quantized weight arrays"


class TestQuantizedCodeGeneration:
    """Test C code generation for quantized models"""
    
    def test_generate_quantized_c_code(self):
        """Test generating C code for quantized model"""
        model = TinyMLP()
        example_input = torch.randn(1, 784)
        
        # Compile and quantize
        ir_graph = compile_model(model, example_input, None, False, return_ir=True)
        rules = [StaticQuantRule(
            r'.*fc.*', 'int8',
            input_scale=0.05, input_offset=0,
            weight_scale=0.05, weight_offset=0,
            output_scale=0.05, output_offset=0
        )]
        transform = QuantizationTransform(rules)
        quant_ir = transform.apply(ir_graph)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate C code
            printer = CPrinter(quant_ir)
            printer.generate_all(tmpdir)
            
            # Check generated files
            assert os.path.exists(os.path.join(tmpdir, 'model.c'))
            assert os.path.exists(os.path.join(tmpdir, 'model.h'))
            assert os.path.exists(os.path.join(tmpdir, 'weights.h'))
            assert os.path.exists(os.path.join(tmpdir, 'nn_ops_float.h'))
            assert os.path.exists(os.path.join(tmpdir, 'nn_ops_int8.h'))
            
            # Check model.c includes int8 header
            with open(os.path.join(tmpdir, 'model.c')) as f:
                content = f.read()
                assert '#include "nn_ops_int8.h"' in content
                assert 'int8_t' in content  # Should have int8 buffers
                assert 'dense_int8' in content  # Should call quantized dense
    
    def test_quantized_weights_in_weights_h(self):
        """Test that weights.h contains int8 arrays"""
        model = TinyMLP()
        example_input = torch.randn(1, 784)
        
        # Compile and quantize
        ir_graph = compile_model(model, example_input, None, False, return_ir=True)
        rules = [StaticQuantRule(
            r'.*fc.*', 'int8',
            input_scale=0.05, input_offset=0,
            weight_scale=0.05, weight_offset=0,
            output_scale=0.05, output_offset=0
        )]
        transform = QuantizationTransform(rules)
        quant_ir = transform.apply(ir_graph)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            printer = CPrinter(quant_ir)
            printer.generate_all(tmpdir)
            
            with open(os.path.join(tmpdir, 'weights.h')) as f:
                content = f.read()
                assert 'int8_t' in content, "Expected int8_t arrays in weights.h"
                assert 'dtype: int8' in content, "Expected dtype comment in weights.h"


class TestQuantizedCCompilation:
    """Test that quantized C code compiles with gcc"""
    
    @pytest.fixture
    def gcc_available(self):
        """Check if gcc is available"""
        try:
            result = subprocess.run(['gcc', '--version'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def test_quantized_c_compiles(self, gcc_available):
        """Test that generated quantized C code compiles"""
        if not gcc_available:
            pytest.skip("gcc not available")
        
        model = TinyMLP()
        example_input = torch.randn(1, 784)
        
        # Compile and quantize
        ir_graph = compile_model(model, example_input, None, False, return_ir=True)
        rules = [StaticQuantRule(
            r'.*fc.*', 'int8',
            input_scale=0.05, input_offset=0,
            weight_scale=0.05, weight_offset=0,
            output_scale=0.05, output_offset=0
        )]
        transform = QuantizationTransform(rules)
        quant_ir = transform.apply(ir_graph)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate C code
            printer = CPrinter(quant_ir)
            printer.generate_all(tmpdir)
            
            # Compile with gcc (just check syntax, don't link)
            model_c = os.path.join(tmpdir, 'model.c')
            result = subprocess.run(
                ['gcc', '-c', '-Wall', '-Wextra', '-I', tmpdir, model_c, '-o', '/dev/null'],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"GCC stderr:\n{result.stderr}")
                print(f"\n--- model.c ---")
                with open(model_c) as f:
                    print(f.read())
            
            assert result.returncode == 0, f"GCC compilation failed:\n{result.stderr}"


class TestDynamicQuantization:
    """Test dynamic quantization rule"""
    
    def test_dynamic_quant_computes_scale(self):
        """Test that dynamic rule computes scale from weights"""
        model = TinyMLP()
        example_input = torch.randn(1, 784)
        
        # Compile to float IR
        ir_graph = compile_model(model, example_input, None, False, return_ir=True)
        
        # Find fc1 weight parameter (name format may vary)
        fc1_weight_name = None
        for name in ir_graph.parameters.keys():
            if 'fc1' in name and 'weight' in name:
                fc1_weight_name = name
                break
        
        assert fc1_weight_name is not None, f"fc1 weight not found. Params: {list(ir_graph.parameters.keys())}"
        
        fc1_weight = ir_graph.parameters[fc1_weight_name]
        w_min = float(np.min(fc1_weight))
        w_max = float(np.max(fc1_weight))
        
        # Apply dynamic quantization
        rules = [DynamicQuantRuleMinMaxPerTensor(r'.*fc.*', 'int8')]
        transform = QuantizationTransform(rules)
        quant_ir = transform.apply(ir_graph)
        
        # Verify quantized nodes have computed scale
        found_fc1 = False
        for node in quant_ir.nodes:
            if isinstance(node, DynamicQuantLinearNode) and 'fc1' in node.name:
                # Scale should be computed from weights
                assert hasattr(node, 'scale')
                assert node.scale > 0, "Scale should be positive"
                found_fc1 = True
        
        assert found_fc1, "Expected to find quantized fc1 node"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

