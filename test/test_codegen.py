"""
Tests for C code generation
"""

import pytest
import os
import tempfile
import torch

from src.pytorch_to_c.frontend.fx_tracer import trace_model
from src.pytorch_to_c.lowering.lower import lower_fx_graph
from src.pytorch_to_c.codegen.c_printer import CPrinter, generate_c_code
from test.test_models import TinyMLP


class TestCodegen:
    """Test C code generation."""
    
    def test_generate_weights_h(self):
        """Test weights.h generation."""
        model = TinyMLP(input_size=10, hidden_size=5, output_size=2)
        example_input = torch.randn(1, 10)
        
        fx_graph = trace_model(model, example_input)
        ir_graph = lower_fx_graph(fx_graph)
        
        printer = CPrinter(ir_graph)
        weights_h = printer.generate_weights_h()
        
        assert isinstance(weights_h, str)
        assert len(weights_h) > 0
        assert "#ifndef WEIGHTS_H_" in weights_h
        assert "#define WEIGHTS_H_" in weights_h
        assert "static const float" in weights_h
    
    def test_generate_model_h(self):
        """Test model.h generation."""
        model = TinyMLP()
        example_input = torch.randn(1, 784)
        
        fx_graph = trace_model(model, example_input)
        ir_graph = lower_fx_graph(fx_graph)
        
        printer = CPrinter(ir_graph)
        model_h = printer.generate_model_h()
        
        assert isinstance(model_h, str)
        assert len(model_h) > 0
        assert "#ifndef MODEL_H_" in model_h
        assert "void model_forward" in model_h
    
    def test_generate_model_c(self):
        """Test model.c generation."""
        model = TinyMLP()
        example_input = torch.randn(1, 784)
        
        fx_graph = trace_model(model, example_input)
        ir_graph = lower_fx_graph(fx_graph)
        
        printer = CPrinter(ir_graph)
        model_c = printer.generate_model_c()
        
        assert isinstance(model_c, str)
        assert len(model_c) > 0
        assert "#include \"model.h\"" in model_c
        assert "#include \"weights.h\"" in model_c
        assert "void model_forward" in model_c
    
    def test_generate_all_files(self):
        """Test that all files are generated to disk."""
        model = TinyMLP(input_size=10, hidden_size=5, output_size=2)
        example_input = torch.randn(1, 10)
        
        fx_graph = trace_model(model, example_input)
        ir_graph = lower_fx_graph(fx_graph)
        
        # Generate to temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_c_code(ir_graph, tmpdir)
            
            # Verify files exist
            assert os.path.exists(os.path.join(tmpdir, "model.h"))
            assert os.path.exists(os.path.join(tmpdir, "model.c"))
            assert os.path.exists(os.path.join(tmpdir, "weights.h"))
            
            # Verify files are non-empty
            assert os.path.getsize(os.path.join(tmpdir, "model.h")) > 0
            assert os.path.getsize(os.path.join(tmpdir, "model.c")) > 0
            assert os.path.getsize(os.path.join(tmpdir, "weights.h")) > 0
    
    def test_sanitize_names(self):
        """Test that names are properly sanitized for C."""
        model = TinyMLP()
        example_input = torch.randn(1, 784)
        
        fx_graph = trace_model(model, example_input)
        ir_graph = lower_fx_graph(fx_graph)
        
        printer = CPrinter(ir_graph)
        
        # Test sanitization
        assert printer._sanitize_name("fc1.weight") == "fc1_weight"
        assert printer._sanitize_name("layer-1") == "layer_1"
        assert printer._sanitize_name("1layer") == "_1layer"
    
    def test_buffer_allocation(self):
        """Test that buffers are allocated for nodes."""
        model = TinyMLP()
        example_input = torch.randn(1, 784)
        
        fx_graph = trace_model(model, example_input)
        ir_graph = lower_fx_graph(fx_graph)
        
        printer = CPrinter(ir_graph)
        model_c = printer.generate_model_c()
        
        # Check that buffers are declared
        assert "float buf_" in model_c
        
        # Check that operations are called
        assert "dense(" in model_c or "linear" in model_c.lower()

