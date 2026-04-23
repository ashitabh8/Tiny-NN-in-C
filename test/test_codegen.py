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
    
    def test_codegen_conv1d_pointwise(self):
        """Conv1d (k=1) should emit a conv2d_nhwc call with k_h=1, in_h=1."""
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv1d(4, 6, kernel_size=1, bias=True)

            def forward(self, x):
                return self.conv(x)

        model = M().eval()
        example_input = torch.randn(1, 4, 8)
        fx_graph = trace_model(model, example_input)
        ir_graph = lower_fx_graph(fx_graph, example_input)

        model_c = CPrinter(ir_graph).generate_model_c()
        assert "conv2d_nhwc(" in model_c
        # The wrapper sets in_h=1 and k_h=1, k_w=1
        # Find the conv2d_nhwc call line and verify its 1-D wrap shape
        conv_line = [ln for ln in model_c.splitlines() if "conv2d_nhwc(" in ln][0]
        assert ", 1, 8, 4," in conv_line, conv_line  # in_h=1, in_w=L=8, in_c=4
        assert ", 1, 1, 6," in conv_line, conv_line  # k_h=1, k_w=1, out_c=6

    def test_codegen_conv1d_depthwise(self):
        """Depthwise Conv1d (groups=C, k=3) should emit depthwise_conv2d_nhwc with H=1, k_h=1, k_w=3."""
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv1d(8, 8, kernel_size=3, padding=1, groups=8, bias=False)

            def forward(self, x):
                return self.conv(x)

        model = M().eval()
        example_input = torch.randn(1, 8, 16)
        fx_graph = trace_model(model, example_input)
        ir_graph = lower_fx_graph(fx_graph, example_input)

        model_c = CPrinter(ir_graph).generate_model_c()
        assert "depthwise_conv2d_nhwc(" in model_c
        dw_line = [ln for ln in model_c.splitlines() if "depthwise_conv2d_nhwc(" in ln][0]
        # in_h=1, in_w=16, channels=8, ..., k_h=1, k_w=3, ..., stride_h=1, stride_w=1, pad_h=0, pad_w=1
        assert ", 1, 16, 8," in dw_line, dw_line
        assert ", 1, 3, " in dw_line, dw_line
        assert ", 1, 1, 0, 1," in dw_line, dw_line  # stride_h, stride_w, pad_h, pad_w

    def test_codegen_permute_3d(self):
        """A 4D PyTorch permute that strips to 3D should emit permute_3d (not permute_4d)."""
        class M(torch.nn.Module):
            def forward(self, x):
                # Permute that swaps non-batch dims: [B, C, I, S] -> [B, I, S, C]
                return x.permute(0, 2, 3, 1)

        model = M().eval()
        example_input = torch.randn(1, 4, 5, 6)
        fx_graph = trace_model(model, example_input)
        ir_graph = lower_fx_graph(fx_graph, example_input)

        model_c = CPrinter(ir_graph).generate_model_c()
        assert "permute_3d(" in model_c, model_c
        # 4D input strips to [4, 5, 6]; perm [0,2,3,1] strips/shifts to [1, 2, 0]
        perm_line = [ln for ln in model_c.splitlines() if "permute_3d(" in ln][0]
        assert ", 4, 5, 6," in perm_line, perm_line
        assert ", 1, 2, 0," in perm_line, perm_line

    def test_codegen_mean_dim_neg1_3d(self):
        """tensor.mean(dim=-1) on a 3D [B, T, I] tensor should emit mean_last_dim with rows=T, cols=I."""
        class M(torch.nn.Module):
            def forward(self, x):
                return x.mean(dim=-1)  # [B, T, I] -> [B, T]

        model = M().eval()
        example_input = torch.randn(1, 5, 7)
        fx_graph = trace_model(model, example_input)
        ir_graph = lower_fx_graph(fx_graph, example_input)

        model_c = CPrinter(ir_graph).generate_model_c()
        assert "mean_last_dim(" in model_c, model_c
        mean_line = [ln for ln in model_c.splitlines() if "mean_last_dim(" in ln][0]
        # rows=5 (T after batch strip), cols=7
        assert ", 5, 7," in mean_line, mean_line

    def test_codegen_batchnorm1d(self):
        """BatchNorm1d should emit batchnorm2d_nhwc with h=1."""
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm1d(8)

            def forward(self, x):
                return self.bn(x)

        model = M().eval()
        example_input = torch.randn(2, 8, 16)
        fx_graph = trace_model(model, example_input)
        ir_graph = lower_fx_graph(fx_graph, example_input)

        model_c = CPrinter(ir_graph).generate_model_c()
        assert "batchnorm2d_nhwc(" in model_c
        bn_line = [ln for ln in model_c.splitlines() if "batchnorm2d_nhwc(" in ln][0]
        assert ", 1, 16, 8," in bn_line, bn_line  # h=1, w=L=16, c=8

    def test_buffer_allocation(self):
        """Test that buffers are allocated via slots (flat, no nesting)."""
        model = TinyMLP()
        example_input = torch.randn(1, 784)
        
        fx_graph = trace_model(model, example_input)
        ir_graph = lower_fx_graph(fx_graph)
        
        printer = CPrinter(ir_graph)
        model_c = printer.generate_model_c()
        
        # Check that slot buffers are declared (interval graph coloring)
        assert "float slot_" in model_c
        # Linear chain (MLP) should need exactly 2 slots
        assert "float slot_0[" in model_c
        assert "float slot_1[" in model_c
        
        # Check that operations are called
        assert "dense(" in model_c or "linear" in model_c.lower()

