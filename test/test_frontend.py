"""
Tests for the torch.fx frontend
"""

import pytest
import torch
import torch.nn as nn

from src.pytorch_to_c.frontend.fx_tracer import FXTracer, trace_model
from test.test_models import TinyMLP, ResNetBlock, MixedNet


class TestFXTracer:
    """Test the FX tracing frontend."""
    
    def test_trace_tiny_mlp(self):
        """Test tracing a simple MLP."""
        model = TinyMLP()
        example_input = torch.randn(1, 784)
        
        tracer = FXTracer()
        traced = tracer.trace_model(model, example_input)
        
        assert traced is not None
        assert len(list(traced.graph.nodes)) > 0
        
        # Verify trace produces same output
        model.eval()
        with torch.no_grad():
            expected = model(example_input)
            actual = traced(example_input)
            assert torch.allclose(expected, actual, rtol=1e-5)
    
    def test_trace_resnet_block(self):
        """Test tracing a ResNet block with skip connection."""
        model = ResNetBlock()
        example_input = torch.randn(1, 64, 32, 32)
        
        traced = trace_model(model, example_input)
        
        assert traced is not None
        
        # Verify output matches
        model.eval()
        with torch.no_grad():
            expected = model(example_input)
            actual = traced(example_input)
            assert torch.allclose(expected, actual, rtol=1e-5)
    
    def test_trace_mixed_net(self):
        """Test tracing a network with mixed operations."""
        model = MixedNet()
        example_input = torch.randn(1, 3, 32, 32)
        
        tracer = FXTracer()
        traced = tracer.trace_model(model, example_input)
        
        assert traced is not None
        
        # Verify output matches
        model.eval()
        with torch.no_grad():
            expected = model(example_input)
            actual = traced(example_input)
            assert torch.allclose(expected, actual, rtol=1e-5)
    
    def test_print_graph(self):
        """Test graph printing functionality."""
        model = TinyMLP()
        example_input = torch.randn(1, 784)
        
        tracer = FXTracer()
        traced = tracer.trace_model(model, example_input)
        
        graph_str = tracer.print_graph(traced)
        assert isinstance(graph_str, str)
        assert len(graph_str) > 0
        assert "FX Graph:" in graph_str
    
    def test_validate_graph(self):
        """Test graph validation."""
        model = TinyMLP()
        example_input = torch.randn(1, 784)
        
        tracer = FXTracer()
        traced = tracer.trace_model(model, example_input)
        
        # Should not raise exception
        assert tracer.validate_graph(traced)

