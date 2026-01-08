"""
Tests for the lowering pass (FX graph to IR graph conversion)
"""

import pytest
import torch

from src.pytorch_to_c.frontend.fx_tracer import trace_model
from src.pytorch_to_c.lowering.lower import Lowering, lower_fx_graph
from test.test_models import TinyMLP, ResNetBlock, MixedNet


class TestLowering:
    """Test the lowering pass."""
    
    def test_lower_tiny_mlp(self):
        """Test lowering a simple MLP to IR."""
        model = TinyMLP()
        example_input = torch.randn(1, 784)
        
        # Trace
        fx_graph = trace_model(model, example_input)
        
        # Lower
        lowering = Lowering()
        ir_graph = lowering.lower_fx_graph(fx_graph)
        
        # Verify IR graph structure
        assert ir_graph is not None
        assert len(ir_graph.nodes) > 0
        assert len(ir_graph.parameters) > 0
        
        # Check for expected nodes
        node_types = [node.op_type for node in ir_graph.nodes]
        assert 'input' in node_types
        assert 'linear' in node_types
        assert 'relu' in node_types
    
    def test_double_linking(self):
        """Test that nodes are properly double-linked."""
        model = TinyMLP()
        example_input = torch.randn(1, 784)
        
        fx_graph = trace_model(model, example_input)
        ir_graph = lower_fx_graph(fx_graph)
        
        # Check double-linking
        for node in ir_graph.nodes:
            # Each input should list this node as a user
            for input_node in node.inputs:
                assert node in input_node.users, \
                    f"Node {node.name} is not in users of input {input_node.name}"
            
            # Each user should list this node as an input
            for user_node in node.users:
                assert node in user_node.inputs, \
                    f"Node {node.name} is not in inputs of user {user_node.name}"
    
    def test_parameters_extracted(self):
        """Test that parameters are properly extracted."""
        model = TinyMLP(input_size=10, hidden_size=5, output_size=2)
        example_input = torch.randn(1, 10)
        
        fx_graph = trace_model(model, example_input)
        ir_graph = lower_fx_graph(fx_graph)
        
        # Should have weights and biases for both linear layers
        assert len(ir_graph.parameters) >= 4  # 2 weights + 2 biases
        
        # Check parameter shapes
        for param_name, param_data in ir_graph.parameters.items():
            assert param_data is not None
            assert param_data.size > 0
    
    def test_layer_names_preserved(self):
        """Test that PyTorch layer names are preserved in IR."""
        model = TinyMLP()
        example_input = torch.randn(1, 784)
        
        fx_graph = trace_model(model, example_input)
        ir_graph = lower_fx_graph(fx_graph)
        
        # Check that node names exist and are reasonable
        node_names = [node.name for node in ir_graph.nodes]
        assert len(node_names) > 0
        assert all(isinstance(name, str) for name in node_names)
    
    def test_lower_resnet_block(self):
        """Test lowering a ResNet block with skip connection."""
        model = ResNetBlock()
        example_input = torch.randn(1, 64, 32, 32)
        
        fx_graph = trace_model(model, example_input)
        ir_graph = lower_fx_graph(fx_graph)
        
        # Verify conv, batchnorm, and add operations exist
        node_types = [node.op_type for node in ir_graph.nodes]
        assert 'conv2d' in node_types
        assert 'batchnorm' in node_types
        assert 'add' in node_types or 'relu' in node_types
    
    def test_ir_graph_validation(self):
        """Test that lowered IR graphs pass validation."""
        model = TinyMLP()
        example_input = torch.randn(1, 784)
        
        fx_graph = trace_model(model, example_input)
        ir_graph = lower_fx_graph(fx_graph)
        
        # Should not raise exception
        assert ir_graph.validate()
    
    def test_topological_order(self):
        """Test that nodes maintain topological order."""
        model = TinyMLP()
        example_input = torch.randn(1, 784)
        
        fx_graph = trace_model(model, example_input)
        ir_graph = lower_fx_graph(fx_graph)
        
        # Get topologically sorted nodes
        sorted_nodes = ir_graph.topological_sort()
        
        # Verify that each node appears before its users
        node_positions = {node: i for i, node in enumerate(sorted_nodes)}
        
        for node in sorted_nodes:
            for user in node.users:
                assert node_positions[node] < node_positions[user], \
                    f"Node {node.name} should appear before its user {user.name}"

