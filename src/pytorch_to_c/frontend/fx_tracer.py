"""
Frontend module for PyTorch graph capture using torch.fx
"""

import torch
import torch.fx as fx
from typing import Any, Optional


class FXTracer:
    """
    Wrapper around torch.fx for tracing PyTorch models.
    
    Handles common patterns like skip connections and converts the model
    into a symbolic FX graph for further processing.
    """
    
    def __init__(self):
        """Initialize the FX tracer."""
        pass
    
    def trace_model(
        self,
        model: torch.nn.Module,
        example_input: torch.Tensor
    ) -> fx.GraphModule:
        """
        Trace a PyTorch model to extract its computational graph.
        
        Args:
            model: The PyTorch nn.Module to trace
            example_input: An example input tensor to trace through the model
            
        Returns:
            fx.GraphModule: The traced graph module
            
        Raises:
            RuntimeError: If tracing fails
        """
        try:
            # Set model to eval mode to avoid issues with dropout, batchnorm, etc.
            model.eval()
            
            # Trace the model using torch.fx
            traced = fx.symbolic_trace(model)
            
            # Validate the trace by running it
            with torch.no_grad():
                original_output = model(example_input)
                traced_output = traced(example_input)
                
                # Verify outputs match
                if not torch.allclose(original_output, traced_output, rtol=1e-5, atol=1e-5):
                    raise RuntimeError(
                        "Traced model output doesn't match original model output. "
                        "The model may contain dynamic control flow that torch.fx cannot trace."
                    )
            
            return traced
        
        except Exception as e:
            raise RuntimeError(f"Failed to trace model: {e}")
    
    def print_graph(self, graph_module: fx.GraphModule) -> str:
        """
        Generate a human-readable representation of the FX graph.
        
        Args:
            graph_module: The traced graph module
            
        Returns:
            String representation of the graph
        """
        lines = ["FX Graph:"]
        lines.append(str(graph_module.graph))
        lines.append("\nNode Details:")
        
        for node in graph_module.graph.nodes:
            lines.append(f"\nNode: {node.name}")
            lines.append(f"  Op: {node.op}")
            lines.append(f"  Target: {node.target}")
            lines.append(f"  Args: {node.args}")
            lines.append(f"  Kwargs: {node.kwargs}")
            
            # Print users (nodes that use this node's output)
            if node.users:
                users = [u.name for u in node.users]
                lines.append(f"  Users: {users}")
        
        return "\n".join(lines)
    
    def validate_graph(self, graph_module: fx.GraphModule) -> bool:
        """
        Validate that the traced graph is suitable for compilation.
        
        Args:
            graph_module: The traced graph module
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If the graph contains unsupported patterns
        """
        supported_ops = {
            'placeholder',  # Input
            'get_attr',     # Weight/parameter access
            'call_function', # Functional operations
            'call_method',  # Method calls
            'call_module',  # Module calls
            'output',       # Output
        }
        
        for node in graph_module.graph.nodes:
            if node.op not in supported_ops:
                raise ValueError(f"Unsupported operation type: {node.op} in node {node.name}")
        
        return True


def trace_model(model: torch.nn.Module, example_input: torch.Tensor) -> fx.GraphModule:
    """
    Convenience function to trace a PyTorch model.
    
    Args:
        model: The PyTorch nn.Module to trace
        example_input: An example input tensor
        
    Returns:
        The traced FX GraphModule
    """
    tracer = FXTracer()
    return tracer.trace_model(model, example_input)

