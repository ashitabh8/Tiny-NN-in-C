"""
Lowering pass: Convert torch.fx graph to custom IR
"""

import torch
import torch.fx as fx
import numpy as np
from typing import Dict, Any, Optional, Tuple
import math

from ..ir.node import IRNode
from ..ir.graph import IRGraph


class Lowering:
    """
    Converts a torch.fx GraphModule to our custom IR.
    
    Key responsibilities:
    - Map FX nodes to IR nodes
    - Extract weights and biases into parameter storage
    - Build double-linking between nodes
    - Preserve layer names for future quantization rules
    """
    
    def __init__(self):
        """Initialize the lowering pass."""
        self.ir_graph = IRGraph()
        self.node_map: Dict[fx.Node, IRNode] = {}  # FX node -> IR node
        self.module_dict: Dict[str, torch.nn.Module] = {}
        self.shape_map: Dict[fx.Node, Tuple[int, ...]] = {}  # FX node -> output shape
    
    def lower_fx_graph(
        self,
        fx_graph_module: fx.GraphModule,
        example_input: Optional[torch.Tensor] = None
    ) -> IRGraph:
        """
        Convert an FX graph to IR graph with shape inference.
        
        Args:
            fx_graph_module: The traced FX GraphModule
            example_input: Example input for shape inference (optional)
            
        Returns:
            IRGraph: The lowered IR graph
        """
        # Store module references for get_attr operations
        self.module_dict = dict(fx_graph_module.named_modules())
        
        # Infer shapes if example input is provided
        if example_input is not None:
            self.shape_map = self._infer_shapes(fx_graph_module, example_input)
        
        # torch.fx guarantees topological order, so we can iterate directly
        for fx_node in fx_graph_module.graph.nodes:
            ir_node = self._lower_node(fx_node, fx_graph_module)
            
            # Add shape information if available
            if ir_node and fx_node in self.shape_map:
                ir_node.output_shape = self.shape_map[fx_node]
        
        # Validate the IR graph
        self.ir_graph.validate()
        
        return self.ir_graph
    
    def _lower_node(
        self,
        fx_node: fx.Node,
        fx_graph_module: fx.GraphModule
    ) -> Optional[IRNode]:
        """
        Lower a single FX node to an IR node.
        
        Args:
            fx_node: The FX node to lower
            fx_graph_module: The containing graph module
            
        Returns:
            The created IR node, or None for placeholder/get_attr nodes
        """
        if fx_node.op == 'placeholder':
            return self._lower_placeholder(fx_node)
        
        elif fx_node.op == 'get_attr':
            return self._lower_get_attr(fx_node, fx_graph_module)
        
        elif fx_node.op == 'call_module':
            return self._lower_call_module(fx_node, fx_graph_module)
        
        elif fx_node.op == 'call_function':
            return self._lower_call_function(fx_node)
        
        elif fx_node.op == 'call_method':
            return self._lower_call_method(fx_node)
        
        elif fx_node.op == 'output':
            return self._lower_output(fx_node)
        
        else:
            raise ValueError(f"Unsupported FX node operation: {fx_node.op}")
    
    def _lower_placeholder(self, fx_node: fx.Node) -> IRNode:
        """Lower a placeholder (input) node."""
        ir_node = IRNode(
            name=fx_node.name,
            op_type='input',
            dtype='float32'
        )
        self.ir_graph.add_node(ir_node)
        self.ir_graph.mark_input(ir_node)
        self.node_map[fx_node] = ir_node
        return ir_node
    
    def _lower_get_attr(
        self,
        fx_node: fx.Node,
        fx_graph_module: fx.GraphModule
    ) -> None:
        """
        Lower a get_attr node (parameter access).
        Store the parameter in the IR graph's parameter dict.
        """
        # Get the actual parameter tensor
        param = fx_graph_module.get_parameter(fx_node.target)
        
        # Convert to numpy and store
        param_np = param.detach().cpu().numpy()
        self.ir_graph.add_parameter(fx_node.name, param_np)
        
        # Don't create an IR node for get_attr - parameters are stored separately
        # But we need to track it for other nodes that reference it
        self.node_map[fx_node] = None
    
    def _lower_call_module(
        self,
        fx_node: fx.Node,
        fx_graph_module: fx.GraphModule
    ) -> IRNode:
        """Lower a call_module node (e.g., Conv2d, Linear, ReLU)."""
        # Get the actual module
        module = fx_graph_module.get_submodule(fx_node.target)
        module_type = type(module).__name__
        
        # Map module type to IR op type
        if isinstance(module, torch.nn.Conv2d):
            ir_node = self._lower_conv2d(fx_node, module)
        elif isinstance(module, torch.nn.Linear):
            ir_node = self._lower_linear(fx_node, module)
        elif isinstance(module, torch.nn.ReLU):
            ir_node = self._lower_relu(fx_node, module)
        elif isinstance(module, torch.nn.BatchNorm2d):
            ir_node = self._lower_batchnorm2d(fx_node, module)
        elif isinstance(module, torch.nn.Softmax):
            ir_node = self._lower_softmax(fx_node, module)
        else:
            raise ValueError(f"Unsupported module type: {module_type}")
        
        # Add input dependencies (double-linking)
        for arg in fx_node.args:
            if isinstance(arg, fx.Node) and arg in self.node_map:
                input_ir_node = self.node_map[arg]
                if input_ir_node is not None:  # Skip get_attr nodes
                    ir_node.add_input(input_ir_node)
        
        self.ir_graph.add_node(ir_node)
        self.node_map[fx_node] = ir_node
        return ir_node
    
    def _lower_conv2d(
        self,
        fx_node: fx.Node,
        module: torch.nn.Conv2d
    ) -> IRNode:
        """Lower a Conv2d module."""
        # Extract weight and bias
        weight = module.weight.detach().cpu().numpy()
        bias = module.bias.detach().cpu().numpy() if module.bias is not None else None
        
        # Convert from PyTorch format [out_c, in_c, k_h, k_w] to HWIO [k_h, k_w, in_c, out_c]
        weight_hwio = np.transpose(weight, (2, 3, 1, 0))
        
        # Store parameters
        weight_name = f"{fx_node.name}_weight"
        self.ir_graph.add_parameter(weight_name, weight_hwio)
        if bias is not None:
            bias_name = f"{fx_node.name}_bias"
            self.ir_graph.add_parameter(bias_name, bias)
        
        # Create IR node
        ir_node = IRNode(
            name=fx_node.name,
            op_type='conv2d',
            dtype='float32',
            metadata={
                'weight_name': weight_name,
                'bias_name': f"{fx_node.name}_bias" if bias is not None else None,
                'kernel_size': module.kernel_size,
                'stride': module.stride,
                'padding': module.padding,
                'in_channels': module.in_channels,
                'out_channels': module.out_channels,
            }
        )
        return ir_node
    
    def _lower_linear(
        self,
        fx_node: fx.Node,
        module: torch.nn.Linear
    ) -> IRNode:
        """Lower a Linear (fully connected) module."""
        # Extract weight and bias
        weight = module.weight.detach().cpu().numpy()
        bias = module.bias.detach().cpu().numpy() if module.bias is not None else None
        
        # PyTorch Linear weight is [out_features, in_features]
        # We need [in_features, out_features] for row-major access
        weight_transposed = weight.T
        
        # Store parameters
        weight_name = f"{fx_node.name}_weight"
        self.ir_graph.add_parameter(weight_name, weight_transposed)
        if bias is not None:
            bias_name = f"{fx_node.name}_bias"
            self.ir_graph.add_parameter(bias_name, bias)
        
        # Create IR node
        ir_node = IRNode(
            name=fx_node.name,
            op_type='linear',
            dtype='float32',
            metadata={
                'weight_name': weight_name,
                'bias_name': f"{fx_node.name}_bias" if bias is not None else None,
                'in_features': module.in_features,
                'out_features': module.out_features,
            }
        )
        return ir_node
    
    def _lower_relu(
        self,
        fx_node: fx.Node,
        module: torch.nn.ReLU
    ) -> IRNode:
        """Lower a ReLU activation."""
        ir_node = IRNode(
            name=fx_node.name,
            op_type='relu',
            dtype='float32',
            metadata={}
        )
        return ir_node
    
    def _lower_batchnorm2d(
        self,
        fx_node: fx.Node,
        module: torch.nn.BatchNorm2d
    ) -> IRNode:
        """Lower a BatchNorm2d module."""
        # Extract parameters
        gamma = module.weight.detach().cpu().numpy() if module.weight is not None else np.ones(module.num_features)
        beta = module.bias.detach().cpu().numpy() if module.bias is not None else np.zeros(module.num_features)
        mean = module.running_mean.detach().cpu().numpy()
        var = module.running_var.detach().cpu().numpy()
        
        # Store parameters
        gamma_name = f"{fx_node.name}_gamma"
        beta_name = f"{fx_node.name}_beta"
        mean_name = f"{fx_node.name}_mean"
        var_name = f"{fx_node.name}_var"
        
        self.ir_graph.add_parameter(gamma_name, gamma)
        self.ir_graph.add_parameter(beta_name, beta)
        self.ir_graph.add_parameter(mean_name, mean)
        self.ir_graph.add_parameter(var_name, var)
        
        # Create IR node
        ir_node = IRNode(
            name=fx_node.name,
            op_type='batchnorm',
            dtype='float32',
            metadata={
                'gamma_name': gamma_name,
                'beta_name': beta_name,
                'mean_name': mean_name,
                'var_name': var_name,
                'eps': module.eps,
                'num_features': module.num_features,
            }
        )
        return ir_node
    
    def _lower_softmax(
        self,
        fx_node: fx.Node,
        module: torch.nn.Softmax
    ) -> IRNode:
        """Lower a Softmax activation."""
        ir_node = IRNode(
            name=fx_node.name,
            op_type='softmax',
            dtype='float32',
            metadata={
                'dim': module.dim,
            }
        )
        return ir_node
    
    def _lower_call_function(self, fx_node: fx.Node) -> IRNode:
        """Lower a functional call (e.g., torch.add, torch.mul)."""
        func_name = fx_node.target.__name__ if hasattr(fx_node.target, '__name__') else str(fx_node.target)
        
        # Map common functional operations
        op_map = {
            'add': 'add',
            'mul': 'mul',
            'relu': 'relu',
        }
        
        op_type = op_map.get(func_name, func_name)
        
        ir_node = IRNode(
            name=fx_node.name,
            op_type=op_type,
            dtype='float32',
            metadata={'func_name': func_name}
        )
        
        # Add input dependencies
        for arg in fx_node.args:
            if isinstance(arg, fx.Node) and arg in self.node_map:
                input_ir_node = self.node_map[arg]
                if input_ir_node is not None:
                    ir_node.add_input(input_ir_node)
        
        self.ir_graph.add_node(ir_node)
        self.node_map[fx_node] = ir_node
        return ir_node
    
    def _lower_call_method(self, fx_node: fx.Node) -> IRNode:
        """Lower a method call (e.g., tensor.view, tensor.flatten, tensor.mean)."""
        method_name = fx_node.target
        
        # Capture both positional args (after self) and keyword args
        ir_node = IRNode(
            name=fx_node.name,
            op_type=f'method_{method_name}',
            dtype='float32',
            metadata={
                'method_name': method_name,
                'args': fx_node.args[1:],  # Skip 'self' arg
                'kwargs': dict(fx_node.kwargs) if fx_node.kwargs else {}
            }
        )
        
        # Add input dependencies (first arg is self)
        if fx_node.args:
            arg = fx_node.args[0]
            if isinstance(arg, fx.Node) and arg in self.node_map:
                input_ir_node = self.node_map[arg]
                if input_ir_node is not None:
                    ir_node.add_input(input_ir_node)
        
        self.ir_graph.add_node(ir_node)
        self.node_map[fx_node] = ir_node
        return ir_node
    
    def _lower_output(self, fx_node: fx.Node) -> None:
        """Mark output nodes in the IR graph."""
        # The output node's args contain the actual output nodes
        for arg in fx_node.args:
            if isinstance(arg, fx.Node) and arg in self.node_map:
                output_ir_node = self.node_map[arg]
                if output_ir_node is not None:
                    self.ir_graph.mark_output(output_ir_node)
            elif isinstance(arg, (list, tuple)):
                # Multiple outputs
                for item in arg:
                    if isinstance(item, fx.Node) and item in self.node_map:
                        output_ir_node = self.node_map[item]
                        if output_ir_node is not None:
                            self.ir_graph.mark_output(output_ir_node)
    
    def _infer_shapes(
        self,
        fx_graph_module: fx.GraphModule,
        example_input: torch.Tensor
    ) -> Dict[fx.Node, Tuple[int, ...]]:
        """
        Infer output shapes for all nodes by running shape propagation.
        
        Args:
            fx_graph_module: The FX graph module
            example_input: Example input tensor
            
        Returns:
            Dictionary mapping FX nodes to their output shapes
        """
        shape_map = {}
        
        try:
            # Use torch.fx's ShapeProp for automatic shape propagation
            from torch.fx.passes.shape_prop import ShapeProp
            
            ShapeProp(fx_graph_module).propagate(example_input)
            
            # Extract shapes from node metadata
            for node in fx_graph_module.graph.nodes:
                if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
                    # ShapeProp stores shape in tensor_meta
                    tensor_meta = node.meta['tensor_meta']
                    shape_map[node] = tuple(tensor_meta.shape)
                elif hasattr(node, 'meta') and 'val' in node.meta:
                    # Some nodes store actual tensor in 'val'
                    val = node.meta['val']
                    if hasattr(val, 'shape'):
                        shape_map[node] = tuple(val.shape)
        
        except Exception as e:
            # Fallback: manual shape inference if ShapeProp fails
            print(f"Warning: ShapeProp failed ({e}), using manual shape inference")
            shape_map = self._manual_shape_inference(fx_graph_module, example_input)
        
        return shape_map
    
    def _manual_shape_inference(
        self,
        fx_graph_module: fx.GraphModule,
        example_input: torch.Tensor
    ) -> Dict[fx.Node, Tuple[int, ...]]:
        """
        Manual shape inference by running actual forward pass.
        
        Args:
            fx_graph_module: The FX graph module
            example_input: Example input tensor
            
        Returns:
            Dictionary mapping FX nodes to their output shapes
        """
        shape_map = {}
        
        # Run forward pass and track intermediate outputs
        with torch.no_grad():
            # Store intermediate values
            def forward_hook(module, input, output):
                # This will be called for each module
                pass
            
            # Execute the graph node by node
            env = {}
            for node in fx_graph_module.graph.nodes:
                if node.op == 'placeholder':
                    env[node] = example_input
                    shape_map[node] = tuple(example_input.shape)
                
                elif node.op == 'get_attr':
                    # Parameter node - get its shape
                    param = fx_graph_module.get_parameter(node.target)
                    shape_map[node] = tuple(param.shape)
                    env[node] = param
                
                elif node.op == 'call_module':
                    # Get the module and run it
                    module = fx_graph_module.get_submodule(node.target)
                    input_val = env[node.args[0]] if node.args else None
                    if input_val is not None:
                        output = module(input_val)
                        env[node] = output
                        shape_map[node] = tuple(output.shape)
                
                elif node.op == 'call_function':
                    # Call a function
                    args = [env[arg] if isinstance(arg, fx.Node) else arg for arg in node.args]
                    output = node.target(*args)
                    env[node] = output
                    if hasattr(output, 'shape'):
                        shape_map[node] = tuple(output.shape)
                
                elif node.op == 'call_method':
                    # Call a method on a tensor
                    self_arg = env[node.args[0]]
                    args = [env[arg] if isinstance(arg, fx.Node) else arg for arg in node.args[1:]]
                    output = getattr(self_arg, node.target)(*args)
                    env[node] = output
                    if hasattr(output, 'shape'):
                        shape_map[node] = tuple(output.shape)
                    elif isinstance(output, int):
                        # size() returns int, not a tensor
                        shape_map[node] = (1,)  # Scalar
        
        return shape_map


def lower_fx_graph(fx_graph_module: fx.GraphModule, example_input: Optional[torch.Tensor] = None) -> IRGraph:
    """
    Convenience function to lower an FX graph to IR.
    
    Args:
        fx_graph_module: The traced FX GraphModule
        example_input: Example input for shape inference (optional)
        
    Returns:
        The lowered IR graph
    """
    lowering = Lowering()
    return lowering.lower_fx_graph(fx_graph_module, example_input)

