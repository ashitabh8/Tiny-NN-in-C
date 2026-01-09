"""
C code generator - converts IR graph to C code

Supports both float32 and quantized (int8/int16) operations.
For quantized nodes, delegates code generation to the node itself.
"""

import os
import shutil
from typing import List, Dict, Any
from pathlib import Path
import numpy as np

from ..ir.graph import IRGraph
from ..ir.node import IRNode
# Note: QuantIRNode and other quantized nodes implement generate_c_code()
# for custom code generation. See _generate_node_code() for details.
from .ops_map import OpMapping


class CPrinter:
    """
    Generates C code from an IR graph.
    
    Outputs:
    - model.c: Main model implementation
    - model.h: Function declarations and interface
    - weights.h: Serialized parameter data
    """
    
    def __init__(self, ir_graph: IRGraph):
        """
        Initialize the C code generator.
        
        Args:
            ir_graph: The IR graph to generate code from
        """
        self.ir_graph = ir_graph
        self.buffer_counter = 0
    
    def generate_all(self, output_dir: str) -> None:
        """
        Generate all C files and copy necessary headers.
        
        Args:
            output_dir: Directory to write generated files to
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate each file
        weights_h = self.generate_weights_h()
        model_h = self.generate_model_h()
        model_c = self.generate_model_c()
        
        # Write files
        with open(os.path.join(output_dir, 'weights.h'), 'w') as f:
            f.write(weights_h)
        
        with open(os.path.join(output_dir, 'model.h'), 'w') as f:
            f.write(model_h)
        
        with open(os.path.join(output_dir, 'model.c'), 'w') as f:
            f.write(model_c)
        
        # Copy C ops header to output directory for self-contained deployment
        self._copy_c_ops_headers(output_dir)
    
    def _copy_c_ops_headers(self, output_dir: str) -> None:
        """
        Copy C operation headers to the output directory.
        
        This makes the generated code self-contained and portable.
        Copies both float and quantized operation headers.
        
        Args:
            output_dir: Directory to copy headers to
        """
        # Find the c_ops directory
        # Go up from this file: codegen/c_printer.py -> pytorch_to_c -> src -> project_root
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent
        c_ops_dir = project_root / "src" / "c_ops"
        
        # List of headers to copy
        headers = ["nn_ops_float.h", "nn_ops_int8.h", "nn_ops_int16.h"]
        
        for header in headers:
            src = c_ops_dir / header
            if src.exists():
                dst = os.path.join(output_dir, header)
                shutil.copy2(src, dst)
    
    def generate_weights_h(self) -> str:
        """
        Generate weights.h with serialized parameters.
        
        Supports float32, int8, and int16 weight types.
        
        Returns:
            The C code as a string
        """
        lines = []
        lines.append("// Auto-generated weights file")
        lines.append("// DO NOT EDIT")
        lines.append("")
        lines.append("#ifndef WEIGHTS_H_")
        lines.append("#define WEIGHTS_H_")
        lines.append("")
        lines.append("#include <stddef.h>")
        lines.append("#include <stdint.h>")  # For int8_t, int16_t
        lines.append("")
        
        # Generate arrays for each parameter
        for param_name, param_data in self.ir_graph.parameters.items():
            # Flatten the array
            flat_data = param_data.flatten()
            
            # Determine C type from numpy dtype
            if param_data.dtype == np.int8:
                c_type = 'int8_t'
                format_func = lambda v: str(int(v))
            elif param_data.dtype == np.int16:
                c_type = 'int16_t'
                format_func = lambda v: str(int(v))
            elif param_data.dtype == np.int32:
                c_type = 'int32_t'
                format_func = lambda v: str(int(v))
            else:
                # float32 or default
                c_type = 'float'
                format_func = lambda v: f"{float(v):.8f}f"
            
            # Generate C array
            c_name = self._sanitize_name(param_name)
            lines.append(f"// Shape: {param_data.shape}, dtype: {param_data.dtype}")
            lines.append(f"static const {c_type} {c_name}[{len(flat_data)}] = {{")
            
            # Write data in chunks of 8 values per line
            for i in range(0, len(flat_data), 8):
                chunk = flat_data[i:i+8]
                values_str = ", ".join([format_func(v) for v in chunk])
                lines.append(f"    {values_str},")
            
            lines.append("};")
            lines.append("")
        
        lines.append("#endif // WEIGHTS_H_")
        return "\n".join(lines)
    
    def generate_model_h(self) -> str:
        """
        Generate model.h with function declarations.
        
        Returns:
            The C code as a string
        """
        lines = []
        lines.append("// Auto-generated model header")
        lines.append("// DO NOT EDIT")
        lines.append("")
        lines.append("#ifndef MODEL_H_")
        lines.append("#define MODEL_H_")
        lines.append("")
        lines.append("#include <stddef.h>")
        lines.append("")
        lines.append("// =============================================================================")
        lines.append("// IMPORTANT: Input Layout")
        lines.append("// =============================================================================")
        lines.append("// This model expects input in NHWC format (batch, height, width, channels).")
        lines.append("// PyTorch uses NCHW format. Convert before calling:")
        lines.append("//   PyTorch: input.permute(0, 2, 3, 1).numpy().flatten()")
        lines.append("// =============================================================================")
        lines.append("")
        
        # Get input and output shapes (simplified - assume single input/output)
        if self.ir_graph.inputs:
            input_node = self.ir_graph.inputs[0]
            lines.append(f"// Input: {input_node.name}")
        
        if self.ir_graph.outputs:
            output_node = self.ir_graph.outputs[0]
            lines.append(f"// Output: {output_node.name}")
        
        lines.append("")
        lines.append("// Main model inference function")
        lines.append("void model_forward(const float* input, float* output);")
        lines.append("")
        lines.append("#endif // MODEL_H_")
        return "\n".join(lines)
    
    def _has_nodes_with_dtype(self, dtype: str) -> bool:
        """
        Check if graph has any nodes with the specified dtype.
        
        Used to determine which C headers to include (e.g., nn_ops_int8.h).
        
        Args:
            dtype: The dtype to check for ('int8', 'int16', etc.)
        """
        return any(node.dtype == dtype for node in self.ir_graph.nodes)
    
    def _get_buffer_dtype(self, node: IRNode) -> str:
        """
        Get C data type for a node's buffer.
        
        All IRNodes now have get_c_dtype() which maps dtype -> C type.
        """
        return node.get_c_dtype()
    
    def generate_model_c(self) -> str:
        """
        Generate model.c with the main implementation.
        
        Returns:
            The C code as a string
        """
        lines = []
        lines.append("// Auto-generated model implementation")
        lines.append("// DO NOT EDIT")
        lines.append("")
        lines.append("#include \"model.h\"")
        lines.append("#include \"weights.h\"")
        lines.append("#include \"nn_ops_float.h\"")
        
        # Include quantized ops headers if needed
        if self._has_nodes_with_dtype('int8'):
            lines.append("#include \"nn_ops_int8.h\"")
        if self._has_nodes_with_dtype('int16'):
            lines.append("#include \"nn_ops_int16.h\"")
        
        lines.append("")
        lines.append("#include <string.h>")
        lines.append("")
        
        # Generate the forward function
        lines.append("void model_forward(const float* input, float* output) {")
        lines.append("    // Intermediate buffers")
        
        # Declare buffers for each node with correct dtype
        buffer_sizes = self._calculate_buffer_sizes()
        for node in self.ir_graph.nodes:
            if node.op_type != 'input':
                size = buffer_sizes.get(node.name, 1024)  # Default size
                dtype = self._get_buffer_dtype(node)
                lines.append(f"    {dtype} {self._get_buffer_name(node)}[{size}];")
        
        lines.append("")
        lines.append("    // Forward pass")
        
        # Generate code for each node
        for node in self.ir_graph.nodes:
            node_code = self._generate_node_code(node)
            if node_code:
                lines.append("")
                lines.append(f"    // {node.name} [{node.op_type}]")
                lines.extend([f"    {line}" for line in node_code])
        
        lines.append("")
        
        # Copy final output
        if self.ir_graph.outputs:
            output_node = self.ir_graph.outputs[0]
            output_buffer = self._get_buffer_name(output_node)
            output_size = buffer_sizes.get(output_node.name, 1024)
            lines.append(f"    // Copy output")
            lines.append(f"    memcpy(output, {output_buffer}, {output_size} * sizeof(float));")
        
        lines.append("}")
        lines.append("")
        
        return "\n".join(lines)
    
    def _calculate_buffer_sizes(self) -> Dict[str, int]:
        """
        Calculate buffer sizes for each node using inferred shapes.
        
        Returns:
            Dictionary mapping node name to buffer size (total number of elements)
        """
        sizes = {}
        
        for node in self.ir_graph.nodes:
            if node.op_type == 'input':
                continue
            
            # First priority: use inferred shape if available
            if node.output_shape is not None:
                # Calculate total number of elements from shape
                import math
                # Remove batch dimension (first dimension) if present
                shape = node.output_shape
                if len(shape) > 0 and shape[0] == 1:
                    shape = shape[1:]  # Remove batch dimension
                
                if len(shape) > 0:
                    size = math.prod(shape)
                    sizes[node.name] = size
                else:
                    sizes[node.name] = 1  # Scalar
                
                continue
            
            # Fallback: Estimate size based on operation type and metadata
            if node.op_type == 'linear':
                sizes[node.name] = node.metadata.get('out_features', 1024)
            elif node.op_type == 'conv2d':
                # Estimate from metadata (this is less accurate than shape inference)
                sizes[node.name] = 1024  # Fallback
            elif node.op_type in ['relu', 'softmax', 'batchnorm']:
                # Same size as input
                if node.inputs:
                    input_size = sizes.get(node.inputs[0].name, 1024)
                    sizes[node.name] = input_size
                else:
                    sizes[node.name] = 1024
            else:
                sizes[node.name] = 1024  # Default fallback
        
        return sizes
    
    def _generate_node_code(self, node: IRNode) -> List[str]:
        """
        Generate C code for a single IR node.
        
        Strategy:
        1. If node has generate_c_code() method, use it (custom codegen)
        2. Otherwise, fall back to built-in op_type handlers
        
        This allows any node to provide custom code generation by
        implementing generate_c_code(self, c_printer) -> List[str].
        
        Args:
            node: The IR node to generate code for
            
        Returns:
            List of C code lines
        """
        if node.op_type == 'input':
            return []  # Input is handled by function parameter
        
        # Check if node provides its own code generation
        # (QuantIRNode subclasses, QuantizeNode, DequantizeNode, etc.)
        if hasattr(node, 'generate_c_code'):
            return node.generate_c_code(self)
        
        # Built-in float operations (nodes without custom generate_c_code)
        if node.op_type == 'conv2d':
            return self._generate_conv2d(node)
        
        elif node.op_type == 'linear':
            return self._generate_linear(node)
        
        elif node.op_type == 'relu':
            return self._generate_relu(node)
        
        elif node.op_type == 'batchnorm':
            return self._generate_batchnorm(node)
        
        elif node.op_type == 'softmax':
            return self._generate_softmax(node)
        
        elif node.op_type == 'add':
            return self._generate_add(node)
        
        elif node.op_type == 'method_mean':
            return self._generate_mean(node)
        
        else:
            return [f"// Unsupported operation: {node.op_type}"]
    
    def _generate_conv2d(self, node: IRNode) -> List[str]:
        """Generate code for Conv2d operation."""
        lines = []
        
        input_buffer = self._get_input_buffer(node, 0)
        output_buffer = self._get_buffer_name(node)
        weight_name = self._sanitize_name(node.metadata['weight_name'])
        bias_name = self._sanitize_name(node.metadata['bias_name']) if node.metadata.get('bias_name') else 'NULL'
        
        # Extract parameters
        kernel_size = node.metadata['kernel_size']
        stride = node.metadata['stride']
        padding = node.metadata['padding']
        in_channels = node.metadata['in_channels']
        out_channels = node.metadata['out_channels']
        
        # Convert to scalars if tuples
        k_h, k_w = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        s_h, s_w = stride if isinstance(stride, (tuple, list)) else (stride, stride)
        p_h, p_w = padding if isinstance(padding, (tuple, list)) else (padding, padding)
        
        # Get input shape from the input node
        in_h, in_w = 32, 32  # Default
        if node.inputs and node.inputs[0].output_shape:
            input_shape = node.inputs[0].output_shape
            # Shape is [B, C, H, W] in NCHW or [B, H, W, C] in NHWC
            if len(input_shape) == 4:
                # Assume NCHW from PyTorch
                in_h, in_w = input_shape[2], input_shape[3]
        
        # Determine padding mode
        pad_same = 1 if p_h > 0 or p_w > 0 else 0
        
        lines.append(
            f"conv2d_nhwc({input_buffer}, {in_h}, {in_w}, {in_channels}, "
            f"{weight_name}, {k_h}, {k_w}, {out_channels}, "
            f"{bias_name}, {s_h}, {s_w}, {pad_same}, {output_buffer});"
        )
        
        return lines
    
    def _generate_linear(self, node: IRNode) -> List[str]:
        """Generate code for Linear operation."""
        lines = []
        
        input_buffer = self._get_input_buffer(node, 0)
        output_buffer = self._get_buffer_name(node)
        weight_name = self._sanitize_name(node.metadata['weight_name'])
        bias_name = self._sanitize_name(node.metadata['bias_name']) if node.metadata.get('bias_name') else 'NULL'
        
        in_features = node.metadata['in_features']
        out_features = node.metadata['out_features']
        
        lines.append(
            f"dense({input_buffer}, {in_features}, "
            f"{weight_name}, {bias_name}, {out_features}, {output_buffer});"
        )
        
        return lines
    
    def _generate_relu(self, node: IRNode) -> List[str]:
        """Generate code for ReLU operation."""
        lines = []
        
        input_buffer = self._get_input_buffer(node, 0)
        output_buffer = self._get_buffer_name(node)
        
        # Get actual size from buffer size calculation
        buffer_sizes = self._calculate_buffer_sizes()
        size = buffer_sizes.get(node.name, 1024)  # Fall back to 1024 if not found
        
        # ReLU can be in-place, but we'll copy for clarity in Phase 1
        lines.append(f"memcpy({output_buffer}, {input_buffer}, {size} * sizeof(float));")
        lines.append(f"relu({output_buffer}, {size});")
        
        return lines
    
    def _generate_batchnorm(self, node: IRNode) -> List[str]:
        """Generate code for BatchNorm operation."""
        lines = []
        
        input_buffer = self._get_input_buffer(node, 0)
        output_buffer = self._get_buffer_name(node)
        gamma_name = self._sanitize_name(node.metadata['gamma_name'])
        beta_name = self._sanitize_name(node.metadata['beta_name'])
        mean_name = self._sanitize_name(node.metadata['mean_name'])
        var_name = self._sanitize_name(node.metadata['var_name'])
        eps = node.metadata['eps']
        num_features = node.metadata['num_features']
        
        # Get spatial dimensions from input shape
        h, w = 32, 32  # Default
        if node.inputs and node.inputs[0].output_shape:
            input_shape = node.inputs[0].output_shape
            # Shape is [B, C, H, W] in NCHW
            if len(input_shape) == 4:
                h, w = input_shape[2], input_shape[3]
        
        lines.append(
            f"batchnorm2d_nhwc({input_buffer}, {h}, {w}, {num_features}, "
            f"{gamma_name}, {beta_name}, {mean_name}, {var_name}, "
            f"{eps}f, {output_buffer});"
        )
        
        return lines
    
    def _generate_softmax(self, node: IRNode) -> List[str]:
        """Generate code for Softmax operation."""
        lines = []
        
        input_buffer = self._get_input_buffer(node, 0)
        output_buffer = self._get_buffer_name(node)
        
        # Get actual size from buffer size calculation
        buffer_sizes = self._calculate_buffer_sizes()
        size = buffer_sizes.get(node.name, 10)  # Fall back to 10 if not found
        
        lines.append(f"memcpy({output_buffer}, {input_buffer}, {size} * sizeof(float));")
        lines.append(f"softmax({output_buffer}, {size});")
        
        return lines
    
    def _generate_add(self, node: IRNode) -> List[str]:
        """Generate code for element-wise addition."""
        lines = []
        
        input_buffer_a = self._get_input_buffer(node, 0)
        input_buffer_b = self._get_input_buffer(node, 1)
        output_buffer = self._get_buffer_name(node)
        
        # Get actual size from input shape
        buffer_sizes = self._calculate_buffer_sizes()
        size = buffer_sizes.get(node.name, 1024)
        
        lines.append(f"for (int i = 0; i < {size}; ++i) {{")
        lines.append(f"    {output_buffer}[i] = {input_buffer_a}[i] + {input_buffer_b}[i];")
        lines.append(f"}}")
        
        return lines
    
    def _generate_mean(self, node: IRNode) -> List[str]:
        """
        Generate code for mean reduction over specified dimensions.
        
        Handles tensor.mean(dim=[2, 3]) which reduces [B, C, H, W] -> [B, C]
        In NHWC format this is mean over spatial dimensions H and W.
        """
        lines = []
        
        input_buffer = self._get_input_buffer(node, 0)
        output_buffer = self._get_buffer_name(node)
        
        # Get the dimensions to reduce over from metadata
        # kwargs contains {'dim': [2, 3]} for tensor.mean(dim=[2, 3])
        kwargs = node.metadata.get('kwargs', {})
        args = node.metadata.get('args', ())
        
        # dim can be in kwargs or as first positional arg
        dim = kwargs.get('dim', None)
        if dim is None and args:
            dim = args[0] if isinstance(args[0], (list, tuple, int)) else [2, 3]
        if dim is None:
            dim = [2, 3]  # Default to spatial dims for NCHW
        
        # Get input shape to determine spatial dimensions
        input_node = node.inputs[0] if node.inputs else None
        if input_node and input_node.output_shape:
            input_shape = input_node.output_shape
            # Remove batch dimension if present
            if len(input_shape) == 4 and input_shape[0] == 1:
                # NCHW format in PyTorch: [1, C, H, W]
                # After shape inference, this is [1, C, H, W]
                _, c, h, w = input_shape
                
                # Check if reducing over spatial dims (H, W = dims 2, 3)
                if set(dim) == {2, 3}:
                    # This is global average pooling over H, W
                    # Our C code uses NHWC, so input is [H, W, C]
                    lines.append(f"// Mean over spatial dimensions (global average pool)")
                    lines.append(f"mean_hwc({input_buffer}, {h}, {w}, {c}, {output_buffer});")
                else:
                    lines.append(f"// TODO: Mean over dims {dim} not yet implemented")
            else:
                # Fallback for other shapes
                lines.append(f"// TODO: Mean for shape {input_shape} over dims {dim}")
        else:
            # No shape info - use generic fallback
            lines.append(f"// TODO: Mean operation - shape inference needed")
        
        return lines
    
    def _get_buffer_name(self, node: IRNode) -> str:
        """Get the C variable name for a node's output buffer."""
        if node.op_type == 'input':
            return 'input'
        return f"buf_{self._sanitize_name(node.name)}"
    
    def _get_input_buffer(self, node: IRNode, input_idx: int) -> str:
        """Get the buffer name for a node's input."""
        if input_idx >= len(node.inputs):
            raise ValueError(f"Node {node.name} doesn't have input {input_idx}")
        
        input_node = node.inputs[input_idx]
        return self._get_buffer_name(input_node)
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize a name to be a valid C identifier."""
        # Replace invalid characters with underscore
        sanitized = name.replace('.', '_').replace('-', '_').replace(' ', '_')
        # Ensure it starts with a letter or underscore
        if sanitized and sanitized[0].isdigit():
            sanitized = '_' + sanitized
        return sanitized


def generate_c_code(ir_graph: IRGraph, output_dir: str) -> None:
    """
    Convenience function to generate C code from an IR graph.
    
    Args:
        ir_graph: The IR graph
        output_dir: Directory to write generated files to
    """
    printer = CPrinter(ir_graph)
    printer.generate_all(output_dir)

