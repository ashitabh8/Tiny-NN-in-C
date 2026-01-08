"""
C code generator - converts IR graph to C code
"""

import os
import shutil
from typing import List, Dict, Any
from pathlib import Path
import numpy as np

from ..ir.graph import IRGraph
from ..ir.node import IRNode
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
        
        Args:
            output_dir: Directory to copy headers to
        """
        # Find the c_ops directory
        # Go up from this file: codegen/c_printer.py -> pytorch_to_c -> src -> project_root
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent
        c_ops_src = project_root / "src" / "c_ops" / "nn_ops_float.h"
        
        if c_ops_src.exists():
            c_ops_dst = os.path.join(output_dir, "nn_ops_float.h")
            shutil.copy2(c_ops_src, c_ops_dst)
        else:
            # Fallback: search for the file
            import sys
            for path in sys.path:
                candidate = Path(path) / "src" / "c_ops" / "nn_ops_float.h"
                if candidate.exists():
                    c_ops_dst = os.path.join(output_dir, "nn_ops_float.h")
                    shutil.copy2(candidate, c_ops_dst)
                    break
    
    def generate_weights_h(self) -> str:
        """
        Generate weights.h with serialized parameters.
        
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
        lines.append("")
        
        # Generate arrays for each parameter
        for param_name, param_data in self.ir_graph.parameters.items():
            # Flatten the array
            flat_data = param_data.flatten()
            
            # Generate C array
            c_name = self._sanitize_name(param_name)
            lines.append(f"// Shape: {param_data.shape}")
            lines.append(f"static const float {c_name}[{len(flat_data)}] = {{")
            
            # Write data in chunks of 8 values per line
            for i in range(0, len(flat_data), 8):
                chunk = flat_data[i:i+8]
                values_str = ", ".join([f"{v:.8f}f" for v in chunk])
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
        lines.append("")
        lines.append("#include <string.h>")
        lines.append("")
        
        # Generate the forward function
        lines.append("void model_forward(const float* input, float* output) {")
        lines.append("    // Intermediate buffers")
        
        # Declare buffers for each node (naive approach - Phase 1)
        buffer_sizes = self._calculate_buffer_sizes()
        for node in self.ir_graph.nodes:
            if node.op_type != 'input':
                size = buffer_sizes.get(node.name, 1024)  # Default size
                lines.append(f"    float {self._get_buffer_name(node)}[{size}];")
        
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
        Calculate buffer sizes for each node.
        
        Returns:
            Dictionary mapping node name to buffer size
        """
        sizes = {}
        
        for node in self.ir_graph.nodes:
            if node.op_type == 'input':
                continue
            
            # Estimate size based on operation type
            if node.op_type == 'linear':
                sizes[node.name] = node.metadata['out_features']
            elif node.op_type == 'conv2d':
                # This is a simplified estimate - would need actual shape inference
                sizes[node.name] = 1024  # Placeholder
            elif node.op_type in ['relu', 'softmax']:
                # Same size as input - need to infer from input node
                if node.inputs:
                    input_size = sizes.get(node.inputs[0].name, 1024)
                    sizes[node.name] = input_size
                else:
                    sizes[node.name] = 1024
            elif node.op_type == 'batchnorm':
                sizes[node.name] = 1024  # Placeholder
            else:
                sizes[node.name] = 1024  # Default
        
        return sizes
    
    def _generate_node_code(self, node: IRNode) -> List[str]:
        """
        Generate C code for a single IR node.
        
        Args:
            node: The IR node to generate code for
            
        Returns:
            List of C code lines
        """
        if node.op_type == 'input':
            return []  # Input is handled by function parameter
        
        elif node.op_type == 'conv2d':
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
        
        # For now, assume input size (this should come from shape inference)
        lines.append(f"int in_h = 32, in_w = 32;  // TODO: Infer from previous layer")
        
        # Determine padding mode
        pad_same = 1 if p_h > 0 or p_w > 0 else 0
        
        lines.append(
            f"conv2d_nhwc({input_buffer}, in_h, in_w, {in_channels}, "
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
        
        # Assume NHWC format
        lines.append(f"int h = 32, w = 32;  // TODO: Infer from previous layer")
        
        lines.append(
            f"batchnorm2d_nhwc({input_buffer}, h, w, {num_features}, "
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
        
        # Assume same size for both inputs
        size = 1024  # TODO: Get actual size
        
        lines.append(f"for (int i = 0; i < {size}; ++i) {{")
        lines.append(f"    {output_buffer}[i] = {input_buffer_a}[i] + {input_buffer_b}[i];")
        lines.append(f"}}")
        
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

