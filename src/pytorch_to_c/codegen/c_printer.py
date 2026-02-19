"""
C code generator - converts IR graph to C code

Supports both float32 and quantized (int8/int16) operations.
For quantized nodes, delegates code generation to the node itself.
"""

import os
import shutil
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

from ..ir.graph import IRGraph
from ..ir.node import IRNode
# Note: QuantIRNode and other quantized nodes implement generate_c_code()
# for custom code generation. See _generate_node_code() for details.
from .ops_map import OpMapping

try:
    from ..profiling.ops.profiling_utils import ProfilingWrapperNode
except ImportError:
    ProfilingWrapperNode = None  # type: ignore


class CPrinter:
    """
    Generates C code from an IR graph.

    Outputs (standard mode):
    - model.c: Main model implementation
    - model.h: Function declarations and interface
    - weights.h: Serialized parameter data

    Outputs (arduino_mode=True):
    - All of the above, plus a <sketch_name>.ino with setup()/loop()
    - Timing uses micros() instead of clock(), Serial.print instead of printf
    """

    def __init__(self, ir_graph: IRGraph, arduino_mode: bool = False):
        """
        Initialize the C code generator.

        Args:
            ir_graph: The IR graph to generate code from
            arduino_mode: When True, emit Arduino-compatible timing/print primitives
                          and generate a .ino sketch file
        """
        self.ir_graph = ir_graph
        self.buffer_counter = 0
        self.arduino_mode = arduino_mode

    def generate_all(self, output_dir: str, sketch_name: str = "model_sketch") -> None:
        """
        Generate all C files and copy necessary headers.

        In arduino_mode, also generates a <sketch_name>.ino with setup()/loop().

        Args:
            output_dir:  Directory to write generated files to
            sketch_name: Base name for the .ino sketch (arduino_mode only)
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

        if self.arduino_mode:
            ino = self.generate_arduino_sketch(sketch_name)
            with open(os.path.join(output_dir, f'{sketch_name}.ino'), 'w') as f:
                f.write(ino)

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

    def generate_arduino_sketch(self, sketch_name: str = "model_sketch") -> str:
        """
        Generate an Arduino .ino sketch that calls model_forward once from loop().

        The input buffer is filled with zeros by default — replace with real sensor
        data in loop() before calling model_forward().  Buffers are declared as
        global static arrays to avoid stack overflow on large models.

        Args:
            sketch_name: Base name used in the file header comment

        Returns:
            The .ino source as a string
        """
        import math

        lines = []
        lines.append(f"// Auto-generated Arduino sketch: {sketch_name}")
        lines.append("// DO NOT EDIT")
        lines.append("")

        # Annotate shapes
        if self.ir_graph.inputs:
            in_shape = self.ir_graph.inputs[0].output_shape
            if in_shape:
                lines.append(f"// Input shape  (NCHW): {list(in_shape)}")
        if self.ir_graph.outputs:
            out_shape = self.ir_graph.outputs[0].output_shape
            if out_shape:
                lines.append(f"// Output shape: {list(out_shape)}")
        lines.append("")

        lines.append('#include "model.h"')
        lines.append("")

        # Compute flat buffer sizes
        input_size = 0
        output_size = 0
        if self.ir_graph.inputs:
            shape = self.ir_graph.inputs[0].output_shape
            if shape:
                # NCHW (N, C, H, W) -> NHWC flat size = N*H*W*C
                if len(shape) == 4:
                    n, c, h, w = shape
                    input_size = n * h * w * c
                else:
                    input_size = math.prod(shape)
        if self.ir_graph.outputs:
            shape = self.ir_graph.outputs[0].output_shape
            if shape:
                output_size = math.prod(shape)

        lines.append(f"// Global buffers — avoids stack overflow for large activations")
        lines.append(f"static float input_buf[{input_size}];")
        lines.append(f"static float output_buf[{output_size}];")
        lines.append("static bool _inference_done = false;")
        lines.append("")

        lines.append("void setup() {")
        lines.append("    Serial.begin(115200);")
        lines.append("    while (!Serial) {}")
        lines.append("    // TODO: fill input_buf with real sensor data before inference")
        lines.append("    for (int i = 0; i < " + str(input_size) + "; ++i)")
        lines.append("        input_buf[i] = 0.0f;")
        lines.append("}")
        lines.append("")

        lines.append("void loop() {")
        lines.append("    if (_inference_done) return;")
        lines.append("    _inference_done = true;")
        lines.append("")
        lines.append("    model_forward(input_buf, output_buf);")
        lines.append("}")
        lines.append("")

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
    
    def _has_buffer(self, node: IRNode) -> bool:
        """True if this node produces an output buffer (not input or method_size)."""
        return node.op_type not in ('input', 'method_size')

    def _has_profiling_nodes(self) -> bool:
        """True if the graph contains any ProfilingWrapperNode (needs time.h, stdio.h)."""
        if ProfilingWrapperNode is None:
            return False
        return any(isinstance(n, ProfilingWrapperNode) for n in self.ir_graph.nodes)

    def _compute_buffer_last_use(self, order: List[IRNode]) -> Dict[str, IRNode]:
        """
        Compute for each buffer-producing node the last node (in execution order) that uses it.
        Used to close blocks so buffers go out of scope after their last use (reduces peak memory).
        """
        last_use: Dict[str, IRNode] = {}
        for node in order:
            for inp in node.inputs:
                last_use[inp.name] = node
        return last_use

    def generate_model_c(self) -> str:
        """
        Generate model.c with the main implementation.
        Uses liveness analysis: each activation buffer is declared in a block and the block
        is closed after the buffer's last use to reduce peak stack memory.
        """
        lines = []
        lines.append("// Auto-generated model implementation")
        lines.append("// DO NOT EDIT")
        lines.append("")

        if self.ir_graph.inputs:
            in_shape = self.ir_graph.inputs[0].output_shape
            if in_shape:
                lines.append(f"// Input shape  (NCHW): {list(in_shape)}")
        if self.ir_graph.outputs:
            out_shape = self.ir_graph.outputs[0].output_shape
            if out_shape:
                lines.append(f"// Output shape: {list(out_shape)}")
        lines.append("")

        lines.append("#include \"model.h\"")
        lines.append("#include \"weights.h\"")
        lines.append("#include \"nn_ops_float.h\"")
        
        if self._has_nodes_with_dtype('int8'):
            lines.append("#include \"nn_ops_int8.h\"")
        if self._has_nodes_with_dtype('int16'):
            lines.append("#include \"nn_ops_int16.h\"")
        
        lines.append("")
        lines.append("#include <string.h>")
        if self._has_profiling_nodes() and not self.arduino_mode:
            lines.append("#include <time.h>")
            lines.append("#include <stdio.h>")
        lines.append("")
        
        buffer_sizes = self._calculate_buffer_sizes()
        order = self.ir_graph.topological_sort()
        last_use = self._compute_buffer_last_use(order)
        output_node = self.ir_graph.outputs[0] if self.ir_graph.outputs else None
        
        base_indent = "    "

        lines.append("void model_forward(const float* input, float* output) {")
        if self._has_profiling_nodes():
            if self.arduino_mode:
                lines.append(base_indent + "unsigned long _t_start = micros();")
            else:
                lines.append(base_indent + "clock_t _t_start = clock();")

        # Stack of (node_name, indent for closing "}") for currently open blocks
        stack: List[tuple] = []
        
        for i, node in enumerate(order):
            prev_node = order[i - 1] if i > 0 else None
            
            # Close blocks for buffers whose last use was the previous node
            if prev_node is not None:
                to_close = {
                    p.name for p in order
                    if self._has_buffer(p) and last_use.get(p.name) == prev_node
                }
                while stack and stack[-1][0] in to_close:
                    name, ind = stack.pop()
                    lines.append(ind + "}")
                    to_close.discard(name)
            
            if node.op_type in ('input', 'method_size'):
                node_code = self._generate_node_code(node)
                if node_code:
                    indent = base_indent + "    " * len(stack)
                    for line in node_code:
                        lines.append(indent + line)
                continue
            
            # Node has a buffer: open block, declare, emit code, maybe copy output
            indent_brace = base_indent * (1 + len(stack))
            lines.append(indent_brace + "{")
            stack.append((node.name, indent_brace))
            indent_inner = indent_brace + base_indent
            
            lines.append(indent_inner + f"// {node.name} [{node.op_type}]")
            size = buffer_sizes[node.name]
            dtype = self._get_buffer_dtype(node)
            lines.append(indent_inner + f"{dtype} {self._get_buffer_name(node)}[{size}];")
            node_code = self._generate_node_code(node)
            if node_code:
                for line in node_code:
                    lines.append(indent_inner + line)
            if output_node is not None and node.name == output_node.name:
                lines.append(indent_inner + f"memcpy(output, {self._get_buffer_name(node)}, {size} * sizeof(float));")
        
        # Close any remaining open blocks
        while stack:
            _, ind = stack.pop()
            lines.append(ind + "}")
        
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
            if node.op_type == 'method_size':
                continue  # scalar int, no buffer

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
            
            # No output_shape: require operation-specific info; raise if missing
            if node.op_type == 'linear':
                if 'out_features' not in node.metadata:
                    raise ValueError(f"{node.name} (linear): missing metadata 'out_features'; need shape inference or metadata")
                sizes[node.name] = node.metadata['out_features']
            elif node.op_type == 'conv2d':
                raise ValueError(f"{node.name} (conv2d): missing output_shape; run with example_input for shape inference")
            elif node.op_type in ['relu', 'softmax', 'batchnorm']:
                if not node.inputs:
                    raise ValueError(f"{node.name} ({node.op_type}): no input node")
                input_size = self._node_buffer_size(sizes, node.inputs[0])
                if input_size is None:
                    raise ValueError(f"{node.name} ({node.op_type}): input shape unknown; run with example_input for shape inference")
                sizes[node.name] = input_size
            elif node.op_type == 'adaptive_avg_pool':
                if not node.inputs or not node.inputs[0].output_shape or len(node.inputs[0].output_shape) != 4:
                    raise ValueError(f"{node.name} (adaptive_avg_pool): need input with 4D shape [B,C,H,W]; run with example_input for shape inference")
                sizes[node.name] = node.inputs[0].output_shape[1]
            elif node.op_type in ('method_view', 'method_flatten'):
                if not node.inputs:
                    raise ValueError(f"{node.name} ({node.op_type}): no input node")
                input_size = self._node_buffer_size(sizes, node.inputs[0])
                if input_size is None:
                    raise ValueError(f"{node.name} ({node.op_type}): input shape unknown; run with example_input for shape inference")
                sizes[node.name] = input_size
            else:
                raise ValueError(f"{node.name}: unknown op_type '{node.op_type}' and no output_shape; run with example_input for shape inference")
        
        return sizes

    def _node_buffer_size(self, sizes: Dict[str, int], node: IRNode) -> Optional[int]:
        """Return buffer size for a node from sizes dict or output_shape. None if unknown."""
        if node.name in sizes:
            return sizes[node.name]
        if node.op_type == 'input' and node.output_shape is not None:
            import math
            shape = node.output_shape
            if len(shape) > 0 and shape[0] == 1:
                shape = shape[1:]
            return math.prod(shape) if shape else 1
        if node.output_shape is not None:
            import math
            shape = node.output_shape
            if len(shape) > 0 and shape[0] == 1:
                shape = shape[1:]
            return math.prod(shape) if shape else 1
        return None
    
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

        elif node.op_type == 'adaptive_avg_pool':
            return self._generate_adaptive_avg_pool(node)

        elif node.op_type in ('method_view', 'method_flatten'):
            return self._generate_flatten_or_view(node)

        elif node.op_type == 'method_size':
            return []  # scalar integer, no buffer

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

    def _generate_adaptive_avg_pool(self, node: IRNode) -> List[str]:
        """
        Generate code for AdaptiveAvgPool2d (e.g. (1,1) -> global average pool).
        Uses global_average_pool_2d; input is NHWC [H, W, C] from NCHW [B, C, H, W].
        """
        lines = []
        input_buffer = self._get_input_buffer(node, 0)
        output_buffer = self._get_buffer_name(node)
        h, w, c = 32, 32, 64  # defaults
        if node.inputs and node.inputs[0].output_shape and len(node.inputs[0].output_shape) == 4:
            _, c, h, w = node.inputs[0].output_shape
        lines.append(
            f"global_average_pool_2d({input_buffer}, {h}, {w}, {c}, {output_buffer});"
        )
        return lines

    def _generate_flatten_or_view(self, node: IRNode) -> List[str]:
        """Generate code for view/flatten: copy input buffer to output buffer (reshape)."""
        lines = []
        input_buffer = self._get_input_buffer(node, 0)
        output_buffer = self._get_buffer_name(node)
        buffer_sizes = self._calculate_buffer_sizes()
        size = buffer_sizes[node.name]
        lines.append(f"memcpy({output_buffer}, {input_buffer}, {size} * sizeof(float));")
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

