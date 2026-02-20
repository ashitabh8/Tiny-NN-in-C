"""
C code generator - converts IR graph to C code

Supports both float32 and quantized (int8/int16) operations.
For quantized nodes, delegates code generation to the node itself.
"""

import os
import shutil
from typing import List, Dict, Any, Optional, Tuple
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

    def generate_all(self, output_dir: str, sketch_name: str = None) -> None:
        """
        Generate all C files and copy necessary headers.

        In arduino_mode, also generates a <sketch_name>.ino with setup()/loop().
        The sketch name defaults to the basename of output_dir so the .ino
        filename always matches its containing folder (Arduino requirement).

        Args:
            output_dir:  Directory to write generated files to
            sketch_name: Base name for the .ino sketch (arduino_mode only).
                         Defaults to os.path.basename(output_dir).
        """
        os.makedirs(output_dir, exist_ok=True)
        if sketch_name is None:
            sketch_name = os.path.basename(os.path.abspath(output_dir))

        # Generate each file
        weights_h = self.generate_weights_h()
        model_h = self.generate_model_h()
        model_c = self.generate_model_c()

        # Write files
        with open(os.path.join(output_dir, 'weights.h'), 'w') as f:
            f.write(weights_h)

        with open(os.path.join(output_dir, 'model.h'), 'w') as f:
            f.write(model_h)

        model_filename = 'model.cpp' if self.arduino_mode else 'model.c'
        with open(os.path.join(output_dir, model_filename), 'w') as f:
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

        Includes random input generation, profiling checkpoint output, per-class
        scores, and argmax prediction.  Sizes are derived from the IR graph shapes.

        Args:
            sketch_name: Base name used in file header and compile instructions

        Returns:
            The .ino source as a string
        """
        import math

        # Compute flat buffer sizes from IR graph shapes
        input_size = 0
        output_size = 0
        in_shape_str = "unknown"
        out_shape_str = "unknown"

        if self.ir_graph.inputs:
            shape = self.ir_graph.inputs[0].output_shape
            if shape:
                in_shape_str = str(list(shape))
                if len(shape) == 4:
                    n, c, h, w = shape
                    input_size = n * h * w * c
                    nhwc_comment = f"{n} * {h} * {w} * {c}  (NHWC)"
                else:
                    input_size = math.prod(shape)
                    nhwc_comment = f"flat {input_size}"
            else:
                nhwc_comment = "unknown"

        if self.ir_graph.outputs:
            shape = self.ir_graph.outputs[0].output_shape
            if shape:
                out_shape_str = str(list(shape))
                output_size = math.prod(shape)

        lines = []
        lines.append("/*")
        lines.append(f" * Auto-generated Arduino runner: {sketch_name}")
        lines.append(" *")
        lines.append(f" * Copy this file and all files from the generated code directory")
        lines.append(f" * into a single Arduino sketch folder named \"{sketch_name}\".")
        lines.append(f" * (Folder name must match the .ino filename.)")
        lines.append(" *")
        lines.append(" * Board: Arduino Giga R1  (FQBN: arduino:mbed_giga:giga)")
        lines.append(" *")
        lines.append(" * Compile check (no upload):")
        lines.append(f" *   arduino-cli compile --fqbn arduino:mbed_giga:giga {sketch_name}/")
        lines.append(" *")
        lines.append(f" * Model I/O:")
        lines.append(f" *   Input  (NCHW): {in_shape_str}  ->  NHWC flat: {input_size} floats")
        lines.append(f" *   Output        : {out_shape_str}  ->  {output_size} class scores")
        lines.append(" */")
        lines.append("")
        lines.append('#include "model.h"')
        lines.append("")
        lines.append("// ---------------------------------------------------------------------------")
        lines.append("// Buffer sizes")
        lines.append("// ---------------------------------------------------------------------------")
        lines.append(f"#define INPUT_SIZE  {input_size}   // {nhwc_comment}")
        lines.append(f"#define OUTPUT_SIZE {output_size}")
        lines.append("")
        lines.append("// Global arrays — keeps them off the stack (avoids stack overflow)")
        lines.append("static float input_buf[INPUT_SIZE];")
        lines.append("static float output_buf[OUTPUT_SIZE];")
        lines.append("static bool  _done = false;")
        lines.append("")
        lines.append("// ---------------------------------------------------------------------------")
        lines.append("// setup")
        lines.append("// ---------------------------------------------------------------------------")
        lines.append("void setup() {")
        lines.append("    Serial.begin(115200);")
        lines.append("    while (!Serial) {}   // wait for USB serial on Giga R1")
        lines.append("")
        lines.append("    // Seed RNG from a floating ADC pin (unconnected = noise)")
        lines.append("    randomSeed(analogRead(A0));")
        lines.append("")
        lines.append("    // Fill input with random floats in [-1.0, 1.0]")
        lines.append("    // Replace this block with real sensor data in your application.")
        lines.append("    for (int i = 0; i < INPUT_SIZE; ++i) {")
        lines.append("        // random(-1000, 1001) gives integers in [-1000, 1000]")
        lines.append("        input_buf[i] = (float)random(-1000, 1001) / 1000.0f;")
        lines.append("    }")
        lines.append("")
        lines.append('    Serial.println("Input buffer filled with random data.");')
        lines.append('    Serial.println("Running model_forward...");')
        lines.append('    Serial.println();')
        lines.append("}")
        lines.append("")
        lines.append("// ---------------------------------------------------------------------------")
        lines.append("// loop — runs inference once, then halts")
        lines.append("// ---------------------------------------------------------------------------")
        lines.append("void loop() {")
        lines.append("    if (_done) return;")
        lines.append("    _done = true;")
        lines.append("")
        lines.append("    // model_forward prints profiling checkpoints (Serial.print) internally")
        lines.append("    model_forward(input_buf, output_buf);")
        lines.append("")
        lines.append("    // Print output class scores")
        lines.append('    Serial.println();')
        lines.append('    Serial.println("Output scores:");')
        lines.append("    for (int i = 0; i < OUTPUT_SIZE; ++i) {")
        lines.append('        Serial.print("  class ");')
        lines.append("        Serial.print(i);")
        lines.append('        Serial.print(": ");')
        lines.append("        Serial.println(output_buf[i], 6);")
        lines.append("    }")
        lines.append("")
        lines.append("    // Find argmax")
        lines.append("    int best = 0;")
        lines.append("    for (int i = 1; i < OUTPUT_SIZE; ++i) {")
        lines.append("        if (output_buf[i] > output_buf[best]) best = i;")
        lines.append("    }")
        lines.append('    Serial.print("Predicted class: ");')
        lines.append("    Serial.println(best);")
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

    def _assign_buffer_slots(
        self,
        order: List[IRNode],
        buffer_sizes: Dict[str, int],
        last_use: Dict[str, IRNode],
    ) -> Tuple[Dict[str, int], Dict[int, int], int]:
        """
        Assign each buffer-producing node to a reusable slot using interval graph coloring.
        Relu nodes are skipped (they share their input's slot, in-place).
        Returns (slot_assignments, slot_sizes, num_slots).
        """
        order_index = {n.name: i for i, n in enumerate(order)}

        # last_use_idx: node_name -> index of last node that uses it
        last_use_idx: Dict[str, int] = {}
        for node in order:
            if node.op_type in ('input', 'method_size'):
                continue
            if node.op_type == 'relu':
                continue  # relu does not get its own slot
            if not self._has_buffer(node):
                continue
            def_idx = order_index[node.name]
            lu_node = last_use.get(node.name, node)
            last_use_idx[node.name] = order_index[lu_node.name]

        # Extend relu input's last_use_idx to when relu's output is last used
        for node in order:
            if node.op_type == 'relu' and node.inputs:
                inp = node.inputs[0]
                if inp.name not in last_use_idx:
                    continue
                lu_relu = last_use.get(node.name, node)
                relu_last_idx = order_index[lu_relu.name]
                last_use_idx[inp.name] = max(last_use_idx[inp.name], relu_last_idx)

        # Build list of (node_name, def_idx, last_use_idx, size) for slot assignment
        intervals: List[Tuple[str, int, int, int]] = []
        for node in order:
            if node.op_type in ('input', 'method_size', 'relu'):
                continue
            if not self._has_buffer(node):
                continue
            def_idx = order_index[node.name]
            lu_idx = last_use_idx[node.name]
            size = buffer_sizes.get(node.name, 1024)
            intervals.append((node.name, def_idx, lu_idx, size))

        intervals.sort(key=lambda x: x[1])  # sort by def_idx

        # Greedy interval coloring: assign to lowest free slot
        slot_assignments: Dict[str, int] = {}
        slot_last_use: Dict[int, int] = {}
        slot_sizes: Dict[int, int] = {}

        for node_name, def_idx, lu_idx, size in intervals:
            found_slot = None
            for slot_id in sorted(slot_last_use.keys()):
                if slot_last_use[slot_id] < def_idx:
                    found_slot = slot_id
                    break
            if found_slot is None:
                found_slot = len(slot_last_use)
                slot_last_use[found_slot] = -1
            slot_assignments[node_name] = found_slot
            slot_last_use[found_slot] = lu_idx
            slot_sizes[found_slot] = max(slot_sizes.get(found_slot, 0), size)

        num_slots = len(slot_sizes)
        return slot_assignments, slot_sizes, num_slots

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

        if self.arduino_mode:
            lines.append("#include <Arduino.h>")
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
        slot_assignments, slot_sizes, num_slots = self._assign_buffer_slots(order, buffer_sizes, last_use)
        self._slot_assignments = slot_assignments
        output_node = self.ir_graph.outputs[0] if self.ir_graph.outputs else None

        base_indent = "    "

        lines.append("void model_forward(const float* input, float* output) {")
        if self._has_profiling_nodes():
            if self.arduino_mode:
                lines.append(base_indent + "unsigned long _t_start, _t_end;")
                lines.append(base_indent + "_t_start = micros();")
            else:
                lines.append(base_indent + "clock_t _t_start, _t_end;")
                lines.append(base_indent + "_t_start = clock();")

        # Flat slot declarations at function top (no nesting)
        for slot_id in range(num_slots):
            lines.append(base_indent + f"float slot_{slot_id}[{slot_sizes[slot_id]}];")
        if num_slots > 0:
            lines.append("")

        for node in order:
            if node.op_type in ('input', 'method_size'):
                node_code = self._generate_node_code(node)
                if node_code:
                    for line in node_code:
                        lines.append(base_indent + line)
                continue

            lines.append(base_indent + f"// {node.name} [{node.op_type}]")
            node_code = self._generate_node_code(node)
            if node_code:
                for line in node_code:
                    lines.append(base_indent + line)
            if output_node is not None and node.name == output_node.name:
                size = buffer_sizes[node.name]
                buf_name = self._get_buffer_name(node)
                lines.append(base_indent + f"memcpy(output, {buf_name}, {size} * sizeof(float));")

        self._slot_assignments = None  # clear so other code paths don't use stale slots
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
        """Generate code for ReLU operation (in-place, no memcpy)."""
        input_buffer = self._get_input_buffer(node, 0)
        buffer_sizes = self._calculate_buffer_sizes()
        size = buffer_sizes.get(node.name, 1024)
        return [f"relu({input_buffer}, {size});"]
    
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

    def _get_buffer_name(self, node: IRNode, slot_assignments: Optional[Dict[str, int]] = None) -> str:
        """Get the C variable name for a node's output buffer."""
        if node.op_type == 'input':
            return 'input'
        slots = slot_assignments if slot_assignments is not None else getattr(self, '_slot_assignments', None)
        if slots is not None:
            # Relu shares its input's slot (in-place); relu nodes are not in slot_assignments
            if node.op_type == 'relu' and node.inputs:
                return f"slot_{slots[node.inputs[0].name]}"
            if node.name in slots:
                return f"slot_{slots[node.name]}"
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

