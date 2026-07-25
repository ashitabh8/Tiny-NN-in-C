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
from .memory_planner import (
    assign_buffer_slots,
    calculate_buffer_sizes,
    compute_buffer_last_use,
    node_has_buffer,
)
from .naming import sanitize_name
from .backend_registry import check_backend
from .parallel_regions import ParallelRegion, find_parallel_regions

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
        check_backend("c")
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
        headers = ["nn_ops_float.h", "nn_ops_int8.h", "nn_ops_int16.h", "nn_ops_int4.h"]
        
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
            c_name = sanitize_name(param_name)
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
        lines.append("    pinMode(LED_BUILTIN, OUTPUT);")
        lines.append("")
        lines.append("    // Slow blink = setup started")
        lines.append("    for (int i = 0; i < 10; i++) {")
        lines.append("        digitalWrite(LED_BUILTIN, HIGH); delay(500);")
        lines.append("        digitalWrite(LED_BUILTIN, LOW);  delay(500);")
        lines.append("    }")
        lines.append("    Serial.begin(115200);")
        lines.append("    // unsigned long start = millis();")
        lines.append("    while (!Serial) {}")
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

    def _needs_compression_header(self) -> bool:
        """True if graph uses int4 or palettization weight-only kernels."""
        for node in self.ir_graph.nodes:
            qp = node.metadata.get("quant_params", {}) if node.metadata else {}
            strat = qp.get("strategy", "")
            if strat in (
                "static_int4_per_group",
                "dynamic_int4_per_group",
                "palettization",
            ):
                return True
        return False
    
    def _get_buffer_dtype(self, node: IRNode) -> str:
        """
        Get C data type for a node's buffer.
        
        All IRNodes now have get_c_dtype() which maps dtype -> C type.
        """
        return node.get_c_dtype()

    def _get_sizeof_expr(self, node: IRNode) -> str:
        """Return C sizeof() expression matching a node buffer dtype."""
        c_dtype = self._get_buffer_dtype(node)
        if c_dtype == "int8_t":
            return "sizeof(int8_t)"
        if c_dtype == "int16_t":
            return "sizeof(int16_t)"
        if c_dtype == "int32_t":
            return "sizeof(int32_t)"
        return "sizeof(float)"
    
    def _has_buffer(self, node: IRNode) -> bool:
        return node_has_buffer(node)

    def _has_profiling_nodes(self) -> bool:
        """True if the graph contains any ProfilingWrapperNode (needs time.h, stdio.h)."""
        if ProfilingWrapperNode is None:
            return False
        return any(isinstance(n, ProfilingWrapperNode) for n in self.ir_graph.nodes)

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
        if self._needs_compression_header():
            lines.append("#include \"nn_ops_int4.h\"")
        
        lines.append("")
        lines.append("#include <string.h>")
        if self._has_profiling_nodes() and not self.arduino_mode:
            lines.append("#include <time.h>")
            lines.append("#include <stdio.h>")
        lines.append("")
        
        buffer_sizes = calculate_buffer_sizes(self.ir_graph)
        order = self.ir_graph.topological_sort()
        last_use = compute_buffer_last_use(order)
        slot_assignments, slot_sizes, slot_c_dtypes, num_slots = assign_buffer_slots(
            order, buffer_sizes, last_use
        )
        slot_dtypes = {
            sid: self._ir_dtype_to_c(slot_c_dtypes[sid]) for sid in slot_c_dtypes
        }
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
            lines.append(
                base_indent
                + f"static {slot_dtypes[slot_id]} slot_{slot_id}[{slot_sizes[slot_id]}];"
            )
        if num_slots > 0:
            lines.append("")

        # Parallel regions (LQER diamonds): the two chains are emitted as
        # OpenMP sections. Without -fopenmp the pragmas are ignored and the
        # sections run sequentially (identical numerics).
        regions = find_parallel_regions(order)
        region_of: Dict[str, ParallelRegion] = {}
        for region in regions:
            for region_node in region.main_chain + region.branch_chain:
                region_of[region_node.name] = region
        emitted_region_nodes = set()

        for node in order:
            if node.name in emitted_region_nodes:
                continue

            region = region_of.get(node.name)
            if region is not None:
                lines.extend(self._generate_parallel_region(region, base_indent))
                emitted_region_nodes.update(region.node_names)
                continue

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
                lines.append(base_indent + f"memcpy(output, {buf_name}, {size} * {self._get_sizeof_expr(node)});")

        self._slot_assignments = None  # clear so other code paths don't use stale slots
        lines.append("}")
        lines.append("")
        return "\n".join(lines)
    
    def _generate_parallel_region(self, region: ParallelRegion, base_indent: str) -> List[str]:
        """
        Emit a fan-out/fan-in diamond as two OpenMP sections.

        The chains are topologically independent (both only consume the
        shared source), so they may run concurrently. Buffer aliasing across
        the region is prevented by the memory planner (see
        memory_planner.assign_buffer_slots). Compile with -fopenmp to enable;
        without it the pragmas are ignored and the code runs sequentially.
        """
        lines = []
        label = region.join.metadata.get('lqer_corrected_layer', region.join.name)
        lines.append(base_indent + f"// parallel region: quantized path || error-correction branch ('{label}')")
        lines.append(base_indent + "#pragma omp parallel sections")
        lines.append(base_indent + "{")

        for chain, chain_label in (
            (region.main_chain, "quantized path"),
            (region.branch_chain, "correction branch"),
        ):
            lines.append(base_indent + "    #pragma omp section")
            lines.append(base_indent + "    {")
            for node in chain:
                lines.append(base_indent + f"        // {node.name} [{node.op_type}] ({chain_label})")
                for line in self._generate_node_code(node):
                    lines.append(base_indent + "        " + line)
            lines.append(base_indent + "    }")

        lines.append(base_indent + "}")
        return lines

    def _calculate_buffer_sizes(self) -> Dict[str, int]:
        return calculate_buffer_sizes(self.ir_graph)

    @staticmethod
    def _ir_dtype_to_c(dtype: str) -> str:
        if dtype == "int8":
            return "int8_t"
        if dtype == "int16":
            return "int16_t"
        return "float"

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

        elif node.op_type == 'conv1d':
            return self._generate_conv1d(node)

        elif node.op_type == 'linear':
            return self._generate_linear(node)

        elif node.op_type == 'relu':
            return self._generate_relu(node)

        elif node.op_type == 'gelu':
            return self._generate_gelu(node)

        elif node.op_type == 'batchnorm':
            return self._generate_batchnorm(node)

        elif node.op_type == 'batchnorm1d':
            return self._generate_batchnorm1d(node)
        
        elif node.op_type == 'softmax':
            return self._generate_softmax(node)
        
        elif node.op_type == 'add':
            return self._generate_add(node)
        
        elif node.op_type == 'method_mean':
            return self._generate_mean(node)

        elif node.op_type == 'adaptive_avg_pool':
            return self._generate_adaptive_avg_pool(node)

        elif node.op_type == 'mul':
            if node.output_shape is None:
                return []  # FX shape-arithmetic mul (e.g., C * S in reshape args)
            return self._generate_mul(node)

        elif node.op_type in ('method_view', 'method_flatten', 'method_reshape'):
            return self._generate_flatten_or_view(node)

        elif node.op_type == 'method_unsqueeze':
            return self._generate_unsqueeze(node)

        elif node.op_type == 'method_squeeze':
            return self._generate_squeeze(node)

        elif node.op_type == 'method_permute':
            return self._generate_permute(node)

        elif node.op_type in ('method_size', 'method_getattr', 'method_getitem'):
            return []  # scalar integer, no buffer

        elif node.op_type == 'mul' and node.output_shape is None:
            return []  # shape arithmetic only

        else:
            return [f"// Unsupported operation: {node.op_type}"]
    
    def _generate_conv2d(self, node: IRNode) -> List[str]:
        """Generate code for Conv2d operation."""
        lines = []
        
        input_buffer = self._get_input_buffer(node, 0)
        output_buffer = self._get_buffer_name(node)
        weight_name = sanitize_name(node.metadata['weight_name'])
        bias_name = sanitize_name(node.metadata['bias_name']) if node.metadata.get('bias_name') else 'NULL'
        
        # Extract parameters
        kernel_size = node.metadata['kernel_size']
        stride = node.metadata['stride']
        padding = node.metadata['padding']
        in_channels = node.metadata['in_channels']
        out_channels = node.metadata['out_channels']
        groups = node.metadata['groups'] if 'groups' in node.metadata else 1
        
        # Convert to scalars if tuples
        k_h, k_w = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        s_h, s_w = stride if isinstance(stride, (tuple, list)) else (stride, stride)
        p_h, p_w = padding if isinstance(padding, (tuple, list)) else (padding, padding)
        
        # Get input shape from the input node — require 4D NCHW shape
        if not (node.inputs
                and node.inputs[0].output_shape
                and len(node.inputs[0].output_shape) == 4):
            raise ValueError(
                f"{node.name} (conv2d): input shape unavailable; "
                f"run compile_model with example_input so spatial dims can be determined"
            )
        in_h, in_w = node.inputs[0].output_shape[2], node.inputs[0].output_shape[3]
        
        if groups > 1:
            if groups != in_channels or out_channels != in_channels:
                raise ValueError(
                    f"{node.name}: grouped conv is only supported for depthwise "
                    f"(groups==in_channels==out_channels), got groups={groups}, "
                    f"in_channels={in_channels}, out_channels={out_channels}"
                )
            lines.append(
                f"depthwise_conv2d_nhwc({input_buffer}, {in_h}, {in_w}, {in_channels}, "
                f"{weight_name}, {k_h}, {k_w}, {bias_name}, "
                f"{s_h}, {s_w}, {p_h}, {p_w}, {output_buffer});"
            )
        else:
            # PyTorch-style symmetric padding (pad per side)
            lines.append(
                f"conv2d_nhwc({input_buffer}, {in_h}, {in_w}, {in_channels}, "
                f"{weight_name}, {k_h}, {k_w}, {out_channels}, "
                f"{bias_name}, {s_h}, {s_w}, {p_h}, {p_w}, {output_buffer});"
            )
        
        return lines
    
    def _generate_conv1d(self, node: IRNode) -> List[str]:
        """Generate code for Conv1d as a Conv2d call with H=1."""
        lines = []

        input_buffer = self._get_input_buffer(node, 0)
        output_buffer = self._get_buffer_name(node)
        weight_name = sanitize_name(node.metadata['weight_name'])
        bias_name = sanitize_name(node.metadata['bias_name']) if node.metadata.get('bias_name') else 'NULL'

        kernel_size = node.metadata['kernel_size']
        stride = node.metadata['stride']
        padding = node.metadata['padding']
        in_channels = node.metadata['in_channels']
        out_channels = node.metadata['out_channels']
        groups = node.metadata['groups'] if 'groups' in node.metadata else 1

        k_w = int(kernel_size[0]) if isinstance(kernel_size, (tuple, list)) else int(kernel_size)
        s_w = int(stride[0]) if isinstance(stride, (tuple, list)) else int(stride)
        p_w = int(padding[0]) if isinstance(padding, (tuple, list)) else int(padding)

        if not (node.inputs
                and node.inputs[0].output_shape
                and len(node.inputs[0].output_shape) == 3):
            raise ValueError(
                f"{node.name} (conv1d): input shape unavailable; "
                f"expected 3D [B, C, L]; run compile_model with example_input"
            )
        in_w = int(node.inputs[0].output_shape[2])

        if groups > 1:
            if groups != in_channels or out_channels != in_channels:
                raise ValueError(
                    f"{node.name}: grouped Conv1d only supported as depthwise "
                    f"(groups==in==out), got groups={groups}, in={in_channels}, out={out_channels}"
                )
            lines.append(
                f"depthwise_conv2d_nhwc({input_buffer}, 1, {in_w}, {in_channels}, "
                f"{weight_name}, 1, {k_w}, {bias_name}, "
                f"1, {s_w}, 0, {p_w}, {output_buffer});"
            )
        else:
            lines.append(
                f"conv2d_nhwc({input_buffer}, 1, {in_w}, {in_channels}, "
                f"{weight_name}, 1, {k_w}, {out_channels}, "
                f"{bias_name}, 1, {s_w}, 0, {p_w}, {output_buffer});"
            )

        return lines

    def _generate_linear(self, node: IRNode) -> List[str]:
        """Generate code for Linear operation."""
        lines = []
        
        input_buffer = self._get_input_buffer(node, 0)
        output_buffer = self._get_buffer_name(node)
        weight_name = sanitize_name(node.metadata['weight_name'])
        bias_name = sanitize_name(node.metadata['bias_name']) if node.metadata.get('bias_name') else 'NULL'
        
        in_features = node.metadata['in_features']
        out_features = node.metadata['out_features']
        
        rows = 1
        if node.inputs and node.inputs[0].output_shape is not None:
            shape = list(node.inputs[0].output_shape)
            if len(shape) > 0 and shape[0] == 1:
                shape = shape[1:]
            total = 1
            for dim in shape:
                total *= dim
            if in_features > 0 and total % in_features == 0:
                rows = total // in_features

        if rows == 1:
            lines.append(
                f"dense({input_buffer}, {in_features}, "
                f"{weight_name}, {bias_name}, {out_features}, {output_buffer});"
            )
        else:
            lines.append(f"for (int r = 0; r < {rows}; ++r) {{")
            lines.append(
                f"    dense({input_buffer} + r * {in_features}, {in_features}, "
                f"{weight_name}, {bias_name}, {out_features}, {output_buffer} + r * {out_features});"
            )
            lines.append("}")
        
        return lines
    
    def _generate_relu(self, node: IRNode) -> List[str]:
        """Generate code for ReLU operation (in-place, no memcpy)."""
        input_buffer = self._get_input_buffer(node, 0)
        buffer_sizes = self._calculate_buffer_sizes()
        size = buffer_sizes.get(node.name, 1024)
        return [f"relu({input_buffer}, {size});"]
    
    def _generate_gelu(self, node: IRNode) -> List[str]:
        """Generate code for GELU operation (in-place, no memcpy)."""
        input_buffer = self._get_input_buffer(node, 0)
        buffer_sizes = self._calculate_buffer_sizes()
        size = buffer_sizes.get(node.name, 1024)
        return [f"gelu({input_buffer}, {size});"]
    
    def _generate_batchnorm(self, node: IRNode) -> List[str]:
        """Generate code for BatchNorm operation."""
        lines = []
        
        input_buffer = self._get_input_buffer(node, 0)
        output_buffer = self._get_buffer_name(node)
        gamma_name = sanitize_name(node.metadata['gamma_name'])
        beta_name = sanitize_name(node.metadata['beta_name'])
        mean_name = sanitize_name(node.metadata['mean_name'])
        var_name = sanitize_name(node.metadata['var_name'])
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
    
    def _generate_batchnorm1d(self, node: IRNode) -> List[str]:
        """BatchNorm1d → batchnorm2d_nhwc with h=1, w=L, c=C."""
        lines = []

        input_buffer = self._get_input_buffer(node, 0)
        output_buffer = self._get_buffer_name(node)
        gamma_name = sanitize_name(node.metadata['gamma_name'])
        beta_name = sanitize_name(node.metadata['beta_name'])
        mean_name = sanitize_name(node.metadata['mean_name'])
        var_name = sanitize_name(node.metadata['var_name'])
        eps = node.metadata['eps']
        num_features = node.metadata['num_features']

        if not (node.inputs and node.inputs[0].output_shape and len(node.inputs[0].output_shape) == 3):
            raise ValueError(
                f"{node.name} (batchnorm1d): expected 3D input shape [B, C, L]; "
                f"run compile_model with example_input"
            )
        L = int(node.inputs[0].output_shape[2])

        lines.append(
            f"batchnorm2d_nhwc({input_buffer}, 1, {L}, {num_features}, "
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
        
        lines.append(f"memcpy({output_buffer}, {input_buffer}, {size} * {self._get_sizeof_expr(node)});")
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
    
    def _generate_mul(self, node: IRNode) -> List[str]:
        """Generate code for element-wise multiplication."""
        lines = []
        input_buffer_a = self._get_input_buffer(node, 0)
        input_buffer_b = self._get_input_buffer(node, 1)
        output_buffer = self._get_buffer_name(node)
        buffer_sizes = self._calculate_buffer_sizes()
        size = buffer_sizes.get(node.name, 1024)
        lines.append(f"for (int i = 0; i < {size}; ++i) {{")
        lines.append(f"    {output_buffer}[i] = {input_buffer_a}[i] * {input_buffer_b}[i];")
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
        kwargs = node.metadata['kwargs'] if 'kwargs' in node.metadata else {}
        args = node.metadata.get('args', ())
        
        # dim can be in kwargs or as first positional arg
        dim = kwargs['dim'] if 'dim' in kwargs else None
        if dim is None and args:
            dim = args[0] if isinstance(args[0], (list, tuple, int)) else [2, 3]
        if dim is None:
            dim = [2, 3]  # Default to spatial dims for NCHW
        
        # Get input shape to determine spatial dimensions
        input_node = node.inputs[0] if node.inputs else None
        if input_node and input_node.output_shape:
            input_shape = input_node.output_shape
            shape_no_batch = list(input_shape)
            if len(shape_no_batch) > 0 and shape_no_batch[0] == 1:
                shape_no_batch = shape_no_batch[1:]

            # Handle mean over the last dim, e.g. [B, C, I] -> [B, C]
            if isinstance(dim, int) and dim == -1 and len(shape_no_batch) == 2:
                rows, cols = shape_no_batch
                # Temporal stack output is stored NHWC [1, I=7, C]; average over time.
                if rows == 7 or cols == 7:
                    n_time = 7
                    n_chan = cols if rows == 7 else rows
                    lines.append("/* Mean over last dimension (NCL -> NLC in C) */")
                    lines.append(
                        f"mean_hwc({input_buffer}, 1, {n_time}, {n_chan}, {output_buffer});"
                    )
                    return lines
                lines.append("/* Mean over last dimension */")
                if node.dtype == "int8":
                    input_scale = node.metadata.get("input_scale", 1.0)
                    output_scale = node.metadata.get("output_scale", 1.0)
                    output_offset = node.metadata.get("output_offset", 0)
                    lines.append(
                        f"mean_last_dim_int8({input_buffer}, {rows}, {cols}, "
                        f"{input_scale}f, {output_scale}f, {output_offset}, {output_buffer});"
                    )
                else:
                    lines.append(f"mean_last_dim({input_buffer}, {rows}, {cols}, {output_buffer});")
                return lines

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
                    if node.dtype == "int8":
                        input_scale = node.metadata.get("input_scale", 1.0)
                        output_scale = node.metadata.get("output_scale", 1.0)
                        output_offset = node.metadata.get("output_offset", 0)
                        lines.append(
                            f"mean_hwc_int8({input_buffer}, {h}, {w}, {c}, "
                            f"{input_scale}f, {output_scale}f, {output_offset}, {output_buffer});"
                        )
                    else:
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
        if node.dtype == "int8":
            input_scale = node.metadata.get("input_scale", 1.0)
            output_scale = node.metadata.get("output_scale", 1.0)
            output_offset = node.metadata.get("output_offset", 0)
            lines.append(
                f"global_average_pool_2d_int8({input_buffer}, {h}, {w}, {c}, "
                f"{input_scale}f, {output_scale}f, {output_offset}, {output_buffer});"
            )
        else:
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
        lines.append(f"memcpy({output_buffer}, {input_buffer}, {size} * {self._get_sizeof_expr(node)});")
        return lines

    def _generate_unsqueeze(self, node: IRNode) -> List[str]:
        """
        Generate code for unsqueeze.
        For [C, I] -> [C, I, 1], convert row-major [C, I] into NHWC-friendly [I, 1, C].
        Other cases fallback to memcpy.
        """
        lines = []
        input_buffer = self._get_input_buffer(node, 0)
        output_buffer = self._get_buffer_name(node)

        in_shape = list(node.inputs[0].output_shape) if node.inputs and node.inputs[0].output_shape else []
        out_shape = list(node.output_shape) if node.output_shape else []
        if len(in_shape) > 0 and in_shape[0] == 1:
            in_shape = in_shape[1:]
        if len(out_shape) > 0 and out_shape[0] == 1:
            out_shape = out_shape[1:]

        if len(in_shape) == 2 and len(out_shape) == 3 and out_shape[-1] == 1:
            c, i = in_shape
            lines.append("/* unsqueeze(-1): [C, I] -> NHWC [I, 1, C] */")
            lines.append(f"for (int ii = 0; ii < {i}; ++ii) {{")
            lines.append(f"    for (int cc = 0; cc < {c}; ++cc) {{")
            lines.append(
                f"        {output_buffer}[(ii * {c}) + cc] = {input_buffer}[(cc * {i}) + ii];"
            )
            lines.append("    }")
            lines.append("}")
            return lines

        buffer_sizes = self._calculate_buffer_sizes()
        size = buffer_sizes[node.name]
        lines.append(f"memcpy({output_buffer}, {input_buffer}, {size} * {self._get_sizeof_expr(node)});")
        return lines

    def _generate_squeeze(self, node: IRNode) -> List[str]:
        """
        Generate code for squeeze.
        For [C, I, 1] -> [C, I], convert NHWC [I, 1, C] into row-major [C, I].
        Other cases fallback to memcpy.
        """
        lines = []
        input_buffer = self._get_input_buffer(node, 0)
        output_buffer = self._get_buffer_name(node)

        in_shape = list(node.inputs[0].output_shape) if node.inputs and node.inputs[0].output_shape else []
        out_shape = list(node.output_shape) if node.output_shape else []
        if len(in_shape) > 0 and in_shape[0] == 1:
            in_shape = in_shape[1:]
        if len(out_shape) > 0 and out_shape[0] == 1:
            out_shape = out_shape[1:]

        if len(in_shape) == 3 and len(out_shape) == 2 and in_shape[-1] == 1:
            c, i, _ = in_shape
            lines.append("/* squeeze(-1): NHWC [I, 1, C] -> [C, I] */")
            lines.append(f"for (int cc = 0; cc < {c}; ++cc) {{")
            lines.append(f"    for (int ii = 0; ii < {i}; ++ii) {{")
            lines.append(
                f"        {output_buffer}[(cc * {i}) + ii] = {input_buffer}[(ii * {c}) + cc];"
            )
            lines.append("    }")
            lines.append("}")
            return lines

        buffer_sizes = self._calculate_buffer_sizes()
        size = buffer_sizes[node.name]
        lines.append(f"memcpy({output_buffer}, {input_buffer}, {size} * {self._get_sizeof_expr(node)});")
        return lines

    def _generate_permute(self, node: IRNode) -> List[str]:
        """Generate code for tensor permute using generic 4D permutation."""
        lines = []
        input_buffer = self._get_input_buffer(node, 0)
        output_buffer = self._get_buffer_name(node)

        input_node = node.inputs[0] if node.inputs else None
        if input_node is None or input_node.output_shape is None:
            raise ValueError(f"{node.name} (method_permute): missing input shape")

        raw_shape = list(input_node.output_shape)
        perm_args = list(node.metadata['args'])
        if len(perm_args) == 1 and isinstance(perm_args[0], (tuple, list)):
            perm_args = list(perm_args[0])

        # permute(0,2,1) on [B,I,C] row-major matches NHWC [1,I,C]; no reorder needed.
        if (
            perm_args == [0, 2, 1]
            and len(raw_shape) == 3
            and raw_shape[0] == 1
        ):
            size = raw_shape[1] * raw_shape[2]
            lines.append(
                "/* permute(0,2,1) after linear: row-major == NLC, memcpy */"
            )
            lines.append(
                f"memcpy({output_buffer}, {input_buffer}, {size} * sizeof(float));"
            )
            return lines

        # Special-case NCHW->(B,H,C,W) when source buffer is NHWC from conv/bn/relu.
        if len(raw_shape) == 4 and raw_shape[0] == 1 and perm_args == [0, 2, 1, 3]:
            c = raw_shape[1]
            h = raw_shape[2]
            w = raw_shape[3]
            lines.append("/* permute(0,2,1,3): source NHWC [H,W,C] -> [H,C,W] */")
            lines.append(f"for (int hh = 0; hh < {h}; ++hh) {{")
            lines.append(f"    for (int cc = 0; cc < {c}; ++cc) {{")
            lines.append(f"        for (int ww = 0; ww < {w}; ++ww) {{")
            lines.append(
                f"            {output_buffer}[((hh * {c} + cc) * {w}) + ww] = "
                f"{input_buffer}[((hh * {w} + ww) * {c}) + cc];"
            )
            lines.append("        }")
            lines.append("    }")
            lines.append("}")
            return lines

        # Runtime buffers are batch-stripped when B==1, so adapt perm accordingly.
        if len(raw_shape) > 0 and raw_shape[0] == 1 and 0 in perm_args:
            dims = raw_shape[1:]
            perm = [int(p) - 1 for p in perm_args if int(p) != 0]
        else:
            dims = raw_shape
            perm = [int(p) for p in perm_args]

        if len(dims) == 0:
            raise ValueError(f"{node.name} (method_permute): invalid empty input shape")
        if len(perm) != len(dims):
            raise ValueError(
                f"{node.name} (method_permute): perm rank mismatch, perm={perm_args}, shape={raw_shape}"
            )

        if len(dims) == 3:
            lines.append(
                f"permute_3d({input_buffer}, {dims[0]}, {dims[1]}, {dims[2]}, "
                f"{perm[0]}, {perm[1]}, {perm[2]}, {output_buffer});"
            )
            return lines

        while len(dims) < 4:
            dims.append(1)

        used = set(perm)
        for axis in range(4):
            if axis not in used:
                perm.append(axis)
                used.add(axis)
            if len(perm) == 4:
                break

        if len(perm) != 4:
            raise ValueError(f"{node.name} (method_permute): failed to build 4D perm from {perm_args}")

        lines.append(
            f"permute_4d({input_buffer}, {dims[0]}, {dims[1]}, {dims[2]}, {dims[3]}, "
            f"{perm[0]}, {perm[1]}, {perm[2]}, {perm[3]}, {output_buffer});"
        )
        return lines

    def _get_buffer_name(self, node: IRNode, slot_assignments: Optional[Dict[str, int]] = None) -> str:
        """Get the C variable name for a node's output buffer."""
        if node.op_type == 'input':
            return 'input'
        slots = slot_assignments if slot_assignments is not None else getattr(self, '_slot_assignments', None)
        if slots is not None:
            # Relu shares its input's slot (in-place); relu nodes are not in slot_assignments
            if node.op_type in ('relu', 'gelu') and node.inputs:
                return f"slot_{slots[node.inputs[0].name]}"
            if node.name in slots:
                return f"slot_{slots[node.name]}"
        return f"buf_{sanitize_name(node.name)}"
    
    def _get_input_buffer(self, node: IRNode, input_idx: int) -> str:
        """Get the buffer name for a node's input."""
        if input_idx >= len(node.inputs):
            raise ValueError(f"Node {node.name} doesn't have input {input_idx}")
        
        input_node = node.inputs[input_idx]
        return self._get_buffer_name(input_node)


    def _sanitize_name(self, name: str) -> str:
        """Backward-compatible alias for sanitize_name()."""
        return sanitize_name(name)


def generate_c_code(ir_graph: IRGraph, output_dir: str) -> None:
    """
    Convenience function to generate C code from an IR graph.
    
    Args:
        ir_graph: The IR graph
        output_dir: Directory to write generated files to
    """
    printer = CPrinter(ir_graph)
    printer.generate_all(output_dir)

