"""
Profiling Wrapper Node - Wraps a float op with timing and optional label print
"""

from typing import List, Optional

from ...ir.node import IRNode


def _escape_label_for_c(s: str) -> str:
    """Escape label so it is safe inside a C double-quoted string."""
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )


class ProfilingWrapperNode(IRNode):
    """
    Wraps a float IR node: emits the same C op plus a user label (printed at start)
    and timing (clock() start/end, elapsed ms printed).
    """

    def __init__(self, original_node: IRNode, label: Optional[str] = None):
        """
        Create a profiling wrapper from an existing float node.

        Args:
            original_node: The float node to wrap
            label: String printed at the start of this node (if None, use node.name)
        """
        super().__init__(
            name=original_node.name,
            op_type=original_node.op_type,
            output_shape=original_node.output_shape,
            dtype="float32",
            metadata=original_node.metadata.copy() if original_node.metadata else {},
        )
        self.inputs = original_node.inputs.copy() if original_node.inputs else []
        self.users = original_node.users.copy() if original_node.users else []
        self.label = label if label is not None else original_node.name
        self._original_node = original_node

    def generate_c_code(self, c_printer) -> List[str]:
        """
        Generate C code: vanilla op code, then print cumulative elapsed time from
        _t_start (declared once at the top of model_forward by c_printer).

        Standard mode : uses clock() / CLOCKS_PER_SEC + printf
        Arduino mode  : uses micros() + Serial.print  (no stdio/time headers needed)
        """
        lines: List[str] = []

        inner = self._generate_inner_code(c_printer)
        lines.extend(inner)

        escaped = _escape_label_for_c(self.label)
        arduino = getattr(c_printer, 'arduino_mode', False)

        if arduino:
            lines.append("_t_end = micros();")
            ms_expr = "(_t_end - _t_start) / 1000.0f"
            lines.append(f'Serial.print("{escaped}: ");')
            lines.append(f'Serial.print({ms_expr}, 2);')
            lines.append('Serial.println(" ms");')
        else:
            lines.append("_t_end = clock();")
            lines.append(f'printf("{escaped}: %.2f ms\\n", 1000.0 * (_t_end - _t_start) / CLOCKS_PER_SEC);')

        return lines

    def _generate_inner_code(self, c_printer) -> List[str]:
        """Dispatch to the same logic as vanilla float codegen for this op_type."""
        op = self.op_type
        if op == "conv2d":
            return c_printer._generate_conv2d(self)
        if op == "linear":
            return c_printer._generate_linear(self)
        if op == "relu":
            return c_printer._generate_relu(self)
        if op == "batchnorm":
            return c_printer._generate_batchnorm(self)
        if op == "softmax":
            return c_printer._generate_softmax(self)
        if op == "add":
            return c_printer._generate_add(self)
        if op == "method_mean":
            return c_printer._generate_mean(self)
        if op == "adaptive_avg_pool":
            return c_printer._generate_adaptive_avg_pool(self)
        if op in ("method_view", "method_flatten"):
            return c_printer._generate_flatten_or_view(self)
        return [f"// ProfilingWrapperNode: unsupported op_type '{op}'"]
