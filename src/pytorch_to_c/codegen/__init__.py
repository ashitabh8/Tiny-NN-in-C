"""Code generation module for C and Triton backends."""

from .c_printer import CPrinter, generate_c_code
from .triton_printer import TritonPrinter, generate_triton_code
from .backend_registry import (
    TESTED_BACKENDS,
    check_backend,
    create_printer,
    get_printer_class,
    list_backends,
)
from .memory_planner import (
    assign_buffer_slots,
    calculate_buffer_sizes,
    compute_buffer_last_use,
    node_has_buffer,
)
from .naming import sanitize_name

__all__ = [
    "CPrinter",
    "TritonPrinter",
    "generate_c_code",
    "generate_triton_code",
    "TESTED_BACKENDS",
    "check_backend",
    "create_printer",
    "get_printer_class",
    "list_backends",
    "assign_buffer_slots",
    "calculate_buffer_sizes",
    "compute_buffer_last_use",
    "node_has_buffer",
    "sanitize_name",
]
