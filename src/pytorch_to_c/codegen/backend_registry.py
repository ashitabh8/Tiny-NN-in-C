"""
Backend registry for code generators.

Maps backend names to printer classes and tracks which backends have passed
the project's verification suite.
"""

from __future__ import annotations

import warnings
from typing import Dict, Set, Type

# Backends that have passed verification (no warning emitted).
TESTED_BACKENDS: Set[str] = {"c", "triton"}

# Populated after TritonPrinter is defined (avoid circular import at module load).
_BACKENDS: Dict[str, type] = {}


def register_backend(name: str, printer_cls: type) -> None:
    """Register a code generator backend."""
    _BACKENDS[name] = printer_cls


def get_printer_class(name: str) -> type:
    """Return the printer class for a backend name."""
    _ensure_backends_loaded()
    if name not in _BACKENDS:
        known = ", ".join(sorted(_BACKENDS.keys())) or "(none)"
        raise ValueError(
            f"Unknown codegen backend '{name}'. Known backends: {known}"
        )
    return _BACKENDS[name]


def check_backend(name: str) -> None:
    """
    Validate backend exists; warn if it is not in TESTED_BACKENDS.

    Raises:
        ValueError: if backend name is unknown.
    """
    get_printer_class(name)
    if name not in TESTED_BACKENDS:
        warnings.warn(
            f"Codegen backend '{name}' is registered but has NOT passed the "
            f"project verification suite. Output may be incorrect or incomplete. "
            f"Tested backends: {sorted(TESTED_BACKENDS)}.",
            UserWarning,
            stacklevel=3,
        )


def list_backends() -> list:
    """Return sorted list of registered backend names."""
    _ensure_backends_loaded()
    return sorted(_BACKENDS.keys())


def _ensure_backends_loaded() -> None:
    if _BACKENDS:
        return
    from .c_printer import CPrinter
    from .triton_printer import TritonPrinter

    register_backend("c", CPrinter)
    register_backend("triton", TritonPrinter)


def create_printer(backend: str, ir_graph, **kwargs):
    """Instantiate a printer for the given backend."""
    check_backend(backend)
    cls = get_printer_class(backend)
    return cls(ir_graph, **kwargs)
