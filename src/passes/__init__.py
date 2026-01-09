"""
Compiler passes for IR optimization.

Passes transform the IR graph to optimize for:
- Performance (fewer operations)
- Memory (smaller buffers)
- Code size

Usage:
    from src.passes import FuseDequantQuantPass
    
    ir_graph = compile_model(model, input, return_ir=True)
    
    # Apply optimization passes
    pass_instance = FuseDequantQuantPass()
    optimized_ir = pass_instance.apply(ir_graph)
"""

from .base import IRPass
from .fuse_dequant_quant import FuseDequantQuantPass

__all__ = [
    'IRPass',
    'FuseDequantQuantPass',
]

