"""
Main compiler entry point for PyTorch to C compilation
"""

import os
import torch
from typing import Optional

from .frontend.fx_tracer import FXTracer
from .lowering.lower import Lowering
from .codegen.c_printer import CPrinter
from .ir.graph import IRGraph


class PyTorchToCCompiler:
    """
    Main compiler class that orchestrates the compilation pipeline.
    
    Pipeline:
    1. Frontend: Trace PyTorch model with torch.fx
    2. Lowering: Convert FX graph to custom IR
    3. Codegen: Generate C code (model.c, model.h, weights.h)
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the compiler.
        
        Args:
            verbose: If True, print detailed compilation information
        """
        self.verbose = verbose
        self.tracer = FXTracer()
        self.lowering = Lowering()
    
    def compile(
        self,
        model: torch.nn.Module,
        example_input: torch.Tensor,
        output_dir: str = "generated"
    ) -> IRGraph:
        """
        Compile a PyTorch model to C code.
        
        Args:
            model: The PyTorch nn.Module to compile
            example_input: An example input tensor for tracing
            output_dir: Directory to write generated C files to
            
        Returns:
            The IR graph (for inspection/debugging)
        """
        self._log("=" * 60)
        self._log("PyTorch to C Compiler - Phase 1")
        self._log("=" * 60)
        
        # Step 1: Frontend - Trace with torch.fx
        self._log("\n[1/3] Tracing model with torch.fx...")
        fx_graph = self.tracer.trace_model(model, example_input)
        self._log(f"  ✓ Traced {len(list(fx_graph.graph.nodes))} nodes")
        
        if self.verbose:
            self._log("\nFX Graph:")
            self._log(self.tracer.print_graph(fx_graph))
        
        # Step 2: Lowering - Convert to IR
        self._log("\n[2/3] Lowering FX graph to IR...")
        ir_graph = self.lowering.lower_fx_graph(fx_graph)
        self._log(f"  ✓ Created {len(ir_graph.nodes)} IR nodes")
        self._log(f"  ✓ Extracted {len(ir_graph.parameters)} parameters")
        
        if self.verbose:
            self._log("\nIR Graph:")
            self._log(ir_graph.print_graph())
        
        # Validate IR graph
        ir_graph.validate()
        self._log("  ✓ IR graph validated")
        
        # Step 3: Code Generation
        self._log("\n[3/3] Generating C code...")
        printer = CPrinter(ir_graph)
        printer.generate_all(output_dir)
        self._log(f"  ✓ Generated files in: {output_dir}/")
        self._log(f"    - model.h")
        self._log(f"    - model.c")
        self._log(f"    - weights.h")
        
        # Summary
        self._log("\n" + "=" * 60)
        self._log("Compilation completed successfully!")
        self._log("=" * 60)
        
        # Calculate sizes
        total_params = sum(p.size * 4 for p in ir_graph.parameters.values())  # 4 bytes per float
        self._log(f"\nModel Statistics:")
        self._log(f"  Total parameters: {sum(p.size for p in ir_graph.parameters.values())}")
        self._log(f"  Parameter memory: {total_params / 1024:.2f} KB")
        self._log(f"  Number of operations: {len(ir_graph.nodes)}")
        
        return ir_graph
    
    def _log(self, message: str) -> None:
        """Print a log message if verbose mode is enabled."""
        if self.verbose or True:  # Always log for now
            print(message)


def compile_model(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    output_dir: str = "generated",
    verbose: bool = True
) -> IRGraph:
    """
    Convenience function to compile a PyTorch model to C.
    
    Args:
        model: The PyTorch nn.Module to compile
        example_input: An example input tensor for tracing
        output_dir: Directory to write generated C files to
        verbose: If True, print compilation progress
        
    Returns:
        The IR graph
        
    Example:
        >>> model = MyModel()
        >>> example_input = torch.randn(1, 3, 32, 32)
        >>> ir_graph = compile_model(model, example_input, "output")
    """
    compiler = PyTorchToCCompiler(verbose=verbose)
    return compiler.compile(model, example_input, output_dir)

