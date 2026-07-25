"""
Main compiler entry point for PyTorch to C compilation
"""

import torch
from typing import Optional

from .frontend.fx_tracer import FXTracer
from .lowering.lower import Lowering
from .codegen.backend_registry import create_printer
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
        output_dir: Optional[str] = "generated",
        return_ir_only: bool = False,
        backend: str = "c",
    ) -> IRGraph:
        """
        Compile a PyTorch model to C code.
        
        Args:
            model: The PyTorch nn.Module to compile
            example_input: An example input tensor for tracing
            output_dir: Directory to write generated C files to (None to skip)
            return_ir_only: If True, only return IR (skip code generation)
            
        Returns:
            The IR graph (for inspection/debugging)
        """
        if not return_ir_only:
            self._log("=" * 60)
            self._log("PyTorch to C Compiler - Phase 1")
            self._log("=" * 60)
        
        # Step 1: Frontend - Trace with torch.fx
        if not return_ir_only:
            self._log("\n[1/3] Tracing model with torch.fx...")
        fx_graph = self.tracer.trace_model(model, example_input)
        if not return_ir_only:
            self._log(f"  ✓ Traced {len(list(fx_graph.graph.nodes))} nodes")
        
        if self.verbose and not return_ir_only:
            self._log("\nFX Graph:")
            self._log(self.tracer.print_graph(fx_graph))
        
        # Step 2: Lowering - Convert to IR with shape inference
        if not return_ir_only:
            self._log("\n[2/3] Lowering FX graph to IR...")
        ir_graph = self.lowering.lower_fx_graph(fx_graph, example_input)
        if not return_ir_only:
            self._log(f"  ✓ Created {len(ir_graph.nodes)} IR nodes")
            self._log(f"  ✓ Extracted {len(ir_graph.parameters)} parameters")
        
        # Log shape information
        nodes_with_shapes = sum(1 for node in ir_graph.nodes if node.output_shape is not None)
        if not return_ir_only:
            self._log(f"  ✓ Inferred shapes for {nodes_with_shapes}/{len(ir_graph.nodes)} nodes")
        
        if self.verbose and not return_ir_only:
            self._log("\nIR Graph:")
            self._log(ir_graph.print_graph())
        
        # Validate IR graph
        ir_graph.validate()
        if not return_ir_only:
            self._log("  ✓ IR graph validated")
        
        # If return_ir_only, skip code generation
        if return_ir_only or output_dir is None:
            return ir_graph
        
        # Step 3: Code Generation
        self._log("\n[3/3] Generating code...")
        printer = create_printer(backend, ir_graph)
        printer.generate_all(output_dir)
        self._log(f"  ✓ Generated files in: {output_dir}/")
        if backend == "c":
            self._log(f"    - model.h")
            self._log(f"    - model.c")
            self._log(f"    - weights.h")
        elif backend == "triton":
            self._log(f"    - model.py")
            self._log(f"    - weights.npz")
        
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
        if self.verbose:
            print(message)


def compile_model(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    output_dir: Optional[str] = "generated",
    verbose: bool = True,
    return_ir: bool = False,
    backend: str = "c",
) -> IRGraph:
    """
    Convenience function to compile a PyTorch model to C.
    
    Args:
        model: The PyTorch nn.Module to compile
        example_input: An example input tensor for tracing
        output_dir: Directory to write generated C files to (None to skip)
        verbose: If True, print compilation progress
        return_ir: If True, only return IR graph (skip code generation)
        backend: Code generator backend ('c' or 'triton')
        
    Returns:
        The IR graph
        
    Example:
        >>> model = MyModel()
        >>> example_input = torch.randn(1, 3, 32, 32)
        >>> ir_graph = compile_model(model, example_input, "output")
        
        # Get IR only (for quantization)
        >>> ir_graph = compile_model(model, example_input, return_ir=True)
    """
    compiler = PyTorchToCCompiler(verbose=verbose and not return_ir)
    return compiler.compile(
        model, example_input, output_dir, return_ir_only=return_ir, backend=backend
    )

