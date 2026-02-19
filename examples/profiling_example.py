"""
Profiling Example - Generate C with timing and labels at selected ops
=====================================================================

Shows how to wrap specific float nodes with ProfilingWrapperNode so the
generated C prints a label and elapsed time (ms) for each wrapped op.
Use this to profile which layers are slow on device.

Usage:
    python examples/profiling_example.py

Generated files will be in: tmp/profiling_example/

To run and see timing output:
    python examples/run_generated_inference.py tmp/profiling_example
    (Note: run_generated_main.c is set up for 1,1,200,10 input; this example
     uses a smaller model. For a quick test, compile and run the generated
     model with your own main that calls model_forward and prints output.)
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pytorch_to_c.compiler import compile_model
from src.pytorch_to_c.codegen.c_printer import CPrinter
from src.pytorch_to_c.profiling import ProfilingRule, ProfilingTransform


class SmallNet(nn.Module):
    """Small conv + FC net for profiling demo."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 4, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


def main():
    print("=" * 60)
    print("Profiling Example - C code with timing and labels")
    print("=" * 60)

    output_dir = "tmp/profiling_example"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Create model and example input
    print("\n[1/4] Creating model and example input...")
    model = SmallNet()
    model.eval()
    example_input = torch.randn(1, 2, 4, 4)

    # 2. Compile to IR (no codegen yet)
    print("[2/4] Compiling to IR...")
    ir_graph = compile_model(
        model=model,
        example_input=example_input,
        output_dir=None,
        return_ir=True,
        verbose=False,
    )

    # 3. Apply profiling: wrap selected nodes with label + timing
    print("[3/4] Applying profiling rules...")
    rules = [
        ProfilingRule(r"conv", label="conv"),
        ProfilingRule(r"relu", label="relu"),
        ProfilingRule(r"fc", label="fc"),
    ]
    ir_graph = ProfilingTransform(rules).apply(ir_graph)

    # 4. Generate C to output_dir
    print(f"[4/4] Generating C to {output_dir}/...")
    printer = CPrinter(ir_graph)
    printer.generate_all(output_dir)

    print("\n" + "=" * 60)
    print("Done. Generated files:")
    print("=" * 60)
    print(f"   {output_dir}/model.c   - model_forward() with printf labels + clock() timing")
    print(f"   {output_dir}/model.h")
    print(f"   {output_dir}/weights.h")
    print(f"   {output_dir}/nn_ops_float.h")
    print()
    print("In model.c, wrapped ops emit:")
    print('   printf("your_label\\n");')
    print("   clock_t _t0 = clock();")
    print("   ... op C code ...")
    print("   clock_t _t1 = clock();")
    print('   printf("  %.2f ms\\n", ...);')
    print()
    print("To use in your own pipeline:")
    print("  from src.pytorch_to_c.profiling import ProfilingRule, ProfilingTransform")
    print("  ir = compile_model(model, example_input, return_ir=True)")
    print('  ir = ProfilingTransform([ProfilingRule(r"layer_name", label="my_label")]).apply(ir)')
    print("  CPrinter(ir).generate_all(output_dir)")
    print()


if __name__ == "__main__":
    main()
