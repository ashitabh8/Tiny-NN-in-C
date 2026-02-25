import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.pytorch_to_c.compiler import compile_model
from src.pytorch_to_c.codegen.c_printer import CPrinter
from src.pytorch_to_c.profiling import ProfilingRule, ProfilingTransform

from IDK_project_examples.models.resnet_1 import TinyResNet


def main():
    print("=" * 60)
    print("Profiling Example - C code with timing and labels")
    print("=" * 60)

    output_dir = "IDK_project_examples/generated_code/arduino_resnet_1_very_tinyseismic_profiling_code"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Create model and example input
    print("\n[1/4] Creating model and example input...")
    # model = SmallNet()
    model = TinyResNet(in_channels=4)
    model.eval()
    # NCHW: batch=1, in_channels=6, H=7, W=256
    example_input = torch.randn(1, 4, 7, 256)

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
        ProfilingRule(r"stage1_block1_relu2", label="stage1_block1_relu2"),
        ProfilingRule(r"stage2_block1_relu2", label="stage2_block0_relu2"),
        ProfilingRule(r"final_softmax", label="Final Exit"),
    ]
    ir_graph = ProfilingTransform(rules).apply(ir_graph)

    # 4. Generate C to output_dir
    print(f"[4/4] Generating C to {output_dir}/...")
    printer = CPrinter(ir_graph, arduino_mode=True)
    printer.generate_all(output_dir)

    print("\n" + "=" * 60)
    print("Done. Generated files:")
    print("=" * 60)
    print(f"   {output_dir}/model.cpp   - model_forward() with printf labels + clock() timing")
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