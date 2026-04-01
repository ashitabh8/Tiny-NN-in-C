"""
Compile MNIST CNN to C
======================

Loads the trained MNISTConvNet and generates portable C code (float32).
No quantization is applied.

Prerequisites:
    python examples/mnist_cnn.py   # train & save weights first

Usage:
    python examples/compile_mnist_cnn.py

Generated files will be in: tmp/mnist_cnn/
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from examples.mnist_cnn import MNISTConvNet
from src.pytorch_to_c.compiler import compile_model
from src.pytorch_to_c.codegen.c_printer import CPrinter


def main():
    print("=" * 60)
    print("Compile MNIST CNN to C (float32)")
    print("=" * 60)

    OUTPUT_DIR = "tmp/mnist_cnn"
    WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "mnist_cnn_weights.pth")
    INPUT_SHAPE = (1, 1, 28, 28)  # (batch, channels, height, width)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================================================
    # Step 1: Load trained model
    # =========================================================================
    print("\n[1/4] Loading MNISTConvNet with trained weights...")

    model = MNISTConvNet()

    if not os.path.exists(WEIGHTS_PATH):
        print(f"ERROR: Weights file not found: {WEIGHTS_PATH}")
        print("Run 'python examples/mnist_cnn.py' first to train the model.")
        sys.exit(1)

    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    size_mb = total_params * 4 / (1024 * 1024)
    print(f"   Parameters: {total_params:,}  ({size_mb:.2f} MB)")

    # =========================================================================
    # Step 2: Verify with example input
    # =========================================================================
    print(f"\n[2/4] Running PyTorch inference on example input {list(INPUT_SHAPE)}...")

    example_input = torch.randn(*INPUT_SHAPE)
    with torch.no_grad():
        pytorch_output = model(example_input)

    print(f"   Output shape: {list(pytorch_output.shape)}")
    print(f"   Output: {pytorch_output[0].numpy().round(4)}")

    # =========================================================================
    # Step 3: Compile to IR
    # =========================================================================
    print("\n[3/4] Compiling PyTorch model to IR...")

    ir_graph = compile_model(
        model=model,
        example_input=example_input,
        return_ir=True,
    )
    print(f"   IR nodes: {len(ir_graph.nodes)}")

    # =========================================================================
    # Step 4: Generate C code (float32, no quantization)
    # =========================================================================
    print(f"\n[4/4] Generating C code to {OUTPUT_DIR}/...")

    printer = CPrinter(ir_graph)
    printer.generate_all(OUTPUT_DIR)

    print(f"\n{'=' * 60}")
    print("SUCCESS! Generated files:")
    print(f"{'=' * 60}")
    for fname in ["model.h", "model.c", "weights.h", "nn_ops_float.h"]:
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(fpath):
            fsize = os.path.getsize(fpath) / 1024
            print(f"   {OUTPUT_DIR}/{fname:<20s} {fsize:>8.1f} KB")

    print(f"\n{'=' * 60}")
    print("HOW TO USE THE GENERATED C CODE:")
    print(f"{'=' * 60}")
    print(f"""
1. Include the header in your project:
   #include "model.h"

2. Prepare your input data (NHWC layout):
   - PyTorch input:  NCHW (1, 1, 28, 28)
   - C code expects: NHWC (28, 28, 1) = 784 floats
   - For grayscale MNIST the layouts are equivalent (C=1)

   float input_data[784];
   // Normalize: pixel_value / 255.0, then (x - 0.1307) / 0.3081

3. Allocate output buffer:
   float output[10];  // 10 digit classes

4. Run inference:
   model_forward(input_data, output);

5. Read prediction:
   int predicted_digit = argmax(output, 10);

6. Compile:
   gcc -O2 -o mnist_test main.c model.c -lm
""")

    # Print IR summary
    print(f"{'=' * 60}")
    print("IR GRAPH SUMMARY:")
    print(f"{'=' * 60}")
    print(ir_graph.print_graph())


if __name__ == "__main__":
    main()
