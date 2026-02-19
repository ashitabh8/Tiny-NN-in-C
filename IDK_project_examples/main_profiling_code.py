"""
Run profiling for a generated model directory.

Usage:
    python IDK_project_examples/main_profiling_code.py <dir> <nchw_input_shape> <output_shape>

Arguments:
    dir               Path to generated code directory (contains model.c, model.h, etc.)
    nchw_input_shape  Input shape in NCHW order, comma-separated  e.g. 1,6,7,256
    output_shape      Output shape, comma-separated                e.g. 1,10

Example:
    python IDK_project_examples/main_profiling_code.py \\
        IDK_project_examples/generated_code/resnet_1_profiling_code \\
        1,6,7,256 \\
        1,10

The script:
  1. Generates a temporary C runner with a random NHWC input
  2. Compiles model.c + runner with gcc
  3. Runs the binary  (profiling printouts come from model_forward itself)
  4. Deletes the temp files
"""

import sys
import os
import subprocess
import tempfile
import math


def parse_shape(s: str):
    return [int(x) for x in s.strip().split(",")]


def nhwc_size(nchw: list) -> int:
    """Total floats for a flat NHWC buffer given an NCHW shape [N,C,H,W]."""
    n, c, h, w = nchw
    return n * h * w * c


def generate_runner_c(input_size: int, output_size: int) -> str:
    return f"""\
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "model.h"

#define INPUT_SIZE  {input_size}
#define OUTPUT_SIZE {output_size}

int main(void) {{
    srand((unsigned int)time(NULL));

    float* input  = (float*)malloc(INPUT_SIZE  * sizeof(float));
    float* output = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    if (!input || !output) {{
        fprintf(stderr, "malloc failed\\n");
        return 1;
    }}

    for (int i = 0; i < INPUT_SIZE; ++i)
        input[i] = 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f;

    model_forward(input, output);

    free(input);
    free(output);
    return 0;
}}
"""


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    gen_dir       = sys.argv[1]
    nchw_shape    = parse_shape(sys.argv[2])
    output_shape  = parse_shape(sys.argv[3])

    if len(nchw_shape) != 4:
        print(f"Error: input shape must have 4 dimensions (N,C,H,W), got {nchw_shape}")
        sys.exit(1)

    input_size  = nhwc_size(nchw_shape)
    output_size = math.prod(output_shape)

    model_c = os.path.join(gen_dir, "model.c")
    if not os.path.exists(model_c):
        print(f"Error: {model_c} not found")
        sys.exit(1)

    print(f"Directory : {gen_dir}")
    print(f"Input     : NCHW {nchw_shape}  →  NHWC flat size = {input_size}")
    print(f"Output    : {output_shape}  →  flat size = {output_size}")
    print()

    runner_src = generate_runner_c(input_size, output_size)

    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
        runner_c_path = f.name
        f.write(runner_src)

    binary_path = runner_c_path.replace(".c", "")

    try:
        compile_cmd = [
            "gcc", "-O2",
            "-I", gen_dir,
            "-o", binary_path,
            model_c,
            runner_c_path,
            "-lm",
        ]
        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("Compilation failed:")
            print(result.stderr)
            sys.exit(1)

        run_result = subprocess.run([binary_path], capture_output=True, text=True)
        if run_result.returncode != 0:
            print("Runtime error:")
            print(run_result.stderr)
            sys.exit(1)

        print(run_result.stdout, end="")

    finally:
        if os.path.exists(runner_c_path):
            os.remove(runner_c_path)
        if os.path.exists(binary_path):
            os.remove(binary_path)


if __name__ == "__main__":
    main()
