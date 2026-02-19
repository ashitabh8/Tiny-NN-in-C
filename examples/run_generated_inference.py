"""
Compile and run inference with generated C code (memory-optimized, block-scoped buffers).

Usage:
    python examples/run_generated_inference.py [OUTPUT_DIR]

Example (after running tiny_resnet.py):
    python examples/run_generated_inference.py tmp/tiny_resnet_for_embedded_device_float

This script:
  1. Compiles model.c + run_generated_main.c with gcc
  2. Runs the executable
  3. Exits with 0 on success, non-zero on compile/run error
"""

import os
import sys
import subprocess

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    out_dir = sys.argv[1] if len(sys.argv) > 1 else "tmp/tiny_resnet_for_embedded_device_float"
    out_dir = os.path.normpath(out_dir)
    if not os.path.isdir(out_dir):
        print(f"Error: output directory not found: {out_dir}")
        print("Run e.g. python examples/tiny_resnet.py first to generate C code.")
        sys.exit(1)

    model_c = os.path.join(out_dir, "model.c")
    if not os.path.isfile(model_c):
        print(f"Error: model.c not found in {out_dir}")
        sys.exit(1)

    main_c = os.path.join("examples", "run_generated_main.c")
    exe = os.path.join(out_dir, "run_inference")

    # Compile: gcc -O2 -I OUT_DIR -o exe OUT_DIR/model.c main.c -lm
    cmd = [
        "gcc", "-O2", "-I", out_dir,
        "-o", exe,
        model_c, main_c,
        "-lm"
    ]
    print("Compiling:", " ".join(cmd))
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print("Compilation failed.")
        sys.exit(ret.returncode)

    print("Running inference...")
    ret = subprocess.run([exe], cwd=project_root)
    if ret.returncode != 0:
        print("Run failed.")
        sys.exit(ret.returncode)

    print("Done. No errors.")

if __name__ == "__main__":
    main()
