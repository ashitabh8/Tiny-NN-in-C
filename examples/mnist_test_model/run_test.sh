#!/usr/bin/env bash
# MNIST C model verification: extract -> compile C runner -> run -> compare
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DIR/../.." && pwd)"
GEN="$ROOT/tmp/mnist_cnn"

cd "$DIR"

if [[ ! -f "$GEN/model.c" ]]; then
  echo "ERROR: Generated C model not found at $GEN/"
  echo "Run: python examples/compile_mnist_cnn.py"
  exit 1
fi

echo "[1/4] Extract samples + PyTorch inference..."
python extract_samples.py

echo "[2/4] Compile C test runner..."
gcc -O2 -I"$GEN" -o test_runner main.c "$GEN/model.c" -lm

echo "[3/4] Run C model..."
./test_runner

echo "[4/4] Compare outputs..."
python compare.py "$@"
echo "Done."
