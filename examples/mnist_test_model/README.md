# MNIST C model verification

Compares PyTorch logits from `MNISTConvNet` against the generated C `model_forward` on the same MNIST test images.

## Prerequisites

- Trained weights: `examples/mnist_cnn_weights.pth` (`python examples/mnist_cnn.py`)
- Generated C model: `tmp/mnist_cnn/` (`python examples/compile_mnist_cnn.py`)

## Run

From this directory:

```bash
./run_test.sh
```

Or step by step:

1. `python extract_samples.py` — writes `test_inputs.h` and `pytorch_outputs.txt`
2. `gcc -O2 -I../../tmp/mnist_cnn -o test_runner main.c ../../tmp/mnist_cnn/model.c -lm`
3. `./test_runner` — writes `c_outputs.txt`
4. `python compare.py` — default tolerance `1e-3`

## Generated files (not committed)

- `test_inputs.h`, `pytorch_outputs.txt`, `c_outputs.txt`, `test_runner`

Float convolutions in `src/c_ops/nn_ops_float.h` use **PyTorch-style symmetric padding** (`pad_h`, `pad_w` per side), matching `torch.nn.Conv2d`.
