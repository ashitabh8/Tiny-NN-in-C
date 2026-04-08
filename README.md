# Tiny-NN-in-C

A source-to-source compiler that converts PyTorch `nn.Module` models into standalone, dependency-free C code targeting microcontrollers. Supports float32 and W8A8 (int8/int16) quantized inference. All generated C is header-only, portable, and uses zero dynamic allocation.

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Compile a model

```python
import torch
from src.pytorch_to_c.compiler import compile_model
from models import TinyMLP

model = TinyMLP()
model.eval()
example_input = torch.randn(1, 784)
compile_model(model, example_input, output_dir="output/")
```

The generated `output/` directory is self-contained:

```
output/
  model.h         # void model_forward(const float* input, float* output);
  model.c         # implementation (slot-based buffer reuse)
  weights.h       # static const arrays
  nn_ops_float.h  # header-only C runtime kernels
```

### Use the generated C code

```c
#include "model.h"

float input[784];
float output[10];

// fill input ...
model_forward(input, output);
```

Compile for your target:

```bash
gcc -O2 -o model_test main.c model.c -lm              # host testing
arm-none-eabi-gcc -mcpu=cortex-m4 -O2 -c model.c -o model.o  # ARM Cortex-M
```

## Supported PyTorch Operations

| PyTorch module / function | Status |
|---------------------------|--------|
| `nn.Conv2d`               | float, int8, int16, int8 per-channel |
| `nn.Conv2d` (depthwise)   | float, int8 per-channel |
| `nn.Linear`               | float, int8, int16 |
| `nn.ReLU`                 | float, int8, int16 |
| `nn.BatchNorm2d`          | float |
| `nn.Softmax`              | float |
| `nn.AdaptiveAvgPool2d`    | float, int8 |
| `torch.add` / `+`         | float |
| `torch.mul` / `*`         | float |
| `tensor.view` / `flatten` / `reshape` | float, int8 |
| `tensor.mean(dim=...)`    | float, int8 (spatial + last dim) |
| `tensor.unsqueeze` / `squeeze` | float |
| `tensor.permute`          | float |

## Quantization

Apply int8 or int16 quantization using the rule + transform pattern:

```python
from src.pytorch_to_c.quantization import StaticQuantRule, QuantizationTransform
from src.pytorch_to_c.codegen.c_printer import CPrinter

ir_graph = compile_model(model, example_input, return_ir=True)

rules = [
    StaticQuantRule(pattern=r'.*conv.*', dtype='int8',
                    input_scale=0.05, input_offset=0,
                    weight_scale=0.02, weight_offset=0,
                    output_scale=0.05, output_offset=0),
]
ir_graph = QuantizationTransform(rules).apply(ir_graph)
CPrinter(ir_graph).generate_all("output_quant/")
```

For depthwise-separable blocks, per-channel rules support per-channel weight scales:

```python
from src.pytorch_to_c.quantization import StaticDepthwiseConvRule, StaticPointwiseConvRule
```

See [docs/quantization.md](docs/quantization.md) for the full guide including dynamic quantization, per-channel quantization, mixed precision, and how to add custom rules.

## Arduino Support

Pass `arduino_mode=True` to `CPrinter` to generate an `.ino` sketch:

```python
CPrinter(ir_graph, arduino_mode=True).generate_all("my_sketch/")
```

The generated sketch includes `setup()`/`loop()`, profiling via `micros()`, and `Serial` output.

## Verify Your Model

Use the built-in verification tool to confirm the C output matches PyTorch:

```bash
python -m tools.verify_model \
  --model models/tiny_mlp.py:TinyMLP \
  --input-shape 1,784 \
  --num-samples 50
```

Or from Python:

```python
from tools.verify_model import verify_model

results = verify_model(model, example_input, num_samples=50)
print(results.summary())
```

## Examples

| Example | Description |
|---------|-------------|
| `examples/tiny_mlp.py` | Simplest: MLP to float C |
| `examples/tiny_resnet.py` | ResNet1D with static int8 quantization |
| `examples/tiny_mixed_net.py` | Conv + Linear + Softmax |
| `examples/quantized_mlp.py` | Fine-grained per-layer quantization |
| `examples/dynamic_quantization.py` | Dynamic min-max per-tensor |
| `examples/example_op_no_op.py` | FuseDequantQuantPass optimization demo |
| `examples/profiling_example.py` | Profiling transform demo |
| `examples/mnist_cnn.py` | MNIST training + compilation |

## How to Extend

- **New float op**: add lowering in `lower.py`, codegen in `c_printer.py`, C kernel in `nn_ops_float.h`
- **New quantized op**: subclass `QuantIRNode`, implement `generate_c_code()`, add C kernel to `nn_ops_int8.h`
- **New IR pass**: subclass `IRPass`, implement `apply(ir_graph)`
- **New transform**: follow the `profiling/` module pattern (rule + matcher + transform + ops)

See [docs/quantization.md](docs/quantization.md) for details.

## Input Layout

The generated C code uses **NHWC** (channels-last). PyTorch uses **NCHW** (channels-first). Convert before calling `model_forward()`:

```python
nhwc_input = pytorch_input.permute(0, 2, 3, 1).numpy().flatten()
```

## Testing

```bash
pytest test/ -v                        # all tests
pytest test/test_verify_harness.py -v  # verification harness (requires gcc)
```

## License

MIT
