# Tiny-NN-in-C

A source-to-source compiler that converts PyTorch `nn.Module` models into standalone, dependency-free C code targeting microcontrollers. Supports float32 and W8A8 (int8/int16) quantized inference. All generated C is header-only, portable, and uses zero dynamic allocation.

## Contents

- [Getting Started](#getting-started)
- [Design Philosophy](#design-philosophy)
- [Supported PyTorch Operations](#supported-pytorch-operations)
- [Quantization](#quantization)
- [Arduino Support](#arduino-support)
- [Verify Your Model](#verify-your-model)
- [Examples](#examples)
- [How to Extend](#how-to-extend)
- [Input Layout](#input-layout)
- [Testing](#testing)
- [License](#license)

## Getting Started

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Run an end-to-end example

```bash
python examples/01_float_mnist/run.py
```

### Verify generated C vs PyTorch

```bash
python -m tools.verify_model --model models/tiny_mlp.py:TinyMLP --input-shape 1,784 --num-samples 50
```

## Design Philosophy

Tiny-NN-in-C is designed as a modular compiler pipeline, not a one-off model converter. The core goal is to make optimization and code generation policies easy to swap without rewriting the system.

- **Policy over hardcoding**: quantization and instrumentation are expressed as rule + transform passes, so new schemes can be added by defining rules rather than editing the core compiler flow.
- **Composable graph rewrites**: transforms operate on an IR graph with explicit rewiring, pre/post node insertion, and validation. This makes independent passes easier to combine safely.
- **Node-local behavior**: each IR node (especially quantized nodes) owns how it emits C and what conversion nodes it needs, enabling flexible replacement at the operation level.
- **Backend replaceability**: code generation is separated from tracing/lowering logic, so different runtime targets (for example host C, Arduino-oriented output, or future backends) can be introduced with minimal front-end changes.
- **Extensible by construction**: extension points are first-class (new op nodes, new transforms, new passes), so the system scales by adding modules instead of patching monolithic code.

## Supported PyTorch Operations

| PyTorch module / function | Status |
|---------------------------|--------|
| `nn.Conv2d`               | float, int8, int16 |
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

See [docs/quantization.md](docs/quantization.md) for the full guide including dynamic quantization, mixed precision, and how to add custom rules.

## Arduino Support

Pass `arduino_mode=True` to `CPrinter` to generate an `.ino` sketch:

```python
CPrinter(ir_graph, arduino_mode=True).generate_all("my_sketch/")
```

The generated sketch includes `setup()`/`loop()`, profiling via `micros()`, and `Serial` output.

## Verify Your Model

Use the built-in verification tool to confirm compiled C numerically matches PyTorch.  
Verification is end-to-end: trace/lower the model, generate C, compile with `gcc`, run inference on random samples, and compare C vs PyTorch outputs with error metrics.

### Float32 verification (CLI)

```bash
python -m tools.verify_model \
  --model models/tiny_mlp.py:TinyMLP \
  --input-shape 1,784 \
  --num-samples 50
```

### Float32 verification (Python API)

```python
from tools.verify_model import verify_model

results = verify_model(model, example_input, num_samples=50)
print(results.summary())
```

### Int8 verification (Python API with quantization rules)

```python
from tools.verify_model import verify_model
from src.pytorch_to_c.quantization import StaticQuantRule

rules = [
    StaticQuantRule(
        pattern=r".*fc.*",
        dtype="int8",
        input_scale=0.05,
        input_offset=0,
        weight_scale=0.02,
        weight_offset=0,
        output_scale=0.05,
        output_offset=0,
    )
]

results = verify_model(
    model,
    example_input,
    num_samples=50,
    quantization_rules=rules,
    tolerance=5.0,  # quantized paths usually need looser tolerance
)
print(results.summary())
```

## Examples

Each example is a self-contained, end-to-end script: train (if needed), compile to C, and verify against PyTorch.

| Example | Description |
|---------|-------------|
| `examples/01_float_mnist/run.py` | Train MNIST CNN, compile to float C, verify |
| `examples/02_dynamic_quantization/per_tensor.py` | Dynamic per-tensor int8 quantization + verify |
| `examples/03_qat_resnet/run.py` | QAT training on TinyResNet1D, compile quantized C, verify |
| `examples/misc/profiling_example.py` | Profiling transform demo |
| `examples/misc/fuse_dequant_quant_demo.py` | FuseDequantQuantPass optimization demo |

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
