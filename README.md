# Tiny-NN-in-C

> **Alpha release — branch `release/public-v1`**
> This is an early public release. APIs, generated C interfaces, and quantization rule signatures may change between versions. See [Known Issues](#known-issues) for current limitations.

---

A source-to-source compiler that converts PyTorch `nn.Module` models into standalone, dependency-free C code targeting microcontrollers. Supports float32 and W8A8 (int8/int16) quantized inference. Generated C uses zero dynamic allocation and is portable across bare-metal targets.

> **Extended features** — calibration, per-group/int4 quantization, GPTQ, pruning, palettization, LQER error correction, and the Triton GPU backend are documented separately in [README_FEATURES.md](README_FEATURES.md).

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
- [Known Issues](#known-issues)
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
- **Backend replaceability**: code generation is separated from tracing/lowering logic, so different runtime targets (host C, Arduino-oriented output, or future backends) can be introduced with minimal front-end changes.
- **Extensible by construction**: extension points are first-class (new op nodes, new transforms, new passes), so the system scales by adding modules instead of patching monolithic code.

## Supported PyTorch Operations

The table below describes what the code generator actually emits. "float" means the operation runs on `float32` data in the generated C; "int8/int16" means the generated C calls the corresponding quantized kernel.

| PyTorch module / function | Generated C |
|---------------------------|-------------|
| `nn.Conv2d` (standard) | float, int8, int16 |
| `nn.Conv2d` (depthwise) | float, int8, int16 |
| `nn.Linear` | float, int8, int16 |
| `nn.ReLU` | float only ¹ |
| `nn.BatchNorm2d` | float |
| `nn.Softmax` | float ² |
| `nn.AdaptiveAvgPool2d((1,1))` | float ³ |
| `torch.add` / `+` | float |
| `torch.mul` / `*` | float |
| `tensor.view` / `flatten` / `reshape` | float, int8, int16 |
| `tensor.mean(dim=[2,3])` | float ³ |
| `tensor.mean(dim=-1)` | float ³ |
| `tensor.unsqueeze` / `squeeze` | float |
| `tensor.permute` | float |

**Footnotes**

¹ `relu_int8` / `relu_int16` C kernels are present in `nn_ops_int8.h` / `nn_ops_int16.h` but the code generator never emits calls to them. In the current quantization pipeline, ReLU always follows a `DequantizeNode` and therefore executes on `float32` data.

² The generated `softmax` always reduces over the full flat buffer. The `dim` argument passed to `nn.Softmax` is stored in the IR but ignored by the code generator. This is correct for the common case of a final logit layer but will produce wrong results for any other `dim` value.

³ C kernels for int8 variants of these ops exist (`global_average_pool_2d_int8`, `mean_hwc_int8`, etc.) but there is no quantization rule that produces int8-typed nodes of these op types. The int8 code paths in `c_printer.py` are currently unreachable through the public transform API and should be treated as float-only.

## Quantization

Apply int8 or int16 quantization using the rule + transform pattern.

### Static per-tensor (user-calibrated scales)

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

### Static per-channel (auto-calibrated weight scales)

```python
from src.pytorch_to_c.quantization import (
    StaticPerChannelConvQuantRule,
    StaticPerChannelLinearQuantRule,
    QuantizationTransform,
)

rules = [
    StaticPerChannelConvQuantRule(
        pattern=r'.*conv.*', dtype='int8',
        input_scale=0.05, input_offset=0,
        output_scale=0.05, output_offset=0,
    ),
    StaticPerChannelLinearQuantRule(
        pattern=r'.*fc.*', dtype='int8',
        input_scale=0.05, input_offset=0,
        output_scale=0.05, output_offset=0,
    ),
]
ir_graph = QuantizationTransform(rules).apply(ir_graph)
```

Weight scales are computed per output-channel from `absmax / q_max` at compile time.

### Dynamic per-tensor (runtime activation scales)

```python
from src.pytorch_to_c.quantization import DynamicQuantRuleMinMaxPerTensor, QuantizationTransform

rules = [
    DynamicQuantRuleMinMaxPerTensor(pattern=r'.*', dtype='int8'),
]
ir_graph = QuantizationTransform(rules).apply(ir_graph)
```

Weight scale is computed from the weight tensor at compile time. Activation scale is computed from the input tensor at runtime using `compute_dynamic_scale_int8`. Output is `float32` directly — no requantize/dequantize round-trip.

### Available quantization rules

| Rule class | Activation scale | Weight scale | Output |
|------------|-----------------|--------------|--------|
| `StaticQuantRule` | user-provided | user-provided (per-tensor) | int8/int16 requantized |
| `StaticPerChannelConvQuantRule` | user-provided | auto per-output-channel | int8/int16 requantized |
| `StaticPerChannelLinearQuantRule` | user-provided | auto per-output-column | int8/int16 requantized |
| `DynamicQuantRuleMinMaxPerTensor` | runtime min-max | auto per-tensor | float32 direct |

See [docs/quantization.md](docs/quantization.md) for the full guide including mixed precision and how to add custom rules.

## Arduino Support

Pass `arduino_mode=True` to `CPrinter` to generate an `.ino` sketch alongside the C files:

```python
CPrinter(ir_graph, arduino_mode=True).generate_all("my_sketch/")
```

The generated sketch includes `setup()`/`loop()`, profiling via `micros()`, and `Serial` output. The folder name must match the `.ino` filename (Arduino requirement).

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

### Quantized verification (Python API)

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

Each example is a self-contained, end-to-end script.

| Example | Description |
|---------|-------------|
| `examples/01_float_mnist/run.py` | Train MNIST CNN, compile to float C, verify against PyTorch |
| `examples/02_dynamic_quantization/per_tensor.py` | Dynamic per-tensor int8 quantization on MNIST CNN, verify |
| `examples/02_dynamic_quantization/per_channel_asymmetric.py` | Static per-channel quantization with non-zero zero-points, verify |
| `examples/03_qat_resnet/run.py` | QAT training on TinyResNet1D, compile quantized C, verify |
| `examples/04_dynamic_depthwise/run.py` | Depthwise-separable CNN with dynamic int8 quantization, verify |
| `examples/misc/profiling_example.py` | Profiling transform demo |
| `examples/misc/fuse_dequant_quant_demo.py` | `FuseDequantQuantPass` optimization demo |

## How to Extend

- **New float op**: add lowering in `lower.py`, codegen handler in `c_printer.py`, C kernel in `nn_ops_float.h`
- **New quantized op**: subclass `QuantIRNode`, implement `generate_c_code()`, add C kernels to `nn_ops_int8.h` / `nn_ops_int16.h`
- **New IR pass**: subclass `IRPass`, implement `apply(ir_graph) -> IRGraph`
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

## Known Issues

### `nn.ReLU` always runs as float in generated C

`relu_int8` and `relu_int16` exist in the C headers but the code generator never emits calls to them. ReLU is always executed after a `DequantizeNode`, so it operates on `float32`. If a future change places ReLU between two quantized ops (without an intervening dequantize), the codegen would silently call the float `relu()` on an integer buffer.

### `nn.Softmax`: `dim` argument is ignored

The generated C always applies softmax over the entire flat output buffer. For the standard use case (softmax over logits at the final layer) this is correct. Any other `dim` value produces wrong results with no error.

### Generated `model_forward` is not thread-safe

Intermediate activation buffers (`slot_0`, `slot_1`, …) are declared `static` inside `model_forward`. This prevents stack overflow on constrained devices but means the function cannot be called concurrently from multiple threads or from an interrupt handler. All calls share the same buffers. For single-threaded bare-metal use this is not a concern.

### `AdaptiveAvgPool2d` and `tensor.mean` int8 paths are unreachable

The code generator has int8 branches for these ops, but no quantization rule in the public API can produce int8-typed nodes of these op types. They should be treated as float-only until a matching `QuantIRNode` subclass and rule are added.

### Asymmetric quantization: `mean_hwc_int8` ignores input zero-point

`mean_hwc_int8` and `mean_last_dim_int8` in `nn_ops_int8.h` dequantize with a hardcoded zero-point of 0. There is no `input_zp` parameter. With symmetric quantization (the only mode currently supported) this is correct.

### Per-channel quantization uses a single `weight_zp` for affine correction

The per-channel kernels (`dense_int8_per_channel`, `conv2d_nhwc_int8_per_channel`, etc.) apply per-channel weight *scales* but a single shared `weight_zp` in the affine correction term. With symmetric weight quantization (`weight_zp = 0`) the correction term is zero and this is correct. Passing a non-zero `weight_zp` with a per-channel rule would produce wrong results.

## License

MIT
