# Tiny-NN-in-C — Extended Features

This document covers the features added on top of the base compiler described in [README.md](README.md): the ML compression suite (per-group quantization, int4, GPTQ, pruning, palettization), calibration infrastructure, LQER error correction, and the Triton GPU backend.

Everything follows the two extension patterns from the base design:

- **Quantization formats** = rule + node + kernel triplet (`rules.py` + `ops/` + `nn_ops_*.h` / `triton_ops/`)
- **Structural transforms** = `IRPass` subclasses in `src/passes/` applied to the float IR before quantization

## Contents

- [Calibration](#calibration)
- [Per-Group Quantization](#per-group-quantization)
- [Int4 Per-Group Quantization (W4A8)](#int4-per-group-quantization-w4a8)
- [GPTQ Error-Compensated Rounding](#gptq-error-compensated-rounding)
- [Weight Palettization](#weight-palettization)
- [Structured Pruning Pass](#structured-pruning-pass)
- [LQER: Low-Rank Quantization Error Reconstruction](#lqer-low-rank-quantization-error-reconstruction)
- [Triton GPU Backend](#triton-gpu-backend)
- [Tools Summary](#tools-summary)

---

## Calibration

`src/pytorch_to_c/calibration/` replaces hand-tuned activation scales with measured ones. Hooks are attached to `Linear` / `Conv1d` / `Conv2d` modules by name (module names match IR node names from FX tracing).

```python
from src.pytorch_to_c.calibration import calibrate, make_static_rules

stats = calibrate(
    model,
    calibration_batches,        # iterator or list of input tensors
    collect_hessian=True,       # accumulate X^T X per layer (needed for GPTQ)
    max_batches=32,
)

rules = make_static_rules(
    stats,
    dtype="int8",
    granularity="per_channel",  # "per_tensor" | "per_channel" | "per_group"
)
```

`CalibrationStats` tracks min/max, percentile (99.9%), and MSE-informed ranges per layer (`ActivationObserver`), plus optional per-layer Hessian estimates (`stats.hessians`) consumed by GPTQ.

## Per-Group Quantization

`StaticPerGroupLinearQuantRule` groups linear weights along the **input (reduction) axis**, one scale per `[group, out_column]`. Group sizes are tile-aligned to `BLOCK_K` (32) so group boundaries coincide with kernel tiles.

```python
from src.pytorch_to_c.quantization import StaticPerGroupLinearQuantRule

rule = StaticPerGroupLinearQuantRule(
    pattern=r".*fc.*", dtype="int8",
    input_scale=0.05, input_offset=0,
    output_scale=0.05, output_offset=0,
    group_size="auto",        # or an int (multiple of 32)
    error_budget=1e-3,        # with "auto": largest G whose weight MSE stays under budget
)
```

Kernels: `dense_int8_per_group` / `dense_int16_per_group` in C (`nn_ops_int8.h` / `nn_ops_int16.h`) and Triton. Scales are stored as `{weight_name}_per_group_scales`; the chosen group size lands in `metadata['group_size']`.

## Int4 Per-Group Quantization (W4A8)

`StaticInt4PerGroupLinearQuantRule` / `DynamicInt4PerGroupLinearQuantRule` quantize
weights to 4 bits (per-group symmetric, range [-8, 7], two nibbles packed per byte)
while **activations are int8**, matching the same QuantizeNode / matmul /
DequantizeNode graph pattern as int8 static/dynamic quantization.

```python
from src.pytorch_to_c.calibration import calibrate
from src.pytorch_to_c.quantization import (
    StaticInt4PerGroupLinearQuantRule,
    DynamicInt4PerGroupLinearQuantRule,
)

stats = calibrate(model, batches)
static_rule = StaticInt4PerGroupLinearQuantRule(
    pattern=r".*fc.*",
    input_scale=stats.get_input_scale("fc1"),
    input_offset=0,
    output_scale=stats.get_output_scale("fc1"),
    output_offset=0,
    group_size=64,
)
# Or dynamic (runtime activation scale, float output, no DequantizeNode):
dyn_rule = DynamicInt4PerGroupLinearQuantRule(pattern=r".*fc.*", group_size=64)
```

Kernels: `dense_int8_int4w_per_group` (static → int8) and
`dense_int8_int4w_per_group_to_float` (dynamic → float) in `src/c_ops/nn_ops_int4.h`
(and matching Triton reference kernels). Packed weights are plain `int8` arrays in
`weights.h` / `weights.npz`. Roughly 8x weight compression vs float32; accuracy
depends on activation calibration.

## GPTQ Error-Compensated Rounding

Compile-time only — output format is identical to round-to-nearest, so **no kernel or codegen changes**. Column-sequential quantization redistributes rounding error using the damped Hessian (`X^T X + lambda*I`, Cholesky-based) from calibration.

```python
stats = calibrate(model, batches, collect_hessian=True)

rule = StaticPerGroupLinearQuantRule(
    ..., rounding="gptq", calibration=stats,
)
# also available on StaticInt4PerGroupLinearQuantRule / DynamicInt4PerGroupLinearQuantRule
```

Direct API: `from src.pytorch_to_c.quantization import gptq_quantize`.

## Weight Palettization

`PaletteWeightRule` clusters weights with 1-D k-means into a small codebook; the weight tensor is replaced by packed indices (4-bit when `num_centroids <= 16`, else uint8) plus a float codebook. Weight-only: activations stay float32.

```python
from src.pytorch_to_c.quantization import PaletteWeightRule

rule = PaletteWeightRule(pattern=r".*fc.*", num_centroids=16)
```

Kernel: `dense_float_palettized` (LUT lookup, then float MAC). Primary win is flash/weight size on MCU targets.

## Structured Pruning Pass

`StructuredPruningPass` removes whole output columns (linear) or output channels (conv) and propagates the shape change into downstream consumers (next layer's input rows / `in_c`, BN params, `output_shape` along the chain).

```python
from src.passes import StructuredPruningPass, PruneRule

pruning = StructuredPruningPass([
    PruneRule(pattern=r".*fc1.*", amount=0.5, criterion="l2"),   # or "activation"
])
ir_graph = pruning.apply(ir_graph)
```

Guards: never prunes graph output layers; raises a clear error at residual-add junctions (feed-forward chains only). The `"activation"` criterion consumes calibration stats.

## LQER: Low-Rank Quantization Error Reconstruction

LQER corrects a quantized layer's output with a low-rank approximation of its weight-quantization error, implemented **entirely as IR nodes** (`src/pytorch_to_c/quantization/ops/quant_LQER.py`) — not as a pass. You supply the full error matrix `E` and a rank; the node factorizes `E ~= A @ B` at compile time.

```text
x (float32) --+--> quantize --> int8 matmul --> [dequant] --+
              |                                             v
              +--> x @ A --> @ B  (float, rank r) -----> add --> users
```

The correction branch taps the **original float input**, so it is topologically independent of the quantized path — the C backend emits the two chains as OpenMP sections that run concurrently.

```python
import numpy as np
from src.pytorch_to_c.quantization import LQERDynamicQuantRule, QuantizationTransform

# 1. Compute the error matrix (user responsibility) in the compiler's stored
#    weight layout: linear [in_features, out_features].
ir = compile_model(model, example_input, return_ir=True)
W = ir.parameters["fc1_weight"]
scale = np.abs(W).max() / 127.0
E = W - np.clip(np.round(W / scale), -128, 127) * scale

# 2. Quantize with correction.
rules = [LQERDynamicQuantRule("fc1", "int8", error_matrix=E, rank=8)]
ir = compile_model(model, example_input, return_ir=True)
ir = QuantizationTransform(rules).apply(ir)
```

Key points:

- **Rules**: `LQERDynamicQuantRule(pattern, dtype, error_matrix, rank, factorizer=None)` and `LQERStaticQuantRule(...)` (static joins after the `DequantizeNode`). `error_matrix` is a single array, or a `{node_name: array}` dict when the pattern matches several layers.
- **Layers**: linear and conv2d, dynamic and static. Conv error layout is HWIO `[kh, kw, in_c, out_c]` (or pre-flattened `[kh*kw*in_c, out_c]`); the branch becomes a float conv with `r` output channels followed by a 1x1 conv — no new kernels needed. Grouped/depthwise conv is not supported.
- **Pluggable factorization**: the default is `svd_factorizer(E, rank) -> (A, B)`. Pass any callable with that signature (PCA, randomized SVD, ...) via `factorizer=`. Swapping the method touches nothing outside the node.
- **OpenMP**: generated C wraps the two chains in `#pragma omp parallel sections`. Build with `-fopenmp` to enable (e.g. `verify_model(..., openmp=True)`); without the flag the pragmas are ignored and the code runs sequentially with identical numerics.
- **Memory safety**: the planner (`codegen/memory_planner.py` + `codegen/parallel_regions.py`) keeps every buffer inside a parallel region live until the join, so the concurrent chains never alias an activation slot.
- **Extension point**: the diamond wiring is generic. Any `QuantIRNode` can declare a side branch by overriding `get_parallel_branch(ir_graph) -> (branch_nodes, join_node)`; the transform wires it without knowing what the branch computes.

## Triton GPU Backend

The compiler can target Triton instead of C:

```python
compile_model(model, example_input, "generated_triton/", backend="triton")
```

Generates `model.py` (calling kernels from `src/triton_ops/nn_ops_float.py` / `nn_ops_quant.py`) plus `weights.npz`. All quantization rules and passes above work identically on this backend, LQER included (emitted sequentially — OpenMP applies to the C backend only).

Verify and benchmark on GPU:

```python
from tools.verify_model_triton import verify_model_triton
results = verify_model_triton(model, example_input, num_samples=10, device="cuda")
```

```bash
python -m tools.benchmark_triton
```

## Tools Summary

| Tool | Purpose |
|------|---------|
| `tools/verify_model.py` | PyTorch vs generated C comparison; supports `passes=`, `quantization_rules=`, and `openmp=True` for LQER parallel regions |
| `tools/verify_model_triton.py` | PyTorch vs generated Triton comparison (CUDA) |
| `tools/evaluate_compression.py` | Weight-size / error / top-1 comparison table across compression schemes |
| `tools/benchmark_triton.py` | Triton kernel benchmarks |

Composition example — pruning, then quantization with LQER, verified end-to-end:

```python
from tools.verify_model import verify_model
from src.passes import StructuredPruningPass, PruneRule
from src.pytorch_to_c.quantization import LQERDynamicQuantRule

results = verify_model(
    model, example_input, num_samples=50,
    passes=[
        StructuredPruningPass([PruneRule(r".*fc1.*", amount=0.25)]),
    ],
    quantization_rules=[LQERDynamicQuantRule(r".*fc3.*", "int8",
                                             error_matrix=E, rank=8)],
    openmp=True,
)
print(results.summary())
```

Structural passes run on the float IR first, then quantization rules, then codegen — the same order `verify_model` applies internally.
