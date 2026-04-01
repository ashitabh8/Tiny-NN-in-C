# Tiny-NN-in-C: Project Context for LLMs

A source-to-source compiler that converts PyTorch `nn.Module` models into standalone, dependency-free C code targeting microcontrollers. Supports float32 and W8A8 (int8/int16) quantized inference. All generated C is header-only, portable, and uses zero dynamic allocation.

---

## 1. Project Structure

```
Tiny-NN-in-C/
├── src/
│   ├── pytorch_to_c/              # Main compiler pipeline (Python)
│   │   ├── __init__.py            # Package, version 0.1.0
│   │   ├── compiler.py            # Entry point: PyTorchToCCompiler, compile_model()
│   │   ├── frontend/
│   │   │   └── fx_tracer.py       # torch.fx symbolic tracing
│   │   ├── ir/
│   │   │   ├── node.py            # IRNode (double-linked)
│   │   │   ├── graph.py           # IRGraph (nodes + parameters + topo sort)
│   │   │   └── quant_node.py      # QuantIRNode (base for quantized ops)
│   │   ├── lowering/
│   │   │   └── lower.py           # FX graph → IR: weight extraction, shape inference
│   │   ├── codegen/
│   │   │   ├── c_printer.py       # IR → C code: model.c, model.h, weights.h
│   │   │   └── ops_map.py         # IR op_type → C function name mapping
│   │   ├── quantization/
│   │   │   ├── rules.py           # QuantRule, StaticQuantRule, DynamicQuantRuleMinMaxPerTensor
│   │   │   ├── rule_matcher.py    # First-match rule engine
│   │   │   ├── graph_transform.py # QuantizationTransform: apply rules to IR
│   │   │   └── ops/
│   │   │       ├── quant_utils.py # QuantizeNode, DynamicQuantizeInputNode, DequantizeNode
│   │   │       ├── quant_linear.py   # StaticQuantLinearNode, DynamicQuantLinearNode
│   │   │       └── quant_conv2d.py   # StaticQuantConv2dNode, DynamicQuantConv2dNode
│   │   └── profiling/
│   │       ├── rules.py           # ProfilingRule (regex match → timing wrapper)
│   │       ├── rule_matcher.py    # ProfilingRuleMatcher
│   │       ├── graph_transform.py # ProfilingTransform
│   │       └── ops/
│   │           └── profiling_utils.py # ProfilingWrapperNode (clock/printf or Arduino micros)
│   ├── c_ops/                     # Header-only C runtime kernels
│   │   ├── nn_ops_float.h         # Float: conv2d_nhwc, dense, relu, batchnorm2d_nhwc, softmax, mean_hwc, global_average_pool_2d
│   │   ├── nn_ops_int8.h          # Int8: quantize/dequantize, dense_int8, conv2d_nhwc_int8, relu_int8, compute_dynamic_scale_int8
│   │   └── nn_ops_int16.h         # Int16: same structure as int8
│   └── passes/
│       ├── base.py                # IRPass abstract base class
│       └── fuse_dequant_quant.py  # FuseDequantQuantPass: eliminate redundant dequant→quant pairs
├── examples/
│   ├── tiny_resnet.py             # End-to-end: ResNet1D → quantized C code
│   └── profiling_example.py       # Profiling transform demo
├── test/
└── requirements.txt
```

---

## 2. Compilation Pipeline

```
PyTorch nn.Module + example_input
         │
         ▼
  ┌─────────────────┐
  │  1. FX Tracing   │  FXTracer.trace_model() → fx.GraphModule
  │  (frontend/)     │  Uses torch.fx.symbolic_trace(), validates with allclose
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │  2. Lowering     │  Lowering.lower_fx_graph() → IRGraph
  │  (lowering/)     │  Maps FX nodes to IRNodes, extracts weights to numpy,
  │                  │  transposes Conv2d weights to HWIO, Linear to [in, out],
  │                  │  runs ShapeProp (or manual fallback) for shape inference
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │  3. Transforms   │  Optional: QuantizationTransform, ProfilingTransform,
  │  (user-applied)  │  or custom passes (FuseDequantQuantPass)
  │                  │  Applied by user between lowering and codegen
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │  4. Code Gen     │  CPrinter.generate_all() →
  │  (codegen/)      │    model.h: void model_forward(const float* input, float* output);
  │                  │    model.c: implementation with slot-based buffer reuse
  │                  │    weights.h: static const arrays (float, int8_t, int16_t)
  │                  │    nn_ops_*.h: copied C runtime headers
  └─────────────────┘
```

### Typical Usage Pattern

```python
from src.pytorch_to_c.compiler import compile_model
from src.pytorch_to_c.codegen.c_printer import CPrinter

# Get IR only (step 1+2)
ir_graph = compile_model(model, example_input, return_ir=True)

# Apply transforms (step 3) - quantization, profiling, passes
ir_graph = SomeTransform(rules).apply(ir_graph)

# Generate C code (step 4)
CPrinter(ir_graph).generate_all("output_dir/")
```

---

## 3. IR (Intermediate Representation)

### IRNode (`src/pytorch_to_c/ir/node.py`)

The fundamental unit of computation. Double-linked for bidirectional graph traversal.

**Key fields:**
- `name: str` - unique identifier (from PyTorch layer name)
- `op_type: str` - operation kind: `'input'`, `'conv2d'`, `'linear'`, `'relu'`, `'batchnorm'`, `'softmax'`, `'add'`, `'method_mean'`, `'method_flatten'`, `'method_view'`, `'method_size'`, `'adaptive_avg_pool'`, `'quantize'`, `'dequantize'`, `'dynamic_quantize'`
- `output_shape: tuple` - inferred from ShapeProp (NCHW from PyTorch)
- `dtype: str` - `'float32'`, `'int8'`, or `'int16'`
- `metadata: dict` - operation-specific data (weight names, kernel size, stride, padding, etc.)
- `inputs: List[IRNode]` - nodes this depends on (predecessors)
- `users: List[IRNode]` - nodes that depend on this (successors)

**Key methods:**
- `add_input(node)` / `add_user(node)` - auto-updates both sides of the link
- `remove_input(node)` / `remove_user(node)` - auto-updates both sides
- `get_c_dtype()` - maps `dtype` to C type string: `'float32'→'float'`, `'int8'→'int8_t'`, `'int16'→'int16_t'`
- `is_quantized` - property, True if `dtype != 'float32'`
- `validate_input_dtypes()` - override in subclasses for dtype checking

### IRGraph (`src/pytorch_to_c/ir/graph.py`)

Container holding the full computation graph.

**Key fields:**
- `nodes: List[IRNode]` - all nodes in topological order
- `inputs: List[IRNode]` - graph input placeholder nodes
- `outputs: List[IRNode]` - graph output nodes
- `parameters: Dict[str, np.ndarray]` - weight storage (name → numpy array)

**Key methods:**
- `add_node(node)` - adds to graph, enforces unique names
- `add_parameter(name, data)` - stores numpy array (weights, biases, BN stats)
- `topological_sort()` - cycle-detecting topological sort
- `validate()` - checks for cycles and dangling references
- `print_graph()` - human-readable dump

### QuantIRNode (`src/pytorch_to_c/ir/quant_node.py`)

Abstract base class for quantized operations. Extends IRNode.

**Key fields (beyond IRNode):**
- `scale: float` - quantization scale (for weights)
- `offset: int` - zero point
- `quant_strategy: str` - `'static'` or `'dynamic'`

**Key methods:**
- `generate_c_code(c_printer) -> List[str]` - **abstract** - each subclass emits its own C code
- `get_pre_nodes() -> List[IRNode]` - nodes to insert BEFORE (e.g., QuantizeNode)
- `get_post_nodes() -> List[IRNode]` - nodes to insert AFTER (e.g., DequantizeNode)
- `validate_input_dtypes()` - expects quantized inputs

---

## 4. Concept: Rules and Transforms

The compiler uses a **rule + transform** pattern for graph modifications. This is the primary extension mechanism.

### How Rules Work

1. **A Rule** defines two things: (a) **which nodes to match** via regex on `node.name`, and (b) **what to do** when matched (create a replacement node).
2. **A RuleMatcher** iterates over the IR graph and finds nodes matching each rule. First-match-wins ordering.
3. **A Transform** applies the matched rules to the graph: replaces nodes, inserts pre/post conversion nodes, rewires graph edges.

### Quantization Rules (`src/pytorch_to_c/quantization/rules.py`)

```
QuantRule (ABC)
├── StaticQuantRule      - user provides input_scale, weight_scale, output_scale, offsets
└── DynamicQuantRuleMinMaxPerTensor - weight_scale computed from weight stats at compile time,
                                      input_scale computed at runtime
```

**`StaticQuantRule`** constructor:
```python
StaticQuantRule(
    pattern=r'.*conv.*',     # regex matching node names
    dtype='int8',            # target dtype
    input_scale=0.05,        # scale for input activation quantization
    input_offset=0,
    weight_scale=0.02,       # scale for weight quantization
    weight_offset=0,
    output_scale=0.05,       # scale for output dequantization
    output_offset=0
)
```

When matched, calls `create_quant_node(node)` which returns the appropriate `QuantIRNode` subclass (e.g., `StaticQuantConv2dNode`).

### QuantizationTransform (`src/pytorch_to_c/quantization/graph_transform.py`)

`QuantizationTransform(rules).apply(ir_graph)` does:

1. **Find** nodes matching rules via `RuleMatcher`
2. **Replace** each matched float node with its `QuantIRNode` (via `rule.create_quant_node()`)
3. **Insert pre/post nodes** - each `QuantIRNode` declares what conversions it needs:
   - `get_pre_nodes()` → e.g., `QuantizeNode` (static) or `DynamicQuantizeInputNode` (dynamic)
   - `get_post_nodes()` → e.g., `DequantizeNode` to return to float32
4. **Validate** float32 output (C API requires `model_forward(float*, float*)`)
5. **Quantize weights** at compile time: `rule.quantize_weights(float_weights) → int8 numpy`
6. **Validate** dtype compatibility across all edges

### Profiling Rules (`src/pytorch_to_c/profiling/rules.py`)

Same pattern. `ProfilingRule(pattern, label)` matches nodes by name, wraps them in `ProfilingWrapperNode` which emits the original op's C code plus timing instrumentation.

```python
ProfilingTransform([
    ProfilingRule(r"conv", label="conv"),
    ProfilingRule(r"fc", label="fc"),
]).apply(ir_graph)
```

---

## 5. Quantized Op Nodes

Each quantized op is a `QuantIRNode` subclass that knows how to emit its own C code.

### Node Hierarchy

```
IRNode
├── QuantizeNode          - float32 → intX (static scale)
├── DynamicQuantizeInputNode - float32 → intX (runtime scale via compute_dynamic_scale)
├── DequantizeNode        - intX → float32
└── QuantIRNode (ABC)
    ├── StaticQuantLinearNode   - dense_int8/int16 with user-provided scales
    ├── DynamicQuantLinearNode  - dense_int8/int16 with runtime input scale
    ├── StaticQuantConv2dNode   - conv2d_nhwc_int8/int16 with user-provided scales
    └── DynamicQuantConv2dNode  - conv2d_nhwc_int8/int16 with runtime input scale
```

### What Each Node Does

**`QuantizeNode`**: Emits `quantize_float_to_int8(input, size, scale, offset, output);`

**`DynamicQuantizeInputNode`**: Emits `float scale_xxx = compute_dynamic_scale_int8(input, size);` then `quantize_float_to_int8(input, size, scale_xxx, 0, output);`

**`DequantizeNode`**: Emits `dequantize_int8_to_float(input, size, scale, offset, output);`

**`StaticQuantLinearNode`**: Emits `dense_int8(input, in_features, weight, bias, out_features, input_scale, weight_scale, offset, output);`. Its `get_pre_nodes()` returns `[QuantizeNode]` and `get_post_nodes()` returns `[DequantizeNode]`.

**`StaticQuantConv2dNode`**: Same pattern with `conv2d_nhwc_int8(...)`.

**Dynamic variants** use `DynamicQuantizeInputNode` as pre-node (computes scale at runtime) instead of `QuantizeNode`.

### Graph Shape After Quantization

For a static quantized conv layer:
```
float input → [QuantizeNode] → int8 → [StaticQuantConv2dNode] → int8 → [DequantizeNode] → float output
```

---

## 6. CPrinter / Code Generation (`src/pytorch_to_c/codegen/c_printer.py`)

`CPrinter(ir_graph)` generates all C files.

### Code Generation Strategy

For each IR node in topological order, `_generate_node_code(node)` checks:
1. If node has `generate_c_code()` method (QuantIRNode, QuantizeNode, DequantizeNode, ProfilingWrapperNode) → **delegate to the node**
2. Otherwise, use built-in handlers: `_generate_conv2d()`, `_generate_linear()`, `_generate_relu()`, `_generate_batchnorm()`, `_generate_softmax()`, `_generate_add()`, `_generate_mean()`, `_generate_adaptive_avg_pool()`, `_generate_flatten_or_view()`

### Buffer Management

- **Slot-based reuse**: `_assign_buffer_slots()` uses interval graph coloring to assign `slot_0`, `slot_1`, ... rather than one buffer per node. This minimizes peak stack memory.
- **ReLU is in-place**: ReLU shares its input's slot (no separate buffer).
- Buffer sizes computed from `output_shape` via `_calculate_buffer_sizes()`.

### Generated Files

| File | Contents |
|------|----------|
| `model.h` | `void model_forward(const float* input, float* output);` with NHWC layout notes |
| `model.c` | Full implementation: buffer declarations, op calls in topo order, `memcpy` output |
| `weights.h` | `static const` arrays for all parameters (float, int8_t, int16_t based on numpy dtype) |
| `nn_ops_float.h` | Copied from `src/c_ops/` - float operation kernels |
| `nn_ops_int8.h` | Copied if graph has int8 nodes - quantized int8 kernels |
| `nn_ops_int16.h` | Copied if graph has int16 nodes - quantized int16 kernels |
| `*.ino` | Arduino sketch (if `arduino_mode=True`) with `setup()`/`loop()` |

### Key CPrinter Helper Methods

- `_get_buffer_name(node)` → `"slot_N"` (or `"input"` for input node)
- `_get_input_buffer(node, idx)` → buffer name of node's idx-th input
- `_sanitize_name(name)` → valid C identifier (dots/dashes → underscores)
- `_get_buffer_dtype(node)` → calls `node.get_c_dtype()`
- `_has_nodes_with_dtype(dtype)` → determines which `#include` headers to emit
- `_calculate_buffer_sizes()` → dict of `node_name → element_count`

---

## 7. C Runtime Kernels (`src/c_ops/`)

All header-only, `static inline`. No dynamic allocation. Caller provides output buffers.

### Layout Convention
- **Tensors**: NHWC (batch, height, width, channels)
- **Conv filters**: HWIO (kernel_h, kernel_w, in_channels, out_channels)
- **Linear weights**: [in_features, out_features] row-major
- **Bias**: always float32 even in quantized ops (design decision)

### Float ops (`nn_ops_float.h`)
`conv2d_nhwc`, `dense`, `relu`, `batchnorm2d_nhwc`, `softmax`, `mean_hwc`, `global_average_pool_2d`, `flatten`

### Int8 ops (`nn_ops_int8.h`)
- **Helpers**: `quantize_scalar_int8`, `dequantize_scalar_int8`, `compute_dynamic_scale_int8`
- **Vectorized**: `quantize_float_to_int8`, `dequantize_int8_to_float`
- **Layers**: `dense_int8(x, in_f, W, b, out_f, input_scale, weight_scale, offset, y)`, `conv2d_nhwc_int8(..., input_scale, weight_scale, offset, out)`
- **Activation**: `relu_int8` (in-place, clamp negative to 0)

### Quantized arithmetic pattern
```
int32_t acc = sum(int8 * int8)                  // integer MAC
float result = (float)acc * input_scale * weight_scale  // dequantize
result += bias[o]                                // float bias add
output[o] = quantize_scalar_int8(result, weight_scale, offset) // requantize
```

---

## 8. Passes (`src/passes/`)

IR optimization passes that run on `IRGraph`.

### IRPass (`src/passes/base.py`)
Abstract base: `apply(ir_graph) -> IRGraph`, `get_stats()`, `_log()`.

### FuseDequantQuantPass (`src/passes/fuse_dequant_quant.py`)
Eliminates redundant `DequantizeNode → QuantizeNode` pairs between consecutive quantized layers.

```
Before: quant_fc1 (int8) → dequant (float32) → quantize (int8) → quant_fc2 (int8)
After:  quant_fc1 (int8) → quant_fc2 (int8)
```

Only fuses when: same dtype, same scale/offset, static (not dynamic), dequant has exactly one user.

---

## 9. Lowering Details (`src/pytorch_to_c/lowering/lower.py`)

### Supported PyTorch Modules
| PyTorch | IR op_type | Weight handling |
|---------|-----------|----------------|
| `nn.Conv2d` | `conv2d` | Transposed to HWIO `[kH, kW, Cin, Cout]` |
| `nn.Linear` | `linear` | Transposed to `[in_features, out_features]` |
| `nn.ReLU` | `relu` | No weights |
| `nn.BatchNorm2d` | `batchnorm` | gamma, beta, running_mean, running_var extracted |
| `nn.Softmax` | `softmax` | No weights |
| `nn.AdaptiveAvgPool2d` | `adaptive_avg_pool` | No weights |

### Supported Functional/Method Ops
| PyTorch | IR op_type |
|---------|-----------|
| `torch.add` / `operator.add` | `add` |
| `tensor.view()` | `method_view` |
| `tensor.flatten()` | `method_flatten` |
| `tensor.mean(dim=...)` | `method_mean` |
| `tensor.size()` | `method_size` |

### Shape Inference
1. Primary: `torch.fx.passes.shape_prop.ShapeProp` with example input
2. Fallback: manual forward pass tracking intermediate tensor shapes

---

## 10. End-to-End Example: Quantized Inference

```python
import torch
import torch.nn as nn
from src.pytorch_to_c.compiler import compile_model
from src.pytorch_to_c.codegen.c_printer import CPrinter
from src.pytorch_to_c.quantization import StaticQuantRule, QuantizationTransform
from src.passes import FuseDequantQuantPass

# 1. Define model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 10)
    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)

model = MyModel()
model.eval()
example_input = torch.randn(1, 3, 32, 32)

# 2. Compile to IR (FX trace + lower)
ir_graph = compile_model(model, example_input, return_ir=True)

# 3. Apply quantization rules
rules = [
    StaticQuantRule(
        pattern=r'.*conv.*', dtype='int8',
        input_scale=0.05, input_offset=0,
        weight_scale=0.02, weight_offset=0,
        output_scale=0.05, output_offset=0
    ),
]
ir_graph = QuantizationTransform(rules).apply(ir_graph)

# 4. Optional: optimization pass
ir_graph = FuseDequantQuantPass(verbose=True).apply(ir_graph)

# 5. Generate C code
CPrinter(ir_graph).generate_all("output/")
```

Generated `model.c` will contain:
```c
void model_forward(const float* input, float* output) {
    float slot_0[...];
    int8_t slot_1[...];
    // ...
    
    // conv_input_q [quantize]
    quantize_float_to_int8(input, 3072, 0.05f, 0, slot_1);
    // conv [conv2d]
    conv2d_nhwc_int8(slot_1, 32, 32, 3, conv_weight, 3, 3, 16,
                     conv_bias, 1, 1, 1, 1, 0.05f, 0.02f, 0, slot_2);
    // conv_output_dq [dequantize]
    dequantize_int8_to_float(slot_2, 16384, 0.05f, 0, slot_0);
    // relu [relu]
    relu(slot_0, 16384);
    // ... pool, flatten, fc (float) ...
    memcpy(output, slot_N, 10 * sizeof(float));
}
```

---

## 11. How to Extend

### Adding a New Quantized Op

1. Create `src/pytorch_to_c/quantization/ops/quant_<op>.py`
2. Subclass `QuantIRNode`, implement `generate_c_code()`, `get_pre_nodes()`, `get_post_nodes()`
3. Add C kernel to `src/c_ops/nn_ops_int8.h` (and/or int16)
4. Update rule's `create_quant_node()` to handle the new `op_type`

### Adding a New Float Op

1. Add lowering in `lower.py` → `_lower_call_module()` or `_lower_call_function()`
2. Add code generation in `c_printer.py` → `_generate_<op>()`
3. Add C kernel to `src/c_ops/nn_ops_float.h`
4. Update `ops_map.py` if desired

### Adding a New IR Pass

1. Create `src/passes/my_pass.py`, subclass `IRPass`
2. Implement `apply(ir_graph) -> IRGraph`
3. User applies between lowering and codegen: `ir_graph = MyPass().apply(ir_graph)`

### Adding a New Transform (non-quantization)

Follow the profiling module pattern:
1. `rules.py` - define rule class with `matches()` and `create_*_node()`
2. `rule_matcher.py` - first-match engine
3. `graph_transform.py` - transform class with `apply(ir_graph)`
4. `ops/` - custom IRNode subclass with `generate_c_code()`

---

## 12. Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Weight layout | HWIO (conv), [in, out] (linear) | Matches NHWC C code access pattern |
| Tensor layout in C | NHWC | Standard for embedded/TFLite, cache-friendly for depthwise |
| Bias in quantized ops | float32 | Simpler, avoids int32 accumulation complexity |
| Quant/Dequant | Separate IRNodes | Enables optimization passes to fuse/remove them |
| Weight quantization | At compile time | Weights are known at compile time, no runtime cost |
| Buffer management | Slot reuse via interval coloring | Minimizes peak stack memory |
| ReLU | In-place on input buffer | Zero extra memory, safe because relu is monotonic |
| Code generation delegation | `node.generate_c_code()` if present, else built-in | Extensible: any custom node can emit its own C |
| Dtype strategy | Aggressive (stay quantized as long as possible) | Each quant op has pre/post nodes; passes fuse redundant conversions |
| Graph output | Must be float32 | C API: `model_forward(const float*, float*)` |

---

## 13. Important Caveats

- **torch.fx tracing**: Only static control flow is supported. Dynamic branching (`if x.shape[0] > 5`) will fail.
- **Shape inference**: Requires `example_input`. Without it, many ops cannot compute buffer sizes and will error.
- **NCHW → NHWC**: PyTorch uses NCHW. The generated C uses NHWC. Users must permute input before calling `model_forward()`.
- **No dynamic allocation**: All buffers are stack-allocated in `model_forward()`. Very large models may overflow the stack on constrained devices.
- **Batch size 1 only**: The compiler strips the batch dimension. Generated C always processes a single sample.
