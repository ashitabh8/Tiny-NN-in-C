# Quantization and Extensibility Guide

This document covers how quantization works in Tiny-NN-in-C, how to use the built-in quantization methods, and how the same rule + transform + node architecture can be used for any kind of node replacement.

## Architecture: Rules, Transforms, and Nodes

The compiler uses a three-part pattern for all graph modifications:

1. **Rule** -- defines _which_ nodes to match (via regex on node name) and _what_ to do when matched (create a replacement node).
2. **RuleMatcher** -- iterates over the IR graph and finds nodes matching each rule. First match wins.
3. **Transform** -- applies the matched rules to the graph: replaces nodes, inserts pre/post conversion nodes, rewires edges.

This pattern is used for quantization, profiling, and can be extended for any custom graph transformation.

```
Float IR Graph
      |
      v
 [Rule Matcher] -- which nodes match which rules?
      |
      v
 [Transform]    -- replace matched nodes with new node types
      |            insert pre/post conversion nodes
      |            quantize weights at compile time
      v
Modified IR Graph
      |
      v
 [CPrinter]     -- each custom node emits its own C via generate_c_code()
```

## Built-in Quantization Methods

### Static Quantization (`StaticQuantRule`)

All scales and offsets are provided by the user at compile time. Typically obtained from calibration with representative data.

```python
from src.pytorch_to_c.compiler import compile_model
from src.pytorch_to_c.codegen.c_printer import CPrinter
from src.pytorch_to_c.quantization import StaticQuantRule, QuantizationTransform

model = MyModel()
model.eval()
example_input = torch.randn(1, 784)

ir_graph = compile_model(model, example_input, return_ir=True)

rules = [
    StaticQuantRule(
        pattern=r'.*conv.*',    # regex matching node names
        dtype='int8',           # target dtype: 'int8' or 'int16'
        input_scale=0.05,       # scale for input activation quantization
        input_offset=0,
        weight_scale=0.02,      # scale for weight quantization
        weight_offset=0,
        output_scale=0.05,      # scale for output dequantization
        output_offset=0,
    ),
]

ir_graph = QuantizationTransform(rules).apply(ir_graph)
CPrinter(ir_graph).generate_all("output/")
```

The graph after quantization looks like:

```
float input --> [QuantizeNode] --> int8 --> [StaticQuantConv2dNode] --> int8 --> [DequantizeNode] --> float
```

### Dynamic Quantization (`DynamicQuantRuleMinMaxPerTensor`)

Weight scales are computed automatically from weight statistics at compile time. Input scales are computed at runtime via `compute_dynamic_scale_int8()`.

```python
from src.pytorch_to_c.quantization import DynamicQuantRuleMinMaxPerTensor

rules = [
    DynamicQuantRuleMinMaxPerTensor(
        pattern=r'.*fc.*',
        dtype='int8',
    ),
]
```

No calibration data needed -- the compiler inspects the weights and computes optimal scale/offset automatically.

### Mixed Precision

Different rules can target different layers with different dtypes:

```python
rules = [
    StaticQuantRule(pattern=r'.*conv.*', dtype='int16', ...),  # higher precision for convs
    StaticQuantRule(pattern=r'.*fc.*',   dtype='int8',  ...),  # aggressive for FC
]
```

Rules are matched first-come-first-served. Unmatched layers stay in float32.

## Optimization Passes

After quantization, optimization passes can simplify the graph.

### FuseDequantQuantPass

Eliminates redundant `DequantizeNode -> QuantizeNode` pairs between consecutive quantized layers when the scales match:

```python
from src.passes import FuseDequantQuantPass

ir_graph = FuseDequantQuantPass(verbose=True).apply(ir_graph)
```

Before:
```
quant_fc1 (int8) --> dequant (float32) --> quantize (int8) --> quant_fc2 (int8)
```

After:
```
quant_fc1 (int8) --> quant_fc2 (int8)
```

## How to Create a Custom Quantization Rule

If you need a different quantization strategy (e.g., asymmetric, calibration-based):

1. Subclass `QuantRule` in `src/pytorch_to_c/quantization/rules.py`:

```python
class MyCustomRule(QuantRule):
    def __init__(self, pattern, dtype, my_param):
        super().__init__(pattern, dtype)
        self.my_param = my_param

    def create_quant_node(self, node):
        # Return a QuantIRNode subclass for the matched node
        if node.op_type == 'linear':
            return MyQuantLinearNode(node, dtype=self.dtype, ...)
        raise ValueError(f"Unsupported op: {node.op_type}")

    def quantize_weights(self, weights):
        # Return quantized numpy array
        import numpy as np
        return np.clip(np.round(weights / self.my_param), -128, 127).astype(np.int8)
```

2. If your rule needs a new quantized op node, subclass `QuantIRNode`:

```python
class MyQuantLinearNode(QuantIRNode):
    def generate_c_code(self, c_printer):
        # Return list of C code lines
        return [f"my_custom_dense_int8({...});"]

    def get_pre_nodes(self):
        return [QuantizeNode(...)]

    def get_post_nodes(self):
        return [DequantizeNode(...)]
```

3. Add the C kernel to `src/c_ops/nn_ops_int8.h`.

4. Use your rule with `QuantizationTransform`:

```python
ir_graph = QuantizationTransform([MyCustomRule(...)]).apply(ir_graph)
```

## The Same Pattern for Any Node Replacement

The quantization architecture is not special -- it is a general-purpose node replacement pattern. The profiling module (`src/pytorch_to_c/profiling/`) demonstrates this with timing instrumentation:

```python
from src.pytorch_to_c.profiling import ProfilingRule, ProfilingTransform

rules = [
    ProfilingRule(pattern=r"conv", label="conv_layers"),
    ProfilingRule(pattern=r"fc",   label="fc_layers"),
]
ir_graph = ProfilingTransform(rules).apply(ir_graph)
```

To create your own transform for any purpose:

1. **`rules.py`** -- define a rule class with `matches()` and `create_*_node()`
2. **`rule_matcher.py`** -- first-match engine (can copy the quantization or profiling version)
3. **`graph_transform.py`** -- transform class with `apply(ir_graph)`
4. **`ops/`** -- custom IRNode subclass(es) with `generate_c_code()`

The key is that any `IRNode` with a `generate_c_code(self, c_printer)` method can emit arbitrary C code. The `CPrinter` delegates to it automatically.

## Adding a New Float Operation

1. Add lowering in `src/pytorch_to_c/lowering/lower.py` -- map the PyTorch op to an `IRNode`
2. Add code generation in `src/pytorch_to_c/codegen/c_printer.py` -- `_generate_<op>()`
3. Add the C kernel to `src/c_ops/nn_ops_float.h`

## Verifying Quantized Models

Use the verification tool to compare quantized C output against PyTorch:

```python
from tools.verify_model import verify_model
from src.pytorch_to_c.quantization import StaticQuantRule

results = verify_model(
    model=my_model,
    example_input=torch.randn(1, 784),
    quantization_rules=[StaticQuantRule(...)],
    tolerance=0.5,   # quantization error is larger than float
    num_samples=50,
)
print(results.summary())
```

The report includes per-sample pass/fail, max/mean error, and top-1 class match rate.
