# PyTorch-to-C Compiler: Presentation Slides

---

## PART 0: INTRODUCTION & MOTIVATION

---

### Slide 0.1: The Problem

**Deploying neural networks on microcontrollers is hard:**

- Limited memory (KB, not GB)
- No Python runtime
- No GPU
- Need C/C++ code
- Quantization essential for memory/speed

**Existing tools have friction:**
- TFLite Micro: Requires TensorFlow, has runtime overhead
- Glow: Complex setup, research-oriented
- ONNX Runtime: Heavy dependency, not MCU-focused

---

### Slide 0.2: Comparison with Existing Tools

| Feature | **Tiny-NN-in-C** | TFLite Micro | Glow |
|---------|------------------|--------------|------|
| **Framework** | PyTorch | TensorFlow | ONNX/PyTorch* |
| **Output** | Standalone C | .tflite + Interpreter | Compiled binary |
| **Runtime Required** | ❌ None | ✓ TFLite runtime | ❌ None |
| **Code Readability** | ✓ Human-readable | ❌ Binary format | ❌ Optimized IR |
| **Setup Complexity** | Simple (`pip install`) | Medium | Complex |
| **Binary Size** | ~KB (just your model) | ~100KB+ runtime | Varies |
| **Quantization** | int8, int16, mixed | int8 | int8 |
| **Custom Rules** | ✓ Regex-based | ❌ Per-layer config | ❌ Manual |
| **Extensibility** | ✓ Add nodes/passes easily | ❌ Modify C++ source | ❌ Modify LLVM |
| **Transparency** | ✓ See generated code | ❌ Black box | ❌ Black box |
| **Active Development** | Educational/Research | Production | Limited* |

*Glow development has slowed; PyTorch support via ONNX conversion

---

### Slide 0.3: When to Use Tiny-NN-in-C

**✓ USE THIS WHEN:**
- You work in PyTorch and want direct C generation
- You need human-readable C code (auditing, certification)
- You want fine-grained control over quantization per-layer
- You need minimal binary footprint (no runtime)
- You want to extend/customize the compiler
- Educational purposes / understanding how compilers work

**✗ USE TFLITE MICRO WHEN:**
- You already have TensorFlow models
- You need production-tested, battle-hardened code
- You need the full TFLite operator coverage
- You want Google's support ecosystem

**✗ USE GLOW WHEN:**
- You need maximum optimization (LLVM backend)
- You're targeting specific accelerators
- You have resources for complex setup

---

### Slide 0.4: Key Advantages Detailed

| Advantage | Why It Matters |
|-----------|----------------|
| **No Runtime** | Generated C is self-contained. Just compile and run. No interpreter overhead. |
| **PyTorch Native** | No model conversion. Use `torch.fx` directly on your PyTorch model. |
| **Readable Output** | You can inspect `model.c`. Great for debugging, auditing, certification. |
| **Rule-Based Quantization** | `pattern=r'.*encoder.*'` - quantize by name regex. Mix int8/int16/float32. |
| **IR-Based Architecture** | Clean IR enables optimization passes (like FuseDequantQuant). |
| **Extensible** | Add new ops: ~50 lines of Python + C kernel. No rebuilding compiler. |
| **Lightweight** | The compiler is pure Python. Output is just `.c` and `.h` files. |

---

### Slide 0.5: Architecture Comparison

**TFLite Micro:**
```
TensorFlow Model → Converter → .tflite file → TFLite Interpreter (C++) → Output
                                                    ↑
                                        Runtime on device (~100KB+)
```

**Glow:**
```
ONNX Model → Glow Frontend → High-level IR → Low-level IR → LLVM → Binary
                                   (Complex optimization pipeline)
```

**Tiny-NN-in-C:**
```
PyTorch Model → torch.fx → IRGraph → [Quantization] → [Passes] → C Code
                              ↓              ↓             ↓
                          Readable     Rule-based    User-defined
                          & Simple    per-layer      optimizations
```

**Key insight:** We trade maximum optimization for transparency and simplicity.

---

## PART 1: TUTORIAL & USE CASES

---

### Slide 1: What is This?

**A PyTorch-to-C compiler for microcontrollers**

- Takes PyTorch models → Generates standalone C code
- Supports quantization (int8, int16) for embedded deployment
- Extensible rule-based system for selective quantization
- IR-based architecture enables optimization passes

```bash
# Quick demo: compile a model
python examples/tiny_resnet.py
```

---

### Slide 2: Simple MLP Example

**Model:** 2-layer MLP (Linear → ReLU → Linear)

```python
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 4)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```

**Compile to C (no quantization):**
```python
from src.pytorch_to_c.compiler import compile_model

model = SimpleMLP()
compile_model(model, torch.randn(1, 16), output_dir="output/")
```

**Run test:**
```bash
python -m pytest test/test_integration.py -v -s
```

---

### Slide 3: ResNet-Style Model (Skip Connections)

**Model:** Conv → BatchNorm → ReLU → Skip Connection → Global Avg Pool → FC

```python
class TinyResNet(nn.Module):
    def __init__(self):
        self.conv_init = nn.Conv2d(3, 16, 3, padding=1)
        self.bn_init = nn.BatchNorm2d(16)
        self.block1 = ResNetBlock(16)  # Has skip connection
        self.fc = nn.Linear(16, 4)
    
    def forward(self, x):
        x = self.relu(self.bn_init(self.conv_init(x)))
        x = self.block1(x)
        x = x.mean(dim=[2, 3])  # Global average pool
        return self.fc(x)
```

**Run test (float32):**
```bash
python -m pytest test/test_quantization_e2e.py::TestResNetE2E::test_codegen_resnet_float -v -s
```

**Result:** Max error = `1.19e-07` ✓

---

### Slide 4: Static Quantization

**Key idea:** Scales are known at compile time (from calibration)

```python
from src.pytorch_to_c.quantization import StaticQuantRule, QuantizationTransform

rules = [
    StaticQuantRule(
        pattern=r'fc.*',           # Regex matches layer names
        dtype='int8',
        input_scale=0.01, input_offset=0,
        weight_scale=0.01, weight_offset=0,
        output_scale=0.01, output_offset=0
    )
]

ir_graph = compile_model(model, input, return_ir=True)
transform = QuantizationTransform(rules)
quant_ir = transform.apply(ir_graph)
```

**Run test:**
```bash
python -m pytest test/test_quantization_e2e.py::TestQuantizedE2E::test_static_quant_int8 -v -s
```

---

### Slide 5: Dynamic Quantization

**Key idea:** Input scale computed at runtime from actual data

```python
from src.pytorch_to_c.quantization import DynamicQuantRuleMinMaxPerTensor

rules = [
    DynamicQuantRuleMinMaxPerTensor(
        pattern=r'fc.*',
        dtype='int8'
    )
]
```

**Generated C code computes scale dynamically:**
```c
float scale = compute_dynamic_scale_int8(input, 784);
quantize_float_to_int8(input, 784, scale, 0, buf_quantized);
```

**Run test:**
```bash
python -m pytest test/test_quantization_e2e.py::TestQuantizedE2E::test_dynamic_quant_int8 -v -s
```

---

### Slide 6: Mixed Precision Quantization

**Selective quantization:** Different rules for different layers

```python
rules = [
    # Encoder: aggressive int8
    StaticQuantRule(pattern=r'.*encoder.*', dtype='int8', ...),
    
    # Output: higher precision int16  
    StaticQuantRule(pattern=r'.*output.*', dtype='int16', ...),
    
    # precision_layer: NO RULE → stays float32
]
```

**Model with mixed precision:**
```
encoder_fc1 (int8) → encoder_fc2 (int8) → precision_layer (float32) → output (int16)
```

**Run example:**
```bash
python examples/example_op_no_op.py
```

---

### Slide 7: ResNet with Quantization

**Full pipeline: ResNet + Static/Dynamic Quantization**

```bash
# Static int8 on conv + linear
python -m pytest test/test_quantization_e2e.py::TestResNetE2E::test_resnet_static_quant_int8 -v -s

# Dynamic int8 on conv + linear  
python -m pytest test/test_quantization_e2e.py::TestResNetE2E::test_resnet_dynamic_quant_int8 -v -s

# All ResNet tests
python -m pytest test/test_quantization_e2e.py::TestResNetE2E -v -s
```

**Results:**
| Test | Max Error |
|------|-----------|
| Float32 | 1.19e-07 |
| Static int8 | 1.63% |
| Static int16 | 0.07% |
| Dynamic int8 | 2.95% |

---

### Slide 8: Optimization Pass - FuseDequantQuant

**Problem:** Consecutive quantized layers have redundant conversions

```
BEFORE: fc1(int8) → dequant(float) → quant(int8) → fc2(int8)
AFTER:  fc1(int8) → fc2(int8)
```

**When scales match, this is a TRUE NO-OP:**
```python
from src.passes import FuseDequantQuantPass

fuse_pass = FuseDequantQuantPass(verbose=True)
optimized_ir = fuse_pass.apply(quant_ir)
```

**Run test (proves numerical equivalence):**
```bash
python -m pytest test/test_passes.py::TestFuseDequantQuantPass::test_fuse_pass_numerical_correctness -v -s
```

**Result:** Max error = `0.00e+00` (bit-identical!)

**Run full demo:**
```bash
python examples/example_op_no_op.py
```

---

## PART 2: INNER WORKINGS

---

### Slide 9: Architecture Overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   PyTorch   │ --> │  torch.fx   │ --> │   IR Graph  │ --> │   C Code    │
│    Model    │     │   Tracer    │     │   (IRNode)  │     │  Generator  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                              │
                                              v
                                     ┌─────────────────┐
                                     │  Quantization   │
                                     │   Transform     │
                                     │  (Rules-based)  │
                                     └─────────────────┘
                                              │
                                              v
                                     ┌─────────────────┐
                                     │  Optimization   │
                                     │    Passes       │
                                     └─────────────────┘
```

**Key files:**
- `src/pytorch_to_c/frontend/fx_tracer.py` - torch.fx tracing
- `src/pytorch_to_c/lowering/lower.py` - FX → IR conversion
- `src/pytorch_to_c/ir/graph.py` - IR graph structure
- `src/pytorch_to_c/codegen/c_printer.py` - C code generation

---

### Slide 10: IR Node Structure

**Base class:** `src/pytorch_to_c/ir/node.py`

```python
class IRNode:
    def __init__(self, name, op_type, inputs=None, dtype='float32'):
        self.name = name           # Unique identifier
        self.op_type = op_type     # 'linear', 'conv2d', 'relu', etc.
        self.inputs = inputs or [] # List of input IRNodes
        self.users = []            # List of nodes that use this output
        self.dtype = dtype         # 'float32', 'int8', 'int16'
        self.output_shape = None   # Inferred shape
        self.metadata = {}         # Extra info (weights, etc.)
    
    def get_c_dtype(self) -> str:
        """Returns C type: 'float', 'int8_t', 'int16_t'"""
```

**Graph is doubly-linked:** `inputs` + `users` enable traversal in both directions

---

### Slide 11: IR Graph Structure

**File:** `src/pytorch_to_c/ir/graph.py`

```python
class IRGraph:
    def __init__(self):
        self.nodes = []        # All nodes in topological order
        self.inputs = []       # Graph input nodes
        self.outputs = []      # Graph output nodes
        self.parameters = {}   # Weights: {'fc1_weight': np.array, ...}
    
    def print_graph(self):
        """Pretty-print the graph structure"""
```

**Example IR output:**
```
x [input] dtype=float32
  inputs: []
  users: [fc1]
fc1 [linear] dtype=float32
  inputs: [x]
  users: [relu]
  shape: (1, 8)
```

---

### Slide 12: Quantization Rule System

**Base class:** `src/pytorch_to_c/quantization/rules.py`

```python
class QuantRule(ABC):
    def __init__(self, pattern: str, dtype: str):
        self.pattern = re.compile(pattern)  # Regex for layer names
        self.dtype = dtype                   # 'int8' or 'int16'
    
    def matches(self, node: IRNode) -> bool:
        """Check if rule applies to this node"""
        return self.pattern.search(node.name) is not None
    
    @abstractmethod
    def create_quant_node(self, float_node: IRNode) -> QuantIRNode:
        """Create quantized version of the node"""
```

**Rule matching:** First matching rule wins
```python
RuleMatcher([rule1, rule2, rule3])  # Checked in order
```

---

### Slide 13: Writing a Custom Rule

**Example: Static Quantization Rule**

```python
class StaticQuantRule(QuantRule):
    def __init__(self, pattern, dtype, 
                 input_scale, input_offset,
                 weight_scale, weight_offset,
                 output_scale, output_offset):
        super().__init__(pattern, dtype)
        self.input_scale = input_scale
        # ... store all scales/offsets
    
    def create_quant_node(self, float_node: IRNode) -> QuantIRNode:
        if float_node.op_type == 'linear':
            return StaticQuantLinearNode(
                name=float_node.name,
                dtype=self.dtype,
                input_scale=self.input_scale,
                weight_scale=self.weight_scale,
                output_scale=self.output_scale,
                # ... 
            )
        elif float_node.op_type == 'conv2d':
            return StaticQuantConv2dNode(...)
        else:
            raise ValueError(f"Unsupported op: {float_node.op_type}")
```

---

### Slide 14: Quantized Node Structure

**Base class:** `src/pytorch_to_c/ir/quant_node.py`

```python
class QuantIRNode(IRNode):
    """Base class for all quantized operations"""
    
    @abstractmethod
    def get_pre_nodes(self) -> List[IRNode]:
        """Nodes to insert BEFORE this op (e.g., QuantizeNode)"""
    
    @abstractmethod
    def get_post_nodes(self) -> List[IRNode]:
        """Nodes to insert AFTER this op (e.g., DequantizeNode)"""
    
    @abstractmethod
    def generate_c_code(self, c_printer) -> str:
        """Generate the C code for this operation"""
```

**Key insight:** Each QuantIRNode controls its own conversion nodes!

---

### Slide 15: StaticQuantLinearNode Implementation

**File:** `src/pytorch_to_c/quantization/ops/quant_linear.py`

```python
class StaticQuantLinearNode(QuantIRNode):
    def __init__(self, name, dtype, input_scale, weight_scale, output_scale, ...):
        super().__init__(name, 'linear', dtype=dtype)
        self.input_scale = input_scale
        self.weight_scale = weight_scale
        self.output_scale = output_scale
    
    def get_pre_nodes(self):
        # Insert QuantizeNode before this layer
        return [QuantizeNode(
            name=f"{self.name}_input_q",
            dtype=self.dtype,
            scale=self.input_scale,
            offset=self.input_offset
        )]
    
    def get_post_nodes(self):
        # Insert DequantizeNode after this layer
        return [DequantizeNode(
            name=f"{self.name}_output_dq",
            scale=self.output_scale,
            offset=self.output_offset
        )]
    
    def generate_c_code(self, c_printer):
        return f"dense_int8({input}, {size}, {weight}, {bias}, {out_size}, " \
               f"{self.input_scale}f, {self.weight_scale}f, 0, {output});"
```

---

### Slide 16: QuantizationTransform Pipeline

**File:** `src/pytorch_to_c/quantization/graph_transform.py`

```python
class QuantizationTransform:
    def __init__(self, rules: List[QuantRule]):
        self.matcher = RuleMatcher(rules)
    
    def apply(self, ir_graph: IRGraph) -> IRGraph:
        # Step 1: Find nodes matching rules
        nodes_to_quantize = self._find_nodes_to_quantize(ir_graph)
        
        # Step 2: Replace float nodes with QuantIRNodes
        self._replace_nodes(ir_graph, nodes_to_quantize)
        
        # Step 3: Insert pre/post nodes (Quantize/Dequantize)
        self._insert_node_controlled_conversions(ir_graph)
        
        # Step 4: Validate output is float32
        self._validate_float_output(ir_graph)
        
        # Step 5: Quantize weights
        self._quantize_weights(ir_graph, nodes_to_quantize)
        
        return ir_graph
```

---

### Slide 17: Adding a New Quantized Operation

**Steps to add `QuantizedSoftmax`:**

1. **Create the node class:**
```python
# src/pytorch_to_c/quantization/ops/quant_softmax.py
class StaticQuantSoftmaxNode(QuantIRNode):
    def get_pre_nodes(self): ...
    def get_post_nodes(self): ...
    def generate_c_code(self, c_printer): ...
```

2. **Add C implementation:**
```c
// src/c_ops/nn_ops_int8.h
void softmax_int8(const int8_t* input, int size, 
                  float scale, int offset, int8_t* output);
```

3. **Update rule to create it:**
```python
# In StaticQuantRule.create_quant_node():
elif float_node.op_type == 'softmax':
    return StaticQuantSoftmaxNode(...)
```

4. **Export in `__init__.py`**

---

## PART 3: WRITING OPTIMIZATION PASSES

---

### Slide 18: Pass Infrastructure

**Base class:** `src/passes/base.py`

```python
class IRPass(ABC):
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.stats = {}  # Track optimization statistics
    
    @abstractmethod
    def apply(self, ir_graph: IRGraph) -> IRGraph:
        """Transform the IR graph"""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about optimizations applied"""
        return self.stats
```

---

### Slide 19: FuseDequantQuantPass Implementation

**File:** `src/passes/fuse_dequant_quant.py`

```python
class FuseDequantQuantPass(IRPass):
    def apply(self, ir_graph: IRGraph) -> IRGraph:
        # 1. Find fusable pairs
        pairs = self._find_fusable_pairs(ir_graph)
        
        # 2. Fuse each pair
        for dequant, quant in reversed(pairs):
            self._fuse_pair(ir_graph, dequant, quant)
        
        return ir_graph
    
    def _find_fusable_pairs(self, ir_graph):
        pairs = []
        for node in ir_graph.nodes:
            if isinstance(node, DequantizeNode):
                if len(node.users) == 1:
                    user = node.users[0]
                    if isinstance(user, QuantizeNode):
                        # Check: same dtype AND same scale
                        if (node.inputs[0].dtype == user.dtype and
                            self._scales_match(node, user)):
                            pairs.append((node, user))
        return pairs
```

---

### Slide 20: Graph Rewiring in Passes

**Before fusion:**
```
A → dequant → quant → B
    └─inputs  users─┘
```

**After fusion:**
```
A → B
```

**Code:**
```python
def _fuse_pair(self, ir_graph, dequant_node, quant_node):
    # Get source (before dequant) and targets (after quant)
    source = dequant_node.inputs[0]
    targets = quant_node.users.copy()
    
    # Rewire: source → targets
    source.users.remove(dequant_node)
    for target in targets:
        target.inputs = [source if x == quant_node else x 
                         for x in target.inputs]
        source.users.append(target)
    
    # Remove nodes from graph
    ir_graph.nodes.remove(dequant_node)
    ir_graph.nodes.remove(quant_node)
```

---

### Slide 21: Writing Your Own Pass

**Example: Dead Code Elimination Pass**

```python
class DeadCodeEliminationPass(IRPass):
    """Remove nodes whose outputs are never used"""
    
    def apply(self, ir_graph: IRGraph) -> IRGraph:
        changed = True
        while changed:
            changed = False
            for node in ir_graph.nodes.copy():
                # Skip output nodes
                if node in ir_graph.outputs:
                    continue
                # Skip input nodes
                if node in ir_graph.inputs:
                    continue
                # If no users, node is dead
                if len(node.users) == 0:
                    self._remove_node(ir_graph, node)
                    changed = True
        return ir_graph
```

---

### Slide 22: Pass Composition

**Chain multiple passes:**

```python
from src.passes import FuseDequantQuantPass

# Apply quantization
transform = QuantizationTransform(rules)
quant_ir = transform.apply(ir_graph)

# Apply optimization passes
passes = [
    FuseDequantQuantPass(verbose=True),
    # DeadCodeEliminationPass(),
    # ConstantFoldingPass(),
]

optimized_ir = quant_ir
for p in passes:
    optimized_ir = p.apply(optimized_ir)
    print(f"{p.__class__.__name__}: {p.get_stats()}")

# Generate final C code
printer = CPrinter(optimized_ir)
printer.generate_all("output/")
```

---

## PART 4: RUNNING THE FULL TEST SUITE

---

### Slide 23: Test Commands Summary

```bash
# All tests (64 tests)
python -m pytest test/ -v

# Float model tests
python -m pytest test/test_integration.py -v -s

# Quantization unit tests
python -m pytest test/test_quantization.py -v -s

# Quantization end-to-end tests
python -m pytest test/test_quantization_e2e.py -v -s

# Optimization pass tests
python -m pytest test/test_passes.py -v -s

# Specific test
python -m pytest test/test_passes.py::TestFuseDequantQuantPass::test_fuse_pass_numerical_correctness -v -s
```

---

### Slide 24: Key Numerical Results

| Test | Max Error | Command |
|------|-----------|---------|
| Float MLP | ~1e-6 | `pytest test/test_integration.py` |
| Float ResNet | 1.19e-07 | `pytest test/test_quantization_e2e.py::TestResNetE2E::test_codegen_resnet_float` |
| Static int8 | ~1-2% | `pytest test/test_quantization_e2e.py::TestQuantizedE2E::test_static_quant_int8` |
| Static int16 | ~0.1% | `pytest test/test_quantization_e2e.py::TestQuantizedE2E::test_static_quant_int16` |
| FusePass | **0.00e+00** | `pytest test/test_passes.py::TestFuseDequantQuantPass::test_fuse_pass_numerical_correctness` |

---

### Slide 25: File Structure

```
src/
├── pytorch_to_c/
│   ├── compiler.py              # Main entry point
│   ├── frontend/fx_tracer.py    # torch.fx tracing
│   ├── lowering/lower.py        # FX → IR conversion
│   ├── ir/
│   │   ├── node.py              # IRNode base class
│   │   ├── graph.py             # IRGraph
│   │   └── quant_node.py        # QuantIRNode base
│   ├── codegen/c_printer.py     # C code generation
│   └── quantization/
│       ├── rules.py             # QuantRule, StaticQuantRule, etc.
│       ├── graph_transform.py   # QuantizationTransform
│       └── ops/
│           ├── quant_utils.py   # QuantizeNode, DequantizeNode
│           ├── quant_linear.py  # StaticQuantLinearNode, etc.
│           └── quant_conv2d.py  # StaticQuantConv2dNode, etc.
├── passes/
│   ├── base.py                  # IRPass base class
│   └── fuse_dequant_quant.py    # FuseDequantQuantPass
└── c_ops/
    ├── nn_ops_float.h           # Float C kernels
    ├── nn_ops_int8.h            # Int8 C kernels
    └── nn_ops_int16.h           # Int16 C kernels

test/
├── test_integration.py          # Float model E2E tests
├── test_quantization.py         # Quantization unit tests
├── test_quantization_e2e.py     # Quantization E2E tests
└── test_passes.py               # Optimization pass tests

examples/
├── tiny_resnet.py               # ResNet compilation example
├── quantized_mlp.py             # Quantized MLP example
└── example_op_no_op.py          # FusePass demo
```

---

### Slide 26: Summary

**What we built:**
1. ✓ PyTorch → C compiler with torch.fx frontend
2. ✓ Clean IR with doubly-linked node graph
3. ✓ Rule-based quantization (static & dynamic)
4. ✓ Support for int8 and int16
5. ✓ Mixed precision quantization
6. ✓ Optimization pass infrastructure
7. ✓ FuseDequantQuantPass (bit-identical results!)

**Extensibility points:**
- Add new QuantRule subclasses
- Add new QuantIRNode implementations
- Add new IRPass optimizations
- Add new C kernel implementations

**Run all 64 tests:**
```bash
python -m pytest test/ -v
```

---

