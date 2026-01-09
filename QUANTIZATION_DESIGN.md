# Quantization System Design - Phase 2

## Design Philosophy

**Core Principle**: Modular, rule-based quantization where:
- Rules contain quantization parameters (no calibration in compiler)
- Quantization logic lives in rules/nodes themselves
- c_printer.py is just an orchestrator
- Easy for others to extend with new quantization strategies

## Architecture Overview

```
┌─────────────────┐
│ Float IR Graph  │
│ (from Phase 1)  │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Quantization   │  ← Users provide QuantRules
│  Rules Engine   │     (scale, offset, dtype, etc.)
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Quantized IR    │  ← Some nodes become QuantIRNodes
│ Graph           │     (child of IRNode)
└────────┬────────┘
         │
         v
┌─────────────────┐
│  C Code Gen     │  ← Calls node.generate_c_code()
│  (c_printer)    │     Logic in node/rule itself
└─────────────────┘
```

## Directory Structure

```
src/
├── pytorch_to_c/
│   ├── frontend/
│   ├── ir/
│   │   ├── node.py              # Base IRNode
│   │   ├── quant_node.py        # QuantIRNode (new)
│   │   └── graph.py
│   ├── lowering/
│   ├── quantization/            # NEW - moved from root
│   │   ├── __init__.py
│   │   ├── rules.py             # Base QuantRule, StaticQuantRule, etc.
│   │   ├── rule_matcher.py      # Match rules to nodes
│   │   ├── graph_transform.py   # Apply rules, modify IR
│   │   └── ops/                 # Quantized operation definitions
│   │       ├── __init__.py
│   │       ├── quant_linear.py  # QuantLinearNode
│   │       ├── quant_conv2d.py  # QuantConv2dNode
│   │       └── quant_utils.py   # Quantize/Dequantize nodes
│   ├── codegen/
│   │   ├── c_printer.py         # Orchestrator only
│   │   ├── ops_map.py
│   │   └── c_ops/               # NEW - C operation templates
│   │       ├── nn_ops_float.h
│   │       ├── nn_ops_int8.h    # NEW
│   │       └── nn_ops_int16.h   # NEW
│   └── compiler.py
```

## Core Components

### 1. QuantRule Base Class

```python
# src/pytorch_to_c/quantization/rules.py

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class QuantRule(ABC):
    """
    Base class for quantization rules.
    
    Rules are user-defined and specify:
    - Which nodes to quantize (pattern matching)
    - How to quantize them (dtype, scale, offset)
    - How to generate C code (via the node)
    """
    
    def __init__(self, pattern: str, dtype: str):
        """
        Args:
            pattern: Regex pattern to match node names
            dtype: Target data type ('int8', 'int16', 'float32')
        """
        self.pattern = pattern
        self.dtype = dtype
    
    @abstractmethod
    def matches(self, node: IRNode) -> bool:
        """Check if this rule applies to a node."""
        pass
    
    @abstractmethod
    def create_quant_node(self, node: IRNode) -> 'QuantIRNode':
        """Create a quantized version of the node."""
        pass
    
    @abstractmethod
    def get_quant_params(self) -> Dict[str, Any]:
        """Get quantization parameters for this rule."""
        pass
```

### 2. Concrete Rule Implementations

```python
class StaticQuantRule(QuantRule):
    """
    Static quantization with user-provided scale and offset.
    
    Use case: User has pre-calibrated their model
    """
    
    def __init__(self, pattern: str, dtype: str, 
                 scale: float, offset: int, per_channel: bool = False):
        super().__init__(pattern, dtype)
        self.scale = scale
        self.offset = offset  # zero_point
        self.per_channel = per_channel
    
    def matches(self, node: IRNode) -> bool:
        import re
        return re.match(self.pattern, node.name) is not None
    
    def create_quant_node(self, node: IRNode) -> 'QuantIRNode':
        # Factory: Create appropriate QuantNode based on op_type
        if node.op_type == 'linear':
            from .ops.quant_linear import QuantLinearNode
            return QuantLinearNode(
                node, 
                dtype=self.dtype,
                scale=self.scale,
                offset=self.offset,
                per_channel=self.per_channel
            )
        elif node.op_type == 'conv2d':
            from .ops.quant_conv2d import QuantConv2dNode
            return QuantConv2dNode(
                node,
                dtype=self.dtype,
                scale=self.scale,
                offset=self.offset,
                per_channel=self.per_channel
            )
        else:
            raise ValueError(f"No quantized version for {node.op_type}")
    
    def get_quant_params(self) -> Dict[str, Any]:
        return {
            'scale': self.scale,
            'offset': self.offset,
            'per_channel': self.per_channel
        }


class DynamicQuantRulePerTensor(QuantRule):
    """
    Dynamic quantization - quantize weights, keep activations in float.
    
    Use case: Quantize weights to save memory, but compute in float
    """
    
    def __init__(self, pattern: str, dtype: str):
        super().__init__(pattern, dtype)
        # No scale/offset - computed at runtime or from weights
    
    def matches(self, node: IRNode) -> bool:
        import re
        return re.match(self.pattern, node.name) is not None
    
    def create_quant_node(self, node: IRNode) -> 'QuantIRNode':
        # Similar to static, but different C code generation
        # Weights are quantized, activations stay float
        pass
    
    def get_quant_params(self) -> Dict[str, Any]:
        return {'dynamic': True}
```

### 3. QuantIRNode Base Class

```python
# src/pytorch_to_c/ir/quant_node.py

from .node import IRNode
from typing import List, Dict, Any
from abc import abstractmethod

class QuantIRNode(IRNode):
    """
    Quantized IR Node - extends IRNode with quantization info.
    
    Key difference from IRNode:
    - Has quantization parameters (scale, offset)
    - Can generate quantized C code
    - Knows about quantized data types
    """
    
    def __init__(self, original_node: IRNode, dtype: str,
                 scale: float, offset: int, per_channel: bool = False):
        # Copy properties from original node
        super().__init__(
            name=original_node.name,
            op_type=original_node.op_type,
            output_shape=original_node.output_shape,
            dtype=dtype,  # Override with quantized dtype
            metadata=original_node.metadata.copy()
        )
        
        # Copy connections
        self.inputs = original_node.inputs.copy()
        self.users = original_node.users.copy()
        
        # Quantization parameters
        self.scale = scale
        self.offset = offset  # zero_point
        self.per_channel = per_channel
        
        # Store original node for reference
        self.metadata['original_node'] = original_node
        self.metadata['quantized'] = True
        self.metadata['quant_params'] = {
            'scale': scale,
            'offset': offset,
            'per_channel': per_channel
        }
    
    @abstractmethod
    def generate_c_code(self, c_printer) -> List[str]:
        """
        Generate C code for this quantized operation.
        
        Args:
            c_printer: CPrinter instance (for accessing helpers)
        
        Returns:
            List of C code lines
        """
        pass
    
    @abstractmethod
    def get_c_dtype(self) -> str:
        """Return C data type (int8_t, int16_t, etc.)"""
        pass
    
    def get_buffer_dtype(self) -> str:
        """Return buffer data type for allocation."""
        return self.get_c_dtype()
```

### 4. Concrete Quantized Operations

```python
# src/pytorch_to_c/quantization/ops/quant_linear.py

from ...ir.quant_node import QuantIRNode
from typing import List

class QuantLinearNode(QuantIRNode):
    """Quantized Linear/Dense operation."""
    
    def generate_c_code(self, c_printer) -> List[str]:
        """Generate C code for quantized linear operation."""
        lines = []
        
        input_buffer = c_printer._get_input_buffer(self, 0)
        output_buffer = c_printer._get_buffer_name(self)
        weight_name = c_printer._sanitize_name(self.metadata['weight_name'])
        bias_name = c_printer._sanitize_name(self.metadata['bias_name']) \
                    if self.metadata.get('bias_name') else 'NULL'
        
        in_features = self.metadata['in_features']
        out_features = self.metadata['out_features']
        
        if self.dtype == 'int8':
            # Generate int8 quantized linear call
            lines.append(
                f"dense_int8("
                f"{input_buffer}, {in_features}, "
                f"{weight_name}_q, {bias_name}, {out_features}, "
                f"{self.scale}f, {self.offset}, "  # scale, zero_point
                f"{output_buffer});"
            )
        elif self.dtype == 'int16':
            # Similar for int16
            lines.append(
                f"dense_int16("
                f"{input_buffer}, {in_features}, "
                f"{weight_name}_q, {bias_name}, {out_features}, "
                f"{self.scale}f, {self.offset}, "
                f"{output_buffer});"
            )
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        
        return lines
    
    def get_c_dtype(self) -> str:
        return 'int8_t' if self.dtype == 'int8' else 'int16_t'
```

### 5. Rule Matcher

```python
# src/pytorch_to_c/quantization/rule_matcher.py

from typing import List, Optional
from ..ir.node import IRNode
from .rules import QuantRule

class RuleMatcher:
    """Matches quantization rules to IR nodes."""
    
    def __init__(self, rules: List[QuantRule]):
        self.rules = rules
    
    def find_matching_rule(self, node: IRNode) -> Optional[QuantRule]:
        """
        Find the first rule that matches a node.
        
        Rules are checked in order - first match wins.
        """
        for rule in self.rules:
            if rule.matches(node):
                return rule
        return None
    
    def get_quantizable_nodes(self, ir_graph) -> List[IRNode]:
        """Get all nodes that have matching rules."""
        quantizable = []
        for node in ir_graph.nodes:
            if self.find_matching_rule(node):
                quantizable.append(node)
        return quantizable
```

### 6. Graph Transformation

```python
# src/pytorch_to_c/quantization/graph_transform.py

from ..ir.graph import IRGraph
from ..ir.node import IRNode
from .rules import QuantRule
from .rule_matcher import RuleMatcher
from typing import List

class QuantizationTransform:
    """Transforms float IR graph to quantized IR graph."""
    
    def __init__(self, rules: List[QuantRule]):
        self.rules = rules
        self.matcher = RuleMatcher(rules)
    
    def apply(self, ir_graph: IRGraph) -> IRGraph:
        """
        Apply quantization rules to IR graph.
        
        Process:
        1. Find nodes matching rules
        2. Replace with QuantIRNodes
        3. Insert Quantize/Dequantize at boundaries
        4. Update graph connections
        """
        # Find nodes to quantize
        nodes_to_quantize = {}  # node -> rule mapping
        
        for node in ir_graph.nodes:
            rule = self.matcher.find_matching_rule(node)
            if rule:
                nodes_to_quantize[node] = rule
        
        # Replace nodes with quantized versions
        for node, rule in nodes_to_quantize.items():
            quant_node = rule.create_quant_node(node)
            self._replace_node(ir_graph, node, quant_node)
        
        # Insert Quantize/Dequantize at dtype boundaries
        self._insert_dtype_conversions(ir_graph)
        
        return ir_graph
    
    def _replace_node(self, ir_graph: IRGraph, old_node: IRNode, new_node: IRNode):
        """Replace a node in the graph."""
        # Update node list
        idx = ir_graph.nodes.index(old_node)
        ir_graph.nodes[idx] = new_node
        
        # Update connections (inputs already copied in QuantIRNode.__init__)
        # Update users to point to new node
        for user in old_node.users:
            user_inputs = user.inputs
            for i, inp in enumerate(user_inputs):
                if inp is old_node:
                    user_inputs[i] = new_node
    
    def _insert_dtype_conversions(self, ir_graph: IRGraph):
        """Insert Quantize/Dequantize nodes at dtype boundaries."""
        # For each edge in graph, check if dtypes differ
        # If float32 -> int8: insert Quantize
        # If int8 -> float32: insert Dequantize
        pass
```

### 7. Updated C Printer (Orchestrator Only)

```python
# src/pytorch_to_c/codegen/c_printer.py

class CPrinter:
    """
    Generates C code from IR graph.
    
    Now handles both float and quantized nodes.
    Delegates code generation to nodes themselves.
    """
    
    def _generate_node_code(self, node: IRNode) -> List[str]:
        """Generate C code for a node."""
        
        # Check if this is a quantized node
        if isinstance(node, QuantIRNode):
            # Delegate to node's own code generation
            return node.generate_c_code(self)
        
        # Regular float operations
        if node.op_type == 'input':
            return []
        elif node.op_type == 'conv2d':
            return self._generate_conv2d(node)
        elif node.op_type == 'linear':
            return self._generate_linear(node)
        # ... etc
    
    def _calculate_buffer_sizes(self) -> Dict[str, int]:
        """Calculate buffer sizes - now handles quantized dtypes."""
        sizes = {}
        
        for node in self.ir_graph.nodes:
            if node.op_type == 'input':
                continue
            
            # Use inferred shape
            if node.output_shape:
                shape = node.output_shape[1:]  # Remove batch
                size = math.prod(shape) if shape else 1
                sizes[node.name] = size
            else:
                sizes[node.name] = 1024  # Fallback
        
        return sizes
    
    def _get_buffer_dtype(self, node: IRNode) -> str:
        """Get C data type for buffer allocation."""
        if isinstance(node, QuantIRNode):
            return node.get_c_dtype()
        else:
            return 'float'
    
    def _declare_buffers(self) -> List[str]:
        """Declare buffers with correct dtypes."""
        lines = []
        buffer_sizes = self._calculate_buffer_sizes()
        
        for node in self.ir_graph.nodes:
            if node.op_type != 'input':
                size = buffer_sizes.get(node.name, 1024)
                dtype = self._get_buffer_dtype(node)
                buf_name = self._get_buffer_name(node)
                lines.append(f"    {dtype} {buf_name}[{size}];")
        
        return lines
```

## Usage Example

```python
from src.pytorch_to_c.quantization.rules import StaticQuantRule
from src.pytorch_to_c.quantization.graph_transform import QuantizationTransform
from src.pytorch_to_c.compiler import compile_model

# Define quantization rules
rules = [
    # Quantize all linear layers to int8
    StaticQuantRule(
        pattern=r'.*fc.*',      # Match fc1, fc2, etc.
        dtype='int8',
        scale=0.05,             # User provides from calibration
        offset=0,               # zero_point
        per_channel=False
    ),
    
    # Quantize conv layers to int16
    StaticQuantRule(
        pattern=r'.*conv.*',
        dtype='int16',
        scale=0.01,
        offset=0,
        per_channel=True
    ),
]

# Compile with quantization
model = MyModel()
example_input = torch.randn(1, 784)

# Normal compilation gives float IR
ir_graph = compile_model(model, example_input, "generated", verbose=False, 
                         return_ir=True)

# Apply quantization rules
quant_transform = QuantizationTransform(rules)
quantized_ir = quant_transform.apply(ir_graph)

# Generate C code from quantized IR
from src.pytorch_to_c.codegen.c_printer import CPrinter
printer = CPrinter(quantized_ir)
printer.generate_all("generated_quant")
```

## Key Design Decisions

### 1. **No Calibration in Compiler**
- Users provide scale/offset in rules
- Keeps compiler simple and focused
- Users can use their own calibration tools

### 2. **Logic in Nodes, Not Printer**
- Each QuantIRNode knows how to generate its C code
- c_printer.py is just an orchestrator
- Easy to add new quantization strategies

### 3. **Rule-Based System**
- Pattern matching on node names
- Flexible and expressive
- Can have different rules for different layers

### 4. **Inheritance Hierarchy**
```
IRNode (base)
  ├─ Regular ops (conv2d, linear, relu, ...)
  └─ QuantIRNode (quantized base)
       ├─ QuantLinearNode
       ├─ QuantConv2dNode
       └─ QuantizeNode / DequantizeNode
```

### 5. **Modularity**
- quantization/ module is self-contained
- Can be developed independently
- Easy for others to contribute new rules/ops

## Extension Points

Users can extend the system by:

1. **Adding new QuantRules**
   - Inherit from QuantRule
   - Implement matches() and create_quant_node()
   - Example: PerChannelStaticRule, MixedPrecisionRule

2. **Adding new QuantIRNodes**
   - Inherit from QuantIRNode
   - Implement generate_c_code()
   - Example: QuantBatchNormNode, QuantAttentionNode

3. **Adding new C operations**
   - Add to c_ops/nn_ops_int8.h
   - Use in QuantIRNode.generate_c_code()

## Testing Strategy

1. **Unit Tests**
   - Test rule matching
   - Test node creation
   - Test C code generation

2. **Integration Tests**
   - Test full quantization pipeline
   - Compare quantized C output with PyTorch quantized model

3. **Accuracy Tests**
   - Measure accuracy loss from quantization
   - Ensure within acceptable bounds

## Implementation Order

1. ✅ Phase 1 complete (float pipeline)
2. ✅ Shape inference complete
3. **Phase 2.1**: Core infrastructure
   - QuantRule base class
   - QuantIRNode base class
   - RuleMatcher
   - GraphTransform
4. **Phase 2.2**: Basic quantization
   - StaticQuantRule
   - QuantLinearNode
   - int8 C operations
5. **Phase 2.3**: Advanced features
   - QuantConv2dNode
   - Per-channel quantization
   - Dynamic quantization
   - int16 support

## Benefits of This Design

1. **Modular**: Each component has single responsibility
2. **Extensible**: Easy to add new rules/ops
3. **Testable**: Each component can be tested independently
4. **User-Friendly**: Simple API, users provide scale/offset
5. **Maintainable**: Logic is localized, not scattered
6. **Flexible**: Different quantization strategies coexist

## Open Questions / Discussion Points

1. **Weight quantization**: Do we quantize weights during compilation or expect pre-quantized weights?
2. **Mixed precision**: How to handle float->int8->float->int16 chains?
3. **Bias handling**: Keep biases in int32 or quantize differently?
4. **Activation quantization**: Static (at compile time) or dynamic (at runtime)?
5. **Memory layout**: How to store quantized weights in weights.h?

---

**Next Steps**: Review this design, discuss open questions, then implement in phases.

