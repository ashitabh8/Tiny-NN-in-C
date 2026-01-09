# Phase 2: Quantization Implementation Guide

**Status**: Ready for implementation  
**Last Updated**: Based on final design decisions

## Table of Contents
1. [Final Design Decisions](#final-design-decisions)
2. [Architecture Overview](#architecture-overview)
3. [Implementation Details](#implementation-details)
4. [Directory Structure](#directory-structure)
5. [Implementation Phases](#implementation-phases)
6. [Testing Strategy](#testing-strategy)

---

## Final Design Decisions

### ✅ Confirmed Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Weight Quantization** | During compilation, rule-defined | User's Rule Logic controls (e.g., `DynamicQuantRuleMinMaxPerTensor`) |
| **Bias Handling** | float32 | Simpler, can add int32 later as different op |
| **Dtype Strategy** | Aggressive | Keep quantized as long as possible |
| **Quant/Dequant Nodes** | Separate IRNodes | Allows optimization passes to remove redundant conversions |
| **IR Validation** | Check dtype compatibility | IRGraph validates input/output type compatibility |
| **Dynamic Quantization** | Quantize input on-the-fly | `quantize_input → dense_int8 → dequantize_output` |
| **Weight Storage** | Only quantized weights | No need to store float32 if quantized |
| **API Style** | Explicit transformation | User controls each step, can inspect IR |
| **Error Handling** | Fail fast | Raise error for unsupported ops |

---

## Architecture Overview

### Data Flow

```
PyTorch Model
     ↓
Float IR Graph (from Phase 1)
     ↓
Apply QuantRules ← User-provided rules with scale/offset
     ↓
Quantized IR Graph (QuantIRNodes + Quant/Dequant nodes)
     ↓
Validate dtype compatibility
     ↓
C Code Generation (node.generate_c_code())
     ↓
Generated C files
```

### Key Principles

1. **Modular**: Logic in rules/nodes, c_printer is orchestrator
2. **Extensible**: Easy to add new rules and quantized operations
3. **Principled**: Quant/Dequant are first-class IRNodes for optimization
4. **Type-Safe**: IR validates dtype compatibility between nodes

---

## Implementation Details

### 1. IR Node Hierarchy

```python
IRNode (base class)
├── Regular Operations (float32)
│   ├── Conv2dNode
│   ├── LinearNode
│   ├── ReLUNode
│   └── ...
│
└── QuantIRNode (quantized operations)
    ├── QuantLinearNode (int8/int16)
    ├── QuantConv2dNode (int8/int16)
    │
    └── Conversion Nodes (separate!)
        ├── QuantizeNode (float32 → int8/int16)
        └── DequantizeNode (int8/int16 → float32)
```

**Key Point**: `QuantizeNode` and `DequantizeNode` are **separate IRNodes**, NOT fused into `QuantLinearNode`. This allows optimization passes to remove redundant conversions.

### 2. Base Classes

#### IRNode (existing, in `src/pytorch_to_c/ir/node.py`)

Add dtype validation:

```python
class IRNode:
    def __init__(self, name, op_type, dtype='float32', ...):
        self.dtype = dtype  # 'float32', 'int8', 'int16'
        # ... existing fields
    
    def validate_input_dtypes(self) -> bool:
        """Validate that input dtypes are compatible with this op."""
        # To be implemented
        pass
```

#### QuantIRNode (new, in `src/pytorch_to_c/ir/quant_node.py`)

```python
from abc import abstractmethod
from .node import IRNode
from typing import List, Dict, Any

class QuantIRNode(IRNode):
    """
    Base class for quantized operations.
    
    Extends IRNode with quantization parameters and C code generation.
    """
    
    def __init__(self, original_node: IRNode, dtype: str,
                 scale: float, offset: int, **kwargs):
        """
        Args:
            original_node: The float node being quantized
            dtype: 'int8' or 'int16'
            scale: Quantization scale
            offset: Zero point
        """
        # Copy properties from original
        super().__init__(
            name=original_node.name,
            op_type=original_node.op_type,
            output_shape=original_node.output_shape,
            dtype=dtype,
            metadata=original_node.metadata.copy()
        )
        
        # Copy connections
        self.inputs = original_node.inputs.copy()
        self.users = original_node.users.copy()
        
        # Quantization parameters
        self.scale = scale
        self.offset = offset
        
        # Store in metadata for C generation
        self.metadata['quantized'] = True
        self.metadata['quant_params'] = {
            'scale': scale,
            'offset': offset,
            'dtype': dtype
        }
    
    @abstractmethod
    def generate_c_code(self, c_printer) -> List[str]:
        """Generate C code for this quantized operation."""
        pass
    
    def get_c_dtype(self) -> str:
        """Return C data type."""
        return 'int8_t' if self.dtype == 'int8' else 'int16_t'
    
    def validate_input_dtypes(self) -> bool:
        """Quantized ops expect quantized inputs."""
        for inp in self.inputs:
            if inp.dtype not in ['int8', 'int16']:
                raise TypeError(
                    f"QuantNode {self.name} expects quantized input, "
                    f"got {inp.dtype} from {inp.name}"
                )
        return True
```

### 3. Conversion Nodes (Separate IRNodes)

#### QuantizeNode (in `src/pytorch_to_c/quantization/ops/quant_utils.py`)

```python
from ...ir.node import IRNode
from typing import List

class QuantizeNode(IRNode):
    """
    Conversion node: float32 → int8/int16
    
    This is a separate IRNode (not fused into QuantLinearNode).
    Future optimization passes can remove redundant conversions.
    """
    
    def __init__(self, name: str, target_dtype: str, scale: float, offset: int):
        super().__init__(
            name=name,
            op_type='quantize',
            dtype=target_dtype,  # Output dtype
            metadata={
                'scale': scale,
                'offset': offset,
                'target_dtype': target_dtype,
                'source_dtype': 'float32'
            }
        )
        self.scale = scale
        self.offset = offset
        self.target_dtype = target_dtype
    
    def generate_c_code(self, c_printer) -> List[str]:
        """Generate C code for quantization."""
        input_buf = c_printer._get_input_buffer(self, 0)
        output_buf = c_printer._get_buffer_name(self)
        
        # Get size from buffer size calculation
        buffer_sizes = c_printer._calculate_buffer_sizes()
        size = buffer_sizes.get(self.name, 1024)
        
        if self.target_dtype == 'int8':
            return [
                f"quantize_float_to_int8({input_buf}, {size}, "
                f"{self.scale}f, {self.offset}, {output_buf});"
            ]
        elif self.target_dtype == 'int16':
            return [
                f"quantize_float_to_int16({input_buf}, {size}, "
                f"{self.scale}f, {self.offset}, {output_buf});"
            ]
        else:
            raise ValueError(f"Unsupported target dtype: {self.target_dtype}")
    
    def validate_input_dtypes(self) -> bool:
        """Quantize expects float32 input."""
        if self.inputs and self.inputs[0].dtype != 'float32':
            raise TypeError(
                f"QuantizeNode {self.name} expects float32 input, "
                f"got {self.inputs[0].dtype}"
            )
        return True


class DequantizeNode(IRNode):
    """
    Conversion node: int8/int16 → float32
    
    Separate IRNode for optimization flexibility.
    """
    
    def __init__(self, name: str, source_dtype: str, scale: float, offset: int):
        super().__init__(
            name=name,
            op_type='dequantize',
            dtype='float32',  # Output dtype
            metadata={
                'scale': scale,
                'offset': offset,
                'source_dtype': source_dtype,
                'target_dtype': 'float32'
            }
        )
        self.scale = scale
        self.offset = offset
        self.source_dtype = source_dtype
    
    def generate_c_code(self, c_printer) -> List[str]:
        """Generate C code for dequantization."""
        input_buf = c_printer._get_input_buffer(self, 0)
        output_buf = c_printer._get_buffer_name(self)
        
        buffer_sizes = c_printer._calculate_buffer_sizes()
        size = buffer_sizes.get(self.name, 1024)
        
        if self.source_dtype == 'int8':
            return [
                f"dequantize_int8_to_float({input_buf}, {size}, "
                f"{self.scale}f, {self.offset}, {output_buf});"
            ]
        elif self.source_dtype == 'int16':
            return [
                f"dequantize_int16_to_float({input_buf}, {size}, "
                f"{self.scale}f, {self.offset}, {output_buf});"
            ]
        else:
            raise ValueError(f"Unsupported source dtype: {self.source_dtype}")
    
    def validate_input_dtypes(self) -> bool:
        """Dequantize expects quantized input."""
        if self.inputs and self.inputs[0].dtype not in ['int8', 'int16']:
            raise TypeError(
                f"DequantizeNode {self.name} expects quantized input, "
                f"got {self.inputs[0].dtype}"
            )
        return True
```

### 4. Quantized Operations

#### QuantLinearNode (in `src/pytorch_to_c/quantization/ops/quant_linear.py`)

```python
from ...ir.quant_node import QuantIRNode
from typing import List

class QuantLinearNode(QuantIRNode):
    """Quantized Linear/Dense operation."""
    
    def generate_c_code(self, c_printer) -> List[str]:
        """Generate C code for quantized linear."""
        lines = []
        
        input_buffer = c_printer._get_input_buffer(self, 0)
        output_buffer = c_printer._get_buffer_name(self)
        weight_name = c_printer._sanitize_name(self.metadata['weight_name'])
        
        # Bias stays in float32 (design decision)
        bias_name = c_printer._sanitize_name(self.metadata['bias_name']) \
                    if self.metadata.get('bias_name') else 'NULL'
        
        in_features = self.metadata['in_features']
        out_features = self.metadata['out_features']
        
        if self.dtype == 'int8':
            lines.append(
                f"dense_int8("
                f"{input_buffer}, {in_features}, "
                f"{weight_name}_q, {bias_name}, {out_features}, "  # Note: bias is float
                f"{self.scale}f, {self.offset}, "
                f"{output_buffer});"
            )
        elif self.dtype == 'int16':
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

#### QuantConv2dNode (in `src/pytorch_to_c/quantization/ops/quant_conv2d.py`)

```python
from ...ir.quant_node import QuantIRNode
from typing import List

class QuantConv2dNode(QuantIRNode):
    """Quantized Conv2D operation."""
    
    def generate_c_code(self, c_printer) -> List[str]:
        """Generate C code for quantized conv2d."""
        lines = []
        
        input_buffer = c_printer._get_input_buffer(self, 0)
        output_buffer = c_printer._get_buffer_name(self)
        weight_name = c_printer._sanitize_name(self.metadata['weight_name'])
        bias_name = c_printer._sanitize_name(self.metadata['bias_name']) \
                    if self.metadata.get('bias_name') else 'NULL'
        
        # Extract parameters
        kernel_size = self.metadata['kernel_size']
        stride = self.metadata['stride']
        padding = self.metadata['padding']
        in_channels = self.metadata['in_channels']
        out_channels = self.metadata['out_channels']
        
        # Convert to scalars
        k_h, k_w = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        s_h, s_w = stride if isinstance(stride, (tuple, list)) else (stride, stride)
        p_h, p_w = padding if isinstance(padding, (tuple, list)) else (padding, padding)
        
        # Get input shape from shape inference
        if self.inputs and self.inputs[0].output_shape:
            # Assuming NHWC: [batch, height, width, channels]
            in_shape = self.inputs[0].output_shape
            in_h, in_w = in_shape[1], in_shape[2]
        else:
            in_h, in_w = 32, 32  # Fallback
        
        pad_same = 1 if p_h > 0 or p_w > 0 else 0
        
        if self.dtype == 'int8':
            lines.append(
                f"conv2d_nhwc_int8("
                f"{input_buffer}, {in_h}, {in_w}, {in_channels}, "
                f"{weight_name}_q, {k_h}, {k_w}, {out_channels}, "
                f"{bias_name}, {s_h}, {s_w}, {pad_same}, "
                f"{self.scale}f, {self.offset}, "
                f"{output_buffer});"
            )
        elif self.dtype == 'int16':
            lines.append(
                f"conv2d_nhwc_int16("
                f"{input_buffer}, {in_h}, {in_w}, {in_channels}, "
                f"{weight_name}_q, {k_h}, {k_w}, {out_channels}, "
                f"{bias_name}, {s_h}, {s_w}, {pad_same}, "
                f"{self.scale}f, {self.offset}, "
                f"{output_buffer});"
            )
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        
        return lines
    
    def get_c_dtype(self) -> str:
        return 'int8_t' if self.dtype == 'int8' else 'int16_t'
```

### 5. Quantization Rules

#### Base QuantRule (in `src/pytorch_to_c/quantization/rules.py`)

```python
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import re

class QuantRule(ABC):
    """
    Base class for quantization rules.
    
    Rules define:
    - Which nodes to quantize (pattern matching)
    - How to quantize them (dtype, scale, offset)
    - What to do with weights (quantize during compilation)
    """
    
    def __init__(self, pattern: str, dtype: str):
        """
        Args:
            pattern: Regex pattern to match node names
            dtype: Target data type ('int8', 'int16')
        """
        self.pattern = pattern
        self.dtype = dtype
        self._compiled_pattern = re.compile(pattern)
    
    def matches(self, node) -> bool:
        """Check if this rule applies to a node."""
        return self._compiled_pattern.match(node.name) is not None
    
    @abstractmethod
    def create_quant_node(self, node):
        """Create a quantized version of the node."""
        pass
    
    @abstractmethod
    def quantize_weights(self, weights) -> Any:
        """
        Quantize weights during compilation.
        
        This is called by the rule itself, not by the compiler.
        Different rules can have different weight quantization strategies.
        """
        pass
    
    def get_quant_params(self) -> Dict[str, Any]:
        """Get quantization parameters."""
        return {
            'dtype': self.dtype,
        }


class StaticQuantRule(QuantRule):
    """
    Static quantization with user-provided scale and offset.
    
    User has pre-calibrated and provides the quantization parameters.
    """
    
    def __init__(self, pattern: str, dtype: str, scale: float, offset: int):
        super().__init__(pattern, dtype)
        self.scale = scale
        self.offset = offset
    
    def matches(self, node) -> bool:
        return super().matches(node)
    
    def create_quant_node(self, node):
        """Factory: Create appropriate QuantNode based on op_type."""
        if node.op_type == 'linear':
            from .ops.quant_linear import QuantLinearNode
            return QuantLinearNode(
                node,
                dtype=self.dtype,
                scale=self.scale,
                offset=self.offset
            )
        elif node.op_type == 'conv2d':
            from .ops.quant_conv2d import QuantConv2dNode
            return QuantConv2dNode(
                node,
                dtype=self.dtype,
                scale=self.scale,
                offset=self.offset
            )
        else:
            raise ValueError(
                f"Cannot quantize operation '{node.op_type}' for node '{node.name}'. "
                f"Quantized version not implemented."
            )
    
    def quantize_weights(self, weights):
        """Quantize weights using this rule's scale/offset."""
        import numpy as np
        
        # Quantize: Q = round(W / scale) + offset
        weights_q = np.round(weights / self.scale) + self.offset
        
        # Clamp to dtype range
        if self.dtype == 'int8':
            weights_q = np.clip(weights_q, -128, 127).astype(np.int8)
        elif self.dtype == 'int16':
            weights_q = np.clip(weights_q, -32768, 32767).astype(np.int16)
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        
        return weights_q
    
    def get_quant_params(self) -> Dict[str, Any]:
        return {
            'dtype': self.dtype,
            'scale': self.scale,
            'offset': self.offset
        }


class DynamicQuantRuleMinMaxPerTensor(QuantRule):
    """
    Dynamic quantization using min-max per-tensor.
    
    Weights are quantized during compilation using min-max range.
    Activations are quantized on-the-fly at runtime.
    """
    
    def __init__(self, pattern: str, dtype: str):
        super().__init__(pattern, dtype)
        # Scale/offset computed from weight statistics
    
    def create_quant_node(self, node):
        """Similar to StaticQuantRule but compute scale/offset from weights."""
        # Get weights from node metadata
        if 'weights' not in node.metadata and 'weight_name' not in node.metadata:
            raise ValueError(f"Node {node.name} has no weights to quantize")
        
        # Compute scale and offset from weight min/max
        # This happens during compilation!
        weights = node.metadata.get('weights')
        if weights is None:
            # Get from IR graph parameters
            weight_name = node.metadata['weight_name']
            # TODO: Get weights from ir_graph.parameters
            raise NotImplementedError("Need access to ir_graph.parameters")
        
        scale, offset = self._compute_scale_offset(weights)
        
        # Create quantized node
        if node.op_type == 'linear':
            from .ops.quant_linear import QuantLinearNode
            return QuantLinearNode(node, dtype=self.dtype, scale=scale, offset=offset)
        elif node.op_type == 'conv2d':
            from .ops.quant_conv2d import QuantConv2dNode
            return QuantConv2dNode(node, dtype=self.dtype, scale=scale, offset=offset)
        else:
            raise ValueError(f"Cannot quantize {node.op_type}")
    
    def quantize_weights(self, weights):
        """Quantize weights using min-max."""
        import numpy as np
        
        scale, offset = self._compute_scale_offset(weights)
        
        weights_q = np.round(weights / scale) + offset
        
        if self.dtype == 'int8':
            weights_q = np.clip(weights_q, -128, 127).astype(np.int8)
        elif self.dtype == 'int16':
            weights_q = np.clip(weights_q, -32768, 32767).astype(np.int16)
        
        return weights_q
    
    def _compute_scale_offset(self, weights):
        """Compute scale and offset from weight statistics."""
        import numpy as np
        
        w_min = float(np.min(weights))
        w_max = float(np.max(weights))
        
        if self.dtype == 'int8':
            q_min, q_max = -128, 127
        elif self.dtype == 'int16':
            q_min, q_max = -32768, 32767
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        
        scale = (w_max - w_min) / (q_max - q_min)
        offset = q_min - int(round(w_min / scale))
        
        return scale, offset
    
    def get_quant_params(self) -> Dict[str, Any]:
        return {
            'dtype': self.dtype,
            'strategy': 'dynamic_minmax_per_tensor'
        }
```

### 6. Graph Transformation

#### QuantizationTransform (in `src/pytorch_to_c/quantization/graph_transform.py`)

```python
from ..ir.graph import IRGraph
from ..ir.node import IRNode
from .rules import QuantRule
from .rule_matcher import RuleMatcher
from .ops.quant_utils import QuantizeNode, DequantizeNode
from typing import List, Dict

class QuantizationTransform:
    """
    Transform float IR graph to quantized IR graph.
    
    Process:
    1. Find nodes matching rules
    2. Replace with QuantIRNodes
    3. Insert Quantize/Dequantize nodes at dtype boundaries (aggressive strategy)
    4. Validate dtype compatibility
    5. Quantize weights during compilation
    """
    
    def __init__(self, rules: List[QuantRule]):
        self.rules = rules
        self.matcher = RuleMatcher(rules)
    
    def apply(self, ir_graph: IRGraph) -> IRGraph:
        """
        Apply quantization rules to IR graph.
        
        Returns modified IRGraph with QuantIRNodes.
        """
        # Step 1: Find nodes to quantize
        nodes_to_quantize = {}  # node -> rule
        
        for node in ir_graph.nodes:
            rule = self.matcher.find_matching_rule(node)
            if rule:
                nodes_to_quantize[node] = rule
        
        # Step 2: Replace nodes with quantized versions
        for node, rule in nodes_to_quantize.items():
            try:
                quant_node = rule.create_quant_node(node)
                self._replace_node(ir_graph, node, quant_node)
            except ValueError as e:
                # Fail fast (design decision)
                raise ValueError(
                    f"Failed to quantize node '{node.name}' ({node.op_type}): {e}"
                )
        
        # Step 3: Insert Quantize/Dequantize at dtype boundaries (AGGRESSIVE)
        self._insert_dtype_conversions(ir_graph)
        
        # Step 4: Quantize weights during compilation
        self._quantize_weights(ir_graph, nodes_to_quantize)
        
        # Step 5: Validate dtype compatibility
        self._validate_graph(ir_graph)
        
        return ir_graph
    
    def _replace_node(self, ir_graph: IRGraph, old_node: IRNode, new_node: IRNode):
        """Replace a node in the graph."""
        # Update node list
        idx = ir_graph.nodes.index(old_node)
        ir_graph.nodes[idx] = new_node
        
        # Update users to point to new node
        for user in old_node.users:
            for i, inp in enumerate(user.inputs):
                if inp is old_node:
                    user.inputs[i] = new_node
        
        # Update new_node's users
        new_node.users = old_node.users.copy()
    
    def _insert_dtype_conversions(self, ir_graph: IRGraph):
        """
        Insert Quantize/Dequantize nodes at dtype boundaries.
        
        AGGRESSIVE strategy: Only insert conversions when absolutely necessary.
        Future optimization passes can remove redundant conversions.
        """
        nodes_to_insert = []  # (insert_after, new_node)
        
        for node in ir_graph.nodes:
            for i, input_node in enumerate(node.inputs):
                input_dtype = input_node.dtype
                expected_dtype = self._get_expected_input_dtype(node)
                
                # Check if conversion needed
                if input_dtype != expected_dtype:
                    # Insert conversion node
                    if expected_dtype in ['int8', 'int16'] and input_dtype == 'float32':
                        # Need Quantize node
                        # Get quant params from the node that expects quantized input
                        if hasattr(node, 'scale') and hasattr(node, 'offset'):
                            conv_node = QuantizeNode(
                                name=f"{input_node.name}_to_{node.name}_quantize",
                                target_dtype=expected_dtype,
                                scale=node.scale,
                                offset=node.offset
                            )
                            nodes_to_insert.append((input_node, conv_node, node, i))
                    
                    elif expected_dtype == 'float32' and input_dtype in ['int8', 'int16']:
                        # Need Dequantize node
                        if hasattr(input_node, 'scale') and hasattr(input_node, 'offset'):
                            conv_node = DequantizeNode(
                                name=f"{input_node.name}_to_{node.name}_dequantize",
                                source_dtype=input_dtype,
                                scale=input_node.scale,
                                offset=input_node.offset
                            )
                            nodes_to_insert.append((input_node, conv_node, node, i))
        
        # Insert all conversion nodes
        for input_node, conv_node, user_node, input_idx in nodes_to_insert:
            # Insert into graph
            insert_idx = ir_graph.nodes.index(input_node) + 1
            ir_graph.nodes.insert(insert_idx, conv_node)
            
            # Connect: input_node → conv_node → user_node
            conv_node.inputs = [input_node]
            input_node.users.append(conv_node)
            
            # Update user_node's input
            user_node.inputs[input_idx] = conv_node
            conv_node.users = [user_node]
            
            # Infer output shape for conversion node
            conv_node.output_shape = input_node.output_shape
    
    def _get_expected_input_dtype(self, node: IRNode) -> str:
        """Get the expected input dtype for a node."""
        if hasattr(node, 'dtype'):
            # QuantIRNodes expect inputs of the same dtype
            if node.dtype in ['int8', 'int16']:
                return node.dtype
        
        # Default: float32
        return 'float32'
    
    def _quantize_weights(self, ir_graph: IRGraph, nodes_to_quantize: Dict):
        """
        Quantize weights during compilation.
        
        For each quantized node:
        1. Get the rule that quantized it
        2. Use rule.quantize_weights() to quantize
        3. Replace float weights with quantized weights in ir_graph.parameters
        4. Store only quantized weights (design decision)
        """
        for node, rule in nodes_to_quantize.items():
            # Get weight name
            weight_name = node.metadata.get('weight_name')
            if not weight_name:
                continue  # No weights (e.g., ReLU)
            
            # Get float weights from ir_graph.parameters
            if weight_name not in ir_graph.parameters:
                continue
            
            weights_float = ir_graph.parameters[weight_name]
            
            # Quantize using rule's logic
            weights_q = rule.quantize_weights(weights_float)
            
            # Replace in parameters (only store quantized, not float)
            ir_graph.parameters[weight_name + '_q'] = weights_q
            
            # Remove float weights to save space
            del ir_graph.parameters[weight_name]
            
            # Update node metadata to point to quantized weights
            node.metadata['weight_name'] = weight_name + '_q'
    
    def _validate_graph(self, ir_graph: IRGraph):
        """
        Validate dtype compatibility in the graph.
        
        Ensures all nodes have compatible input dtypes.
        """
        for node in ir_graph.nodes:
            try:
                if hasattr(node, 'validate_input_dtypes'):
                    node.validate_input_dtypes()
            except TypeError as e:
                raise TypeError(f"Dtype validation failed for node '{node.name}': {e}")
```

### 7. Updated CPrinter

#### Modifications to `src/pytorch_to_c/codegen/c_printer.py`

```python
class CPrinter:
    """
    C code generator - now handles both float and quantized nodes.
    
    Acts as orchestrator - delegates code generation to nodes.
    """
    
    def _generate_node_code(self, node: IRNode) -> List[str]:
        """Generate C code for a node."""
        
        # Import QuantIRNode to check instance
        from ..ir.quant_node import QuantIRNode
        from ..quantization.ops.quant_utils import QuantizeNode, DequantizeNode
        
        # Check if this is a quantized node or conversion node
        if isinstance(node, (QuantIRNode, QuantizeNode, DequantizeNode)):
            # Delegate to node's own code generation
            return node.generate_c_code(self)
        
        # Regular float operations (existing code)
        if node.op_type == 'input':
            return []
        elif node.op_type == 'conv2d':
            return self._generate_conv2d(node)
        elif node.op_type == 'linear':
            return self._generate_linear(node)
        elif node.op_type == 'relu':
            return self._generate_relu(node)
        # ... etc
    
    def _get_buffer_dtype(self, node: IRNode) -> str:
        """Get C data type for buffer allocation."""
        from ..ir.quant_node import QuantIRNode
        from ..quantization.ops.quant_utils import QuantizeNode, DequantizeNode
        
        if isinstance(node, QuantIRNode):
            return node.get_c_dtype()
        elif isinstance(node, QuantizeNode):
            return 'int8_t' if node.target_dtype == 'int8' else 'int16_t'
        elif isinstance(node, DequantizeNode):
            return 'float'
        else:
            return 'float'
    
    def generate_model_c(self) -> str:
        """Generate model.c with correct buffer dtypes."""
        lines = []
        lines.append("// Auto-generated model implementation")
        lines.append("// DO NOT EDIT")
        lines.append("")
        lines.append("#include \"model.h\"")
        lines.append("#include \"weights.h\"")
        lines.append("#include \"nn_ops_float.h\"")
        lines.append("#include \"nn_ops_int8.h\"")   # NEW
        lines.append("#include \"nn_ops_int16.h\"")  # NEW
        lines.append("")
        lines.append("#include <string.h>")
        lines.append("#include <math.h>")  # For roundf in quant/dequant
        lines.append("")
        
        lines.append("void model_forward(const float* input, float* output) {")
        lines.append("    // Intermediate buffers")
        
        # Declare buffers with correct dtypes
        buffer_sizes = self._calculate_buffer_sizes()
        for node in self.ir_graph.nodes:
            if node.op_type != 'input':
                size = buffer_sizes.get(node.name, 1024)
                dtype = self._get_buffer_dtype(node)
                buf_name = self._get_buffer_name(node)
                lines.append(f"    {dtype} {buf_name}[{size}];")
        
        lines.append("")
        lines.append("    // Forward pass")
        
        # Generate code for each node
        for node in self.ir_graph.nodes:
            node_code = self._generate_node_code(node)
            if node_code:
                lines.append("")
                lines.append(f"    // {node.name} [{node.op_type}, {node.dtype}]")
                lines.extend([f"    {line}" for line in node_code])
        
        lines.append("")
        
        # Copy final output
        if self.ir_graph.outputs:
            output_node = self.ir_graph.outputs[0]
            output_buffer = self._get_buffer_name(output_node)
            output_size = buffer_sizes.get(output_node.name, 1024)
            lines.append(f"    // Copy output")
            lines.append(f"    memcpy(output, {output_buffer}, {output_size} * sizeof(float));")
        
        lines.append("}")
        lines.append("")
        
        return "\n".join(lines)
    
    def generate_weights_h(self) -> str:
        """Generate weights.h with both float and quantized weights."""
        lines = []
        lines.append("// Auto-generated weights file")
        lines.append("// DO NOT EDIT")
        lines.append("")
        lines.append("#ifndef WEIGHTS_H_")
        lines.append("#define WEIGHTS_H_")
        lines.append("")
        lines.append("#include <stddef.h>")
        lines.append("#include <stdint.h>")  # For int8_t, int16_t
        lines.append("")
        
        # Generate arrays for each parameter
        for param_name, param_data in self.ir_graph.parameters.items():
            # Flatten the array
            flat_data = param_data.flatten()
            
            # Determine C type from numpy dtype
            if param_data.dtype == 'int8':
                c_type = 'int8_t'
                format_str = lambda v: f"{int(v)}"
            elif param_data.dtype == 'int16':
                c_type = 'int16_t'
                format_str = lambda v: f"{int(v)}"
            elif param_data.dtype == 'int32':
                c_type = 'int32_t'
                format_str = lambda v: f"{int(v)}"
            else:
                # float32
                c_type = 'float'
                format_str = lambda v: f"{float(v):.8f}f"
            
            c_name = self._sanitize_name(param_name)
            lines.append(f"// Shape: {param_data.shape}, dtype: {param_data.dtype}")
            lines.append(f"static const {c_type} {c_name}[{len(flat_data)}] = {{")
            
            # Write data in chunks
            for i in range(0, len(flat_data), 8):
                chunk = flat_data[i:i+8]
                values_str = ", ".join([format_str(v) for v in chunk])
                lines.append(f"    {values_str},")
            
            lines.append("};")
            lines.append("")
        
        lines.append("#endif // WEIGHTS_H_")
        return "\n".join(lines)
    
    def _copy_c_ops_headers(self, output_dir: str) -> None:
        """Copy C operation headers including quantized versions."""
        import shutil
        from pathlib import Path
        
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent
        c_ops_dir = project_root / "src" / "c_ops"
        
        headers = ["nn_ops_float.h", "nn_ops_int8.h", "nn_ops_int16.h"]
        
        for header in headers:
            src = c_ops_dir / header
            if src.exists():
                dst = Path(output_dir) / header
                shutil.copy2(src, dst)
```

---

## Directory Structure

```
src/
├── pytorch_to_c/
│   ├── frontend/
│   │   └── fx_tracer.py
│   ├── ir/
│   │   ├── __init__.py
│   │   ├── node.py              # IRNode (add dtype validation)
│   │   ├── quant_node.py        # NEW: QuantIRNode base class
│   │   └── graph.py
│   ├── lowering/
│   │   └── lower.py
│   ├── quantization/            # NEW MODULE (moved from root)
│   │   ├── __init__.py
│   │   ├── rules.py             # QuantRule, StaticQuantRule, DynamicQuantRuleMinMaxPerTensor
│   │   ├── rule_matcher.py      # RuleMatcher
│   │   ├── graph_transform.py   # QuantizationTransform
│   │   └── ops/
│   │       ├── __init__.py
│   │       ├── quant_linear.py  # QuantLinearNode
│   │       ├── quant_conv2d.py  # QuantConv2dNode
│   │       └── quant_utils.py   # QuantizeNode, DequantizeNode
│   ├── codegen/
│   │   ├── c_printer.py         # Updated to handle QuantIRNode
│   │   ├── ops_map.py
│   │   └── c_ops/               # NEW subdirectory
│   │       ├── nn_ops_float.h   # Existing
│   │       ├── nn_ops_int8.h    # NEW
│   │       └── nn_ops_int16.h   # NEW
│   └── compiler.py
```

---

## Implementation Phases

### Phase 2.1: Core Infrastructure ⚠️ START HERE

**Goal**: Set up the foundation

**Tasks**:
1. Move `quantization/` to `src/pytorch_to_c/quantization/`
2. Create `src/pytorch_to_c/ir/quant_node.py` with `QuantIRNode` base class
3. Create `src/pytorch_to_c/quantization/rules.py` with:
   - `QuantRule` (base)
   - `StaticQuantRule`
4. Create `src/pytorch_to_c/quantization/rule_matcher.py`
5. Create `src/pytorch_to_c/quantization/ops/quant_utils.py` with:
   - `QuantizeNode`
   - `DequantizeNode`
6. Add dtype validation to `IRNode`

**Validation**:
- [ ] All classes instantiate correctly
- [ ] Rule matching works
- [ ] Dtype validation catches errors

---

### Phase 2.2: Basic Quantization (int8, Linear)

**Goal**: Get int8 linear quantization working end-to-end

**Tasks**:
1. Create `src/pytorch_to_c/quantization/ops/quant_linear.py` with `QuantLinearNode`
2. Implement `QuantizationTransform` in `graph_transform.py`
3. Create `src/c_ops/nn_ops_int8.h` with:
   - `quantize_float_to_int8()`
   - `dequantize_int8_to_float()`
   - `dense_int8()`
4. Update `CPrinter` to:
   - Handle `QuantIRNode` delegation
   - Handle correct buffer dtypes
   - Generate quantized weights in `weights.h`
5. Add helper to copy `nn_ops_int8.h` to output

**Test Model**: TinyMLP with int8 quantization

**Validation**:
- [ ] TinyMLP compiles with int8 quantization
- [ ] Generated C code compiles with gcc
- [ ] C output is numerically close to PyTorch quantized model

---

### Phase 2.3: Conv2D Support

**Goal**: Add quantized Conv2D

**Tasks**:
1. Create `src/pytorch_to_c/quantization/ops/quant_conv2d.py` with `QuantConv2dNode`
2. Add `conv2d_nhwc_int8()` to `nn_ops_int8.h`
3. Test with CNN model

**Test Model**: Simple CNN with conv2d + linear

**Validation**:
- [ ] CNN compiles with int8 quantization
- [ ] Generated C code compiles
- [ ] C output is numerically close to PyTorch

---

### Phase 2.4: Advanced Features

**Goal**: Add int16, dynamic quantization, per-channel

**Tasks**:
1. Create `src/c_ops/nn_ops_int16.h`
2. Add int16 support to `QuantLinearNode` and `QuantConv2dNode`
3. Implement `DynamicQuantRuleMinMaxPerTensor`
4. Add per-channel quantization support
5. Implement quantized ReLU (int8_relu)

**Test Models**: 
- MixedNet with int16
- TinyMLP with dynamic quantization

**Validation**:
- [ ] All test models compile and run
- [ ] Accuracy degradation within acceptable bounds
- [ ] Memory usage reduced as expected

---

## Testing Strategy

### Unit Tests

**Location**: `test/test_quantization.py`

```python
def test_rule_matching():
    """Test pattern matching works."""
    rule = StaticQuantRule(r'.*fc.*', 'int8', 0.05, 0)
    
    fc_node = IRNode(name='fc1', op_type='linear')
    relu_node = IRNode(name='relu1', op_type='relu')
    
    assert rule.matches(fc_node)
    assert not rule.matches(relu_node)

def test_weight_quantization():
    """Test weights are quantized correctly."""
    import numpy as np
    
    rule = StaticQuantRule(r'.*', 'int8', 0.1, 0)
    weights = np.array([0.0, 0.1, 0.2, -0.1, -0.2])
    
    weights_q = rule.quantize_weights(weights)
    
    assert weights_q.dtype == np.int8
    assert len(weights_q) == len(weights)

def test_quant_node_creation():
    """Test QuantIRNode created correctly."""
    from src.pytorch_to_c.quantization.ops.quant_linear import QuantLinearNode
    
    node = IRNode(name='fc1', op_type='linear', metadata={
        'in_features': 10,
        'out_features': 5,
        'weight_name': 'fc1.weight',
        'bias_name': 'fc1.bias'
    })
    
    quant_node = QuantLinearNode(node, dtype='int8', scale=0.05, offset=0)
    
    assert quant_node.dtype == 'int8'
    assert quant_node.scale == 0.05
    assert quant_node.offset == 0

def test_dtype_validation():
    """Test dtype validation catches errors."""
    from src.pytorch_to_c.quantization.ops.quant_linear import QuantLinearNode
    
    float_node = IRNode(name='fc1', op_type='linear', dtype='float32')
    quant_node = QuantLinearNode(..., dtype='int8', ...)
    
    # Connect: float → quant (should fail validation)
    quant_node.inputs = [float_node]
    
    with pytest.raises(TypeError):
        quant_node.validate_input_dtypes()
```

### Integration Tests

**Location**: `test/test_quantization_integration.py`

```python
def test_tiny_mlp_int8_quantization():
    """Test end-to-end quantization of TinyMLP."""
    model = TinyMLP()
    example_input = torch.randn(1, 784)
    
    # Compile to float IR
    ir_graph = compile_model(model, example_input, return_ir=True)
    
    # Apply quantization
    rules = [
        StaticQuantRule(r'.*fc.*', 'int8', 0.05, 0),
    ]
    transform = QuantizationTransform(rules)
    quant_ir = transform.apply(ir_graph)
    
    # Check quantization applied
    assert any(isinstance(n, QuantLinearNode) for n in quant_ir.nodes)
    assert any(n.op_type == 'quantize' for n in quant_ir.nodes)
    assert any(n.op_type == 'dequantize' for n in quant_ir.nodes)
    
    # Generate C code
    printer = CPrinter(quant_ir)
    printer.generate_all("tmp/quant_test")
    
    # Check generated files
    assert os.path.exists("tmp/quant_test/model.c")
    assert os.path.exists("tmp/quant_test/weights.h")
    assert os.path.exists("tmp/quant_test/nn_ops_int8.h")

def test_quantized_c_output():
    """Test quantized C model output matches PyTorch."""
    # ... similar to existing test_compare_outputs but with quantization
```

### Accuracy Tests

```python
def test_quantization_accuracy():
    """Test accuracy degradation is within bounds."""
    model = TinyMLP()
    example_input = torch.randn(100, 784)
    
    # Float output
    float_output = model(example_input).detach().numpy()
    
    # Quantized C output
    quant_output = compile_and_run_quantized(model, example_input, ...)
    
    # Check accuracy
    max_error = np.max(np.abs(float_output - quant_output))
    mean_error = np.mean(np.abs(float_output - quant_output))
    
    assert max_error < 0.1  # Acceptable threshold
    assert mean_error < 0.05
```

---

## C Operations Reference

### nn_ops_int8.h (NEW)

```c
#ifndef NN_OPS_INT8_H_
#define NN_OPS_INT8_H_

#include <stdint.h>
#include <math.h>

// Quantization helpers
static inline int8_t quantize_float_to_int8(float x, float scale, int offset) {
    int32_t val = (int32_t)roundf(x / scale) + offset;
    if (val < -128) val = -128;
    if (val > 127) val = 127;
    return (int8_t)val;
}

static inline float dequantize_int8_to_float(int8_t x, float scale, int offset) {
    return scale * (float)(x - offset);
}

// Vectorized quantization
void quantize_float_to_int8_vec(const float* input, int size, float scale, int offset, int8_t* output) {
    for (int i = 0; i < size; i++) {
        output[i] = quantize_float_to_int8(input[i], scale, offset);
    }
}

void dequantize_int8_to_float_vec(const int8_t* input, int size, float scale, int offset, float* output) {
    for (int i = 0; i < size; i++) {
        output[i] = dequantize_int8_to_float(input[i], scale, offset);
    }
}

// Quantized dense (linear) operation
// Bias stays in float32 (design decision)
void dense_int8(
    const int8_t* x, int in_features,
    const int8_t* W, const float* bias, int out_features,
    float scale, int offset,
    int8_t* y)
{
    for (int o = 0; o < out_features; ++o) {
        int32_t acc = 0;
        
        // Integer accumulation
        for (int i = 0; i < in_features; ++i) {
            acc += (int32_t)x[i] * (int32_t)W[i * out_features + o];
        }
        
        // Convert to float, add bias, quantize back
        float result = (float)acc * scale;
        if (bias) {
            result += bias[o];
        }
        
        y[o] = quantize_float_to_int8(result, scale, offset);
    }
}

// Quantized Conv2D NHWC
void conv2d_nhwc_int8(
    const int8_t* input, int in_h, int in_w, int in_c,
    const int8_t* kernel, int k_h, int k_w, int out_c,
    const float* bias,
    int stride_h, int stride_w, int pad_same,
    float scale, int offset,
    int8_t* output)
{
    // Implementation similar to float version but with int8 arithmetic
    // ... (detailed implementation)
}

// Quantized ReLU (very simple for int8)
void relu_int8(int8_t* x, int size) {
    for (int i = 0; i < size; i++) {
        if (x[i] < 0) x[i] = 0;
    }
}

#endif // NN_OPS_INT8_H_
```

---

## API Usage Examples

### Example 1: Basic Static Quantization

```python
import torch
from src.pytorch_to_c.compiler import compile_model
from src.pytorch_to_c.quantization.rules import StaticQuantRule
from src.pytorch_to_c.quantization.graph_transform import QuantizationTransform
from src.pytorch_to_c.codegen.c_printer import CPrinter

# Model
model = TinyMLP()
example_input = torch.randn(1, 784)

# Step 1: Compile to float IR
ir_graph = compile_model(model, example_input, return_ir=True)

# Step 2: Define quantization rules
rules = [
    StaticQuantRule(
        pattern=r'.*fc.*',  # Match all fc layers
        dtype='int8',
        scale=0.05,
        offset=0
    ),
]

# Step 3: Apply quantization
transform = QuantizationTransform(rules)
quant_ir = transform.apply(ir_graph)

# Step 4: Generate C code
printer = CPrinter(quant_ir)
printer.generate_all("generated_quant/")
```

### Example 2: Dynamic Quantization

```python
from src.pytorch_to_c.quantization.rules import DynamicQuantRuleMinMaxPerTensor

# Dynamic quantization rule
rules = [
    DynamicQuantRuleMinMaxPerTensor(
        pattern=r'.*fc.*',
        dtype='int8'
    ),
]

# Apply (rest is the same)
transform = QuantizationTransform(rules)
quant_ir = transform.apply(ir_graph)
```

### Example 3: Mixed Precision

```python
# Different quantization for different layers
rules = [
    # Conv layers: int16 for higher accuracy
    StaticQuantRule(
        pattern=r'.*conv.*',
        dtype='int16',
        scale=0.01,
        offset=0
    ),
    
    # FC layers: int8 for memory efficiency
    StaticQuantRule(
        pattern=r'.*fc.*',
        dtype='int8',
        scale=0.05,
        offset=0
    ),
]
```

---

## Success Criteria

Phase 2 is complete when:

- [x] All design decisions finalized
- [ ] Core infrastructure implemented (Phase 2.1)
- [ ] TinyMLP compiles with int8 quantization (Phase 2.2)
- [ ] Generated C code compiles with gcc
- [ ] C output is numerically close to PyTorch quantized model
- [ ] Conv2D quantization works (Phase 2.3)
- [ ] Advanced features implemented (Phase 2.4)
- [ ] All tests passing
- [ ] Documentation complete

---

## Next Steps

1. ✅ Design decisions finalized
2. **Move quantization/ to src/** (first task)
3. **Implement Phase 2.1** (core infrastructure)
4. **Test Phase 2.1** (unit tests)
5. **Implement Phase 2.2** (basic quantization)
6. **Test Phase 2.2** (integration tests)
7. Continue with Phase 2.3 and 2.4

---

**Ready to implement!** 🚀

