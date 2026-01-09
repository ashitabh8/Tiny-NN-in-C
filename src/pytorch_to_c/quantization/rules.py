"""
Quantization Rules - Define how to quantize different nodes
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import re
import numpy as np


class QuantRule(ABC):
    """
    Base class for quantization rules.
    
    Rules define:
    - Which nodes to quantize (via pattern matching on node names)
    - How to quantize them (dtype, scale, offset)
    - How to quantize weights (during compilation)
    """
    
    def __init__(self, pattern: str, dtype: str):
        """
        Initialize a quantization rule.
        
        Args:
            pattern: Regex pattern to match node names
            dtype: Target data type ('int8' or 'int16')
        """
        self.pattern = pattern
        self.dtype = dtype
        self._compiled_pattern = re.compile(pattern)
    
    def matches(self, node) -> bool:
        """
        Check if this rule applies to a node.
        
        Args:
            node: IRNode to check
            
        Returns:
            True if the node name matches the pattern
        """
        return self._compiled_pattern.match(node.name) is not None
    
    @abstractmethod
    def create_quant_node(self, node):
        """
        Create a quantized version of the node.
        
        Args:
            node: The float IRNode to quantize
            
        Returns:
            QuantIRNode subclass instance
        """
        pass
    
    @abstractmethod
    def quantize_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Quantize weights during compilation.
        
        Args:
            weights: Float weights as numpy array
            
        Returns:
            Quantized weights as int8 or int16 numpy array
        """
        pass
    
    def get_quant_params(self) -> Dict[str, Any]:
        """Get quantization parameters."""
        return {'dtype': self.dtype}


class StaticQuantRule(QuantRule):
    """
    Static quantization with user-provided scales and offsets.
    
    User provides pre-calibrated scale and offset values for:
    - Input activations
    - Weights
    - Output activations
    
    Weights are quantized during compilation using weight_scale/weight_offset.
    """
    
    def __init__(
        self,
        pattern: str,
        dtype: str,
        input_scale: float,
        input_offset: int,
        weight_scale: float,
        weight_offset: int,
        output_scale: float,
        output_offset: int
    ):
        """
        Initialize static quantization rule.
        
        Args:
            pattern: Regex pattern to match node names
            dtype: Target data type ('int8' or 'int16')
            input_scale: Scale for input activation quantization
            input_offset: Zero point for input activation
            weight_scale: Scale for weight quantization
            weight_offset: Zero point for weights
            output_scale: Scale for output dequantization
            output_offset: Zero point for output
        """
        super().__init__(pattern, dtype)
        self.input_scale = input_scale
        self.input_offset = input_offset
        self.weight_scale = weight_scale
        self.weight_offset = weight_offset
        self.output_scale = output_scale
        self.output_offset = output_offset
    
    def create_quant_node(self, node):
        """
        Create a quantized node based on the operation type.
        
        Args:
            node: The float IRNode to quantize
            
        Returns:
            QuantIRNode subclass instance
            
        Raises:
            ValueError: If operation type is not supported
        """
        if node.op_type == 'linear':
            from .ops.quant_linear import StaticQuantLinearNode
            return StaticQuantLinearNode(
                original_node=node,
                dtype=self.dtype,
                input_scale=self.input_scale,
                weight_scale=self.weight_scale,
                output_scale=self.output_scale,
                offset=self.input_offset  # Use input_offset for QuantizeNode
            )
        elif node.op_type == 'conv2d':
            from .ops.quant_conv2d import StaticQuantConv2dNode
            return StaticQuantConv2dNode(
                original_node=node,
                dtype=self.dtype,
                input_scale=self.input_scale,
                weight_scale=self.weight_scale,
                output_scale=self.output_scale,
                offset=self.input_offset
            )
        else:
            raise ValueError(
                f"Cannot quantize operation '{node.op_type}' for node '{node.name}'. "
                f"Quantized version not implemented."
            )
    
    def quantize_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Quantize weights using weight_scale and weight_offset.
        
        Formula: Q = round(W / scale) + offset
        
        Args:
            weights: Float weights as numpy array
            
        Returns:
            Quantized weights as int8 or int16 numpy array
        """
        weights_q = np.round(weights / self.weight_scale) + self.weight_offset
        
        if self.dtype == 'int8':
            weights_q = np.clip(weights_q, -128, 127).astype(np.int8)
        elif self.dtype == 'int16':
            weights_q = np.clip(weights_q, -32768, 32767).astype(np.int16)
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        
        return weights_q
    
    def get_quant_params(self) -> Dict[str, Any]:
        """Get quantization parameters."""
        return {
            'dtype': self.dtype,
            'input_scale': self.input_scale,
            'input_offset': self.input_offset,
            'weight_scale': self.weight_scale,
            'weight_offset': self.weight_offset,
            'output_scale': self.output_scale,
            'output_offset': self.output_offset
        }
    
    def __repr__(self) -> str:
        return (f"StaticQuantRule(pattern='{self.pattern}', dtype='{self.dtype}', "
                f"input_scale={self.input_scale}, weight_scale={self.weight_scale}, "
                f"output_scale={self.output_scale})")


class DynamicQuantRuleMinMaxPerTensor(QuantRule):
    """
    Dynamic quantization using min-max per-tensor.
    
    Scale and offset are computed from weight statistics during compilation.
    Activations are quantized on-the-fly at runtime.
    """
    
    def __init__(self, pattern: str, dtype: str):
        """
        Initialize dynamic quantization rule.
        
        Args:
            pattern: Regex pattern to match node names
            dtype: Target data type ('int8' or 'int16')
        """
        super().__init__(pattern, dtype)
        # Scale/offset will be computed from weight statistics
        self._computed_scale: Optional[float] = None
        self._computed_offset: Optional[int] = None
    
    def create_quant_node(self, node):
        """
        Create a quantized node with computed scale/offset.
        
        Note: Weights must be available in node.metadata or ir_graph.parameters
        """
        # For dynamic quantization, we need to compute scale/offset from weights
        # This requires access to the weights, which should be in ir_graph.parameters
        raise NotImplementedError(
            "DynamicQuantRuleMinMaxPerTensor.create_quant_node requires "
            "access to weights. Use QuantizationTransform to apply this rule."
        )
    
    def create_quant_node_with_weights(self, node, weights: np.ndarray):
        """
        Create a quantized node with scale computed from weights.
        
        For dynamic quantization:
        - Weight scale is computed from weight values
        - Input scale will be computed at runtime
        
        Args:
            node: The float IRNode to quantize
            weights: The weights to use for computing scale
            
        Returns:
            QuantIRNode subclass instance
        """
        # Compute scale from weight statistics
        self._computed_scale, self._computed_offset = self._compute_scale_offset(weights)
        
        if node.op_type == 'linear':
            from .ops.quant_linear import DynamicQuantLinearNode
            return DynamicQuantLinearNode(
                original_node=node,
                dtype=self.dtype,
                weight_scale=self._computed_scale,
                offset=self._computed_offset
            )
        elif node.op_type == 'conv2d':
            from .ops.quant_conv2d import DynamicQuantConv2dNode
            return DynamicQuantConv2dNode(
                original_node=node,
                dtype=self.dtype,
                weight_scale=self._computed_scale,
                offset=self._computed_offset
            )
        else:
            raise ValueError(f"Cannot quantize {node.op_type}")
    
    def quantize_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Quantize weights using min-max per-tensor.
        
        Args:
            weights: Float weights as numpy array
            
        Returns:
            Quantized weights as int8 or int16 numpy array
        """
        scale, offset = self._compute_scale_offset(weights)
        
        weights_q = np.round(weights / scale) + offset
        
        if self.dtype == 'int8':
            weights_q = np.clip(weights_q, -128, 127).astype(np.int8)
        elif self.dtype == 'int16':
            weights_q = np.clip(weights_q, -32768, 32767).astype(np.int16)
        
        return weights_q
    
    def _compute_scale_offset(self, weights: np.ndarray) -> tuple:
        """
        Compute scale and offset from weight statistics.
        
        Uses SYMMETRIC quantization (zero_point=0) which is safer for
        dynamic quantization because:
        1. The same scale is used for both weights and activations
        2. We need a scale that accommodates typical activation ranges too
        
        For symmetric quantization: scale = max(|min|, |max|) / (q_max)
        We also apply a safety factor of 2x to accommodate activation ranges.
        
        Args:
            weights: Float weights as numpy array
            
        Returns:
            Tuple of (scale, offset)
        """
        # Use symmetric quantization (offset=0)
        w_absmax = max(abs(float(np.min(weights))), abs(float(np.max(weights))))
        
        if self.dtype == 'int8':
            q_max = 127
        elif self.dtype == 'int16':
            q_max = 32767
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        
        # Apply safety factor of 2x to accommodate activation ranges
        # (activations typically have larger range than weights)
        safety_factor = 2.0
        
        # Avoid division by zero
        if w_absmax == 0:
            scale = 1.0 / q_max
        else:
            scale = (w_absmax * safety_factor) / q_max
        
        # Symmetric quantization: offset is always 0
        offset = 0
        
        return scale, offset
    
    def get_quant_params(self) -> Dict[str, Any]:
        """Get quantization parameters."""
        params = {
            'dtype': self.dtype,
            'strategy': 'dynamic_minmax_per_tensor'
        }
        if self._computed_scale is not None:
            params['scale'] = self._computed_scale
            params['offset'] = self._computed_offset
        return params
    
    def __repr__(self) -> str:
        return f"DynamicQuantRuleMinMaxPerTensor(pattern='{self.pattern}', dtype='{self.dtype}')"

