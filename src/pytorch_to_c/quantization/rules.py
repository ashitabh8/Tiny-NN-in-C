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
        return self._compiled_pattern.fullmatch(node.name) is not None
    
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
    def quantize_weights(self, weights: np.ndarray, **kwargs) -> np.ndarray:
        """
        Quantize weights during compilation.
        
        Args:
            weights: Float weights as numpy array
            **kwargs: Optional context. Per-channel rules may pass
                ``ir_graph`` and ``quant_node`` to register scale arrays.
            
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
                input_offset=self.input_offset,
                weight_offset=self.weight_offset,
                output_offset=self.output_offset
            )
        elif node.op_type in ('conv2d', 'conv1d'):
            from .ops.quant_conv2d import StaticQuantConv2dNode
            return StaticQuantConv2dNode(
                original_node=node,
                dtype=self.dtype,
                input_scale=self.input_scale,
                weight_scale=self.weight_scale,
                output_scale=self.output_scale,
                input_offset=self.input_offset,
                weight_offset=self.weight_offset,
                output_offset=self.output_offset
            )
        else:
            raise ValueError(
                f"Cannot quantize operation '{node.op_type}' for node '{node.name}'. "
                f"Quantized version not implemented."
            )
    
    def quantize_weights(self, weights: np.ndarray, **kwargs) -> np.ndarray:
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


def _q_max_for_dtype(dtype: str) -> float:
    if dtype == 'int8':
        return 127.0
    if dtype == 'int16':
        return 32767.0
    raise ValueError(f"Unsupported dtype: {dtype}")


def _symmetric_scales_last_axis(weights: np.ndarray, q_max: float) -> np.ndarray:
    """
    One scale per index along the last axis (output feature / output channel).

    Symmetric range: scale[o] = max(|W[..., o]|) / q_max, or 1/q_max if empty.
    """
    oc = int(weights.shape[-1])
    scales = np.empty(oc, dtype=np.float64)
    for o in range(oc):
        sl = weights[..., o]
        amax = float(np.max(np.abs(sl))) if sl.size else 0.0
        scales[o] = (amax / q_max) if amax > 0.0 else (1.0 / q_max)
    return scales


def _quantize_affine_per_last_axis(
    weights: np.ndarray,
    scales_1d: np.ndarray,
    weight_offset: int,
    dtype: str,
) -> np.ndarray:
    oc = weights.shape[-1]
    assert scales_1d.shape == (oc,), (scales_1d.shape, weights.shape)
    acc = np.zeros_like(weights, dtype=np.float64)
    for o in range(oc):
        acc[..., o] = np.round(weights[..., o].astype(np.float64) / scales_1d[o]) + weight_offset
    if dtype == 'int8':
        return np.clip(acc, -128, 127).astype(np.int8)
    if dtype == 'int16':
        return np.clip(acc, -32768, 32767).astype(np.int16)
    raise ValueError(f"Unsupported dtype: {dtype}")


class StaticPerChannelLinearQuantRule(QuantRule):
    """
    Static activation quantization with **symmetric per-output-column weight scales**
    computed from float weights at compile time (absmax / q_max per column).

    Activation scales/zero-points are fixed like :class:`StaticQuantRule`. A single
    ``weight_offset`` applies to all columns (same as the C per-channel kernels).

    Registers ``{weight_name}_per_channel_scales`` in ``ir_graph.parameters`` and
    sets ``metadata['per_channel_weight_scales_param']`` on the quant node for codegen.
    """

    def __init__(
        self,
        pattern: str,
        dtype: str,
        input_scale: float,
        input_offset: int,
        output_scale: float,
        output_offset: int,
        weight_offset: int = 0,
    ):
        super().__init__(pattern, dtype)
        self.input_scale = input_scale
        self.input_offset = input_offset
        self.output_scale = output_scale
        self.output_offset = output_offset
        self.weight_offset = weight_offset

    def create_quant_node(self, node):
        if node.op_type != 'linear':
            raise ValueError(
                f"StaticPerChannelLinearQuantRule only supports linear, got '{node.op_type}'."
            )
        from .ops.quant_linear import StaticPerChannelQuantLinearNode

        return StaticPerChannelQuantLinearNode(
            original_node=node,
            dtype=self.dtype,
            input_scale=self.input_scale,
            output_scale=self.output_scale,
            input_offset=self.input_offset,
            weight_offset=self.weight_offset,
            output_offset=self.output_offset,
        )

    def quantize_weights(self, weights: np.ndarray, **kwargs) -> np.ndarray:
        ir_graph = kwargs.get('ir_graph')
        quant_node = kwargs.get('quant_node')
        if ir_graph is None or quant_node is None:
            raise ValueError(
                "StaticPerChannelLinearQuantRule.quantize_weights requires "
                "ir_graph= and quant_node= (provided by QuantizationTransform)."
            )
        wn = quant_node.metadata.get('weight_name')
        if not wn:
            raise ValueError("Quant node missing metadata['weight_name']")

        q_max = _q_max_for_dtype(self.dtype)
        scales = _symmetric_scales_last_axis(weights, q_max).astype(np.float32)
        param_name = f"{wn}_per_channel_scales"
        ir_graph.parameters[param_name] = scales
        quant_node.metadata['per_channel_weight_scales_param'] = param_name
        quant_node.scale = float(np.mean(scales))
        quant_node.metadata['quant_params']['scale'] = quant_node.scale

        return _quantize_affine_per_last_axis(
            weights, scales.astype(np.float64), self.weight_offset, self.dtype
        )

    def get_quant_params(self) -> Dict[str, Any]:
        return {
            'dtype': self.dtype,
            'strategy': 'static_per_channel_linear',
            'input_scale': self.input_scale,
            'input_offset': self.input_offset,
            'weight_offset': self.weight_offset,
            'output_scale': self.output_scale,
            'output_offset': self.output_offset,
        }

    def __repr__(self) -> str:
        return (
            f"StaticPerChannelLinearQuantRule(pattern='{self.pattern}', dtype='{self.dtype}', "
            f"input_scale={self.input_scale}, output_scale={self.output_scale})"
        )


class StaticPerChannelConvQuantRule(QuantRule):
    """
    Same as :class:`StaticPerChannelLinearQuantRule` but for ``conv2d`` weights in
    ``[k_h, k_w, in_c, out_c]`` (or depthwise ``[k_h, k_w, c]``): one scale per
    output channel along the last axis.

    Emits ``conv2d_nhwc_*_per_channel`` or ``depthwise_conv2d_nhwc_*_per_channel``.
    """

    def __init__(
        self,
        pattern: str,
        dtype: str,
        input_scale: float,
        input_offset: int,
        output_scale: float,
        output_offset: int,
        weight_offset: int = 0,
    ):
        super().__init__(pattern, dtype)
        self.input_scale = input_scale
        self.input_offset = input_offset
        self.output_scale = output_scale
        self.output_offset = output_offset
        self.weight_offset = weight_offset

    def create_quant_node(self, node):
        if node.op_type not in ('conv2d', 'conv1d'):
            raise ValueError(
                f"StaticPerChannelConvQuantRule only supports conv2d/conv1d, got '{node.op_type}'."
            )
        from .ops.quant_conv2d import StaticPerChannelQuantConv2dNode

        return StaticPerChannelQuantConv2dNode(
            original_node=node,
            dtype=self.dtype,
            input_scale=self.input_scale,
            output_scale=self.output_scale,
            input_offset=self.input_offset,
            weight_offset=self.weight_offset,
            output_offset=self.output_offset,
        )

    def quantize_weights(self, weights: np.ndarray, **kwargs) -> np.ndarray:
        ir_graph = kwargs.get('ir_graph')
        quant_node = kwargs.get('quant_node')
        if ir_graph is None or quant_node is None:
            raise ValueError(
                "StaticPerChannelConvQuantRule.quantize_weights requires "
                "ir_graph= and quant_node= (provided by QuantizationTransform)."
            )
        wn = quant_node.metadata.get('weight_name')
        if not wn:
            raise ValueError("Quant node missing metadata['weight_name']")

        q_max = _q_max_for_dtype(self.dtype)
        scales = _symmetric_scales_last_axis(weights, q_max).astype(np.float32)
        param_name = f"{wn}_per_channel_scales"
        ir_graph.parameters[param_name] = scales
        quant_node.metadata['per_channel_weight_scales_param'] = param_name
        quant_node.scale = float(np.mean(scales))
        quant_node.metadata['quant_params']['scale'] = quant_node.scale

        return _quantize_affine_per_last_axis(
            weights, scales.astype(np.float64), self.weight_offset, self.dtype
        )

    def get_quant_params(self) -> Dict[str, Any]:
        return {
            'dtype': self.dtype,
            'strategy': 'static_per_channel_conv',
            'input_scale': self.input_scale,
            'input_offset': self.input_offset,
            'weight_offset': self.weight_offset,
            'output_scale': self.output_scale,
            'output_offset': self.output_offset,
        }

    def __repr__(self) -> str:
        return (
            f"StaticPerChannelConvQuantRule(pattern='{self.pattern}', dtype='{self.dtype}', "
            f"input_scale={self.input_scale}, output_scale={self.output_scale})"
        )


class StaticPerGroupLinearQuantRule(QuantRule):
    """
    Per-group weight quantization along the input axis with tile-aligned group sizes.
    """

    def __init__(
        self,
        pattern: str,
        dtype: str,
        input_scale: float,
        input_offset: int,
        output_scale: float,
        output_offset: int,
        weight_offset: int = 0,
        group_size: int | str = "auto",
        error_budget: Optional[float] = None,
        rounding: str = "nearest",
        calibration=None,
    ):
        super().__init__(pattern, dtype)
        self.input_scale = input_scale
        self.input_offset = input_offset
        self.output_scale = output_scale
        self.output_offset = output_offset
        self.weight_offset = weight_offset
        self.group_size = group_size
        self.error_budget = error_budget
        self.rounding = rounding
        self.calibration = calibration

    def create_quant_node(self, node):
        if node.op_type != "linear":
            raise ValueError(
                f"StaticPerGroupLinearQuantRule only supports linear, got '{node.op_type}'."
            )
        from .ops.quant_linear import StaticPerGroupQuantLinearNode

        return StaticPerGroupQuantLinearNode(
            original_node=node,
            dtype=self.dtype,
            input_scale=self.input_scale,
            output_scale=self.output_scale,
            input_offset=self.input_offset,
            weight_offset=self.weight_offset,
            output_offset=self.output_offset,
        )

    def quantize_weights(self, weights: np.ndarray, **kwargs) -> np.ndarray:
        from .quant_helpers import (
            quantize_affine_per_group,
            select_group_size,
            symmetric_scales_per_group,
        )
        from .gptq import gptq_quantize

        ir_graph = kwargs.get("ir_graph")
        quant_node = kwargs.get("quant_node")
        if ir_graph is None or quant_node is None:
            raise ValueError("StaticPerGroupLinearQuantRule requires ir_graph and quant_node")

        wn = quant_node.metadata.get("weight_name")
        g = select_group_size(
            weights, self.dtype, self.group_size, self.error_budget
        )
        quant_node.metadata["group_size"] = g

        if self.rounding == "gptq":
            hessian = None
            if self.calibration is not None:
                hessian = self.calibration.hessians.get(quant_node.name)
            wq = gptq_quantize(
                weights, hessian, g, self.dtype, self.weight_offset
            )
            scales = symmetric_scales_per_group(
                weights, g, self.dtype, self.weight_offset
            )
        else:
            scales = symmetric_scales_per_group(
                weights, g, self.dtype, self.weight_offset
            )
            wq = quantize_affine_per_group(
                weights, scales, g, self.weight_offset, self.dtype
            )

        param_name = f"{wn}_per_group_scales"
        ir_graph.parameters[param_name] = scales.astype(np.float32).reshape(-1)
        quant_node.metadata["per_group_weight_scales_param"] = param_name
        quant_node.metadata["num_groups"] = weights.shape[0] // g
        quant_node.scale = float(np.mean(scales))
        return wq

    def get_quant_params(self) -> Dict[str, Any]:
        return {
            "dtype": self.dtype,
            "strategy": "static_per_group_linear",
            "group_size": self.group_size,
            "rounding": self.rounding,
        }


class Int8WeightOnlyLinearRule(QuantRule):
    """Int8 weight-only linear with per-output-column symmetric scales (float activations)."""

    def __init__(self, pattern: str, weight_offset: int = 0):
        super().__init__(pattern, "int8")
        self.weight_offset = weight_offset

    def create_quant_node(self, node):
        if node.op_type != "linear":
            raise ValueError(
                f"Int8WeightOnlyLinearRule only supports linear, got '{node.op_type}'."
            )
        from .ops.quant_linear import Int8WeightOnlyLinearNode

        return Int8WeightOnlyLinearNode(original_node=node)

    def quantize_weights(self, weights: np.ndarray, **kwargs) -> np.ndarray:
        ir_graph = kwargs.get("ir_graph")
        quant_node = kwargs.get("quant_node")
        if ir_graph is None or quant_node is None:
            raise ValueError(
                "Int8WeightOnlyLinearRule.quantize_weights requires "
                "ir_graph= and quant_node="
            )
        wn = quant_node.metadata.get("weight_name")
        if not wn:
            raise ValueError("Quant node missing metadata['weight_name']")

        q_max = _q_max_for_dtype(self.dtype)
        scales = _symmetric_scales_last_axis(weights, q_max).astype(np.float32)
        param_name = f"{wn}_per_channel_scales"
        ir_graph.parameters[param_name] = scales
        quant_node.metadata["per_channel_weight_scales_param"] = param_name
        quant_node.scale = float(np.mean(scales))
        quant_node.metadata["quant_params"]["scale"] = quant_node.scale

        return _quantize_affine_per_last_axis(
            weights, scales.astype(np.float64), self.weight_offset, self.dtype
        )

    def get_quant_params(self) -> Dict[str, Any]:
        return {"dtype": self.dtype, "strategy": "int8_weight_only"}


def _pack_int4_per_group_weights(
    weights: np.ndarray,
    *,
    group_size: int,
    weight_offset: int,
    rounding: str,
    calibration,
    ir_graph,
    quant_node,
) -> np.ndarray:
    """Shared int4 per-group weight packing for static/dynamic int4 rules."""
    from .quant_helpers import (
        pack_int4_nibbles,
        quantize_affine_per_group,
        symmetric_scales_per_group,
    )
    from .gptq import gptq_quantize

    wn = quant_node.metadata.get("weight_name")
    if not wn:
        raise ValueError("Quant node missing metadata['weight_name']")

    g = group_size
    in_features = weights.shape[0]
    if in_features % g != 0:
        g = in_features
    quant_node.metadata["group_size"] = g

    if rounding == "gptq":
        hessian = None
        if calibration is not None:
            hessian = calibration.hessians.get(quant_node.name)
        wq = gptq_quantize(weights, hessian, g, "int8", weight_offset)
        wq = np.clip(wq, -8, 7).astype(np.int8)
        scales = symmetric_scales_per_group(weights, g, "int4", weight_offset)
    else:
        scales = symmetric_scales_per_group(weights, g, "int4", weight_offset)
        wq = quantize_affine_per_group(weights, scales, g, weight_offset, "int8")
        wq = np.clip(wq, -8, 7).astype(np.int8)

    packed = pack_int4_nibbles(wq)
    param_name = f"{wn}_per_group_scales"
    ir_graph.parameters[param_name] = scales.astype(np.float32).reshape(-1)
    quant_node.metadata["per_group_weight_scales_param"] = param_name
    quant_node.metadata["packed_int4"] = True
    quant_node.metadata["original_weight_shape"] = list(weights.shape)
    quant_node.metadata["packed_weight_count"] = int(wq.size)
    quant_node.scale = float(np.mean(scales))
    if "quant_params" in quant_node.metadata:
        quant_node.metadata["quant_params"]["scale"] = quant_node.scale
    return packed


class StaticInt4PerGroupLinearQuantRule(QuantRule):
    """
    Static W4A8 linear: int8 activations + packed int4 weights (per-group scales).

    Inserts QuantizeNode / DequantizeNode like other static rules.
    """

    def __init__(
        self,
        pattern: str,
        input_scale: float,
        input_offset: int,
        output_scale: float,
        output_offset: int,
        group_size: int = 64,
        weight_offset: int = 0,
        rounding: str = "nearest",
        calibration=None,
    ):
        super().__init__(pattern, "int8")
        self.input_scale = input_scale
        self.input_offset = input_offset
        self.output_scale = output_scale
        self.output_offset = output_offset
        self.group_size = group_size
        self.weight_offset = weight_offset
        self.rounding = rounding
        self.calibration = calibration

    def create_quant_node(self, node):
        if node.op_type != "linear":
            raise ValueError(
                f"StaticInt4PerGroupLinearQuantRule only supports linear, got '{node.op_type}'"
            )
        from .ops.quant_int4_linear import StaticInt4PerGroupQuantLinearNode

        return StaticInt4PerGroupQuantLinearNode(
            original_node=node,
            input_scale=self.input_scale,
            output_scale=self.output_scale,
            input_offset=self.input_offset,
            weight_offset=self.weight_offset,
            output_offset=self.output_offset,
            group_size=self.group_size,
        )

    def quantize_weights(self, weights: np.ndarray, **kwargs) -> np.ndarray:
        ir_graph = kwargs.get("ir_graph")
        quant_node = kwargs.get("quant_node")
        if ir_graph is None or quant_node is None:
            raise ValueError(
                "StaticInt4PerGroupLinearQuantRule requires ir_graph and quant_node"
            )
        return _pack_int4_per_group_weights(
            weights,
            group_size=self.group_size,
            weight_offset=self.weight_offset,
            rounding=self.rounding,
            calibration=self.calibration,
            ir_graph=ir_graph,
            quant_node=quant_node,
        )

    def get_quant_params(self) -> Dict[str, Any]:
        return {
            "dtype": "int8",
            "strategy": "static_int4_per_group",
            "group_size": self.group_size,
            "rounding": self.rounding,
            "input_scale": self.input_scale,
            "output_scale": self.output_scale,
        }


class DynamicInt4PerGroupLinearQuantRule(QuantRule):
    """
    Dynamic W4A8 linear: runtime activation scale + packed int4 weights.

    Inserts DynamicQuantizeInputNode; float-output kernel (no DequantizeNode).
    """

    def __init__(
        self,
        pattern: str,
        group_size: int = 64,
        weight_offset: int = 0,
        rounding: str = "nearest",
        calibration=None,
    ):
        super().__init__(pattern, "int8")
        self.group_size = group_size
        self.weight_offset = weight_offset
        self.rounding = rounding
        self.calibration = calibration

    def create_quant_node(self, node):
        raise NotImplementedError(
            "DynamicInt4PerGroupLinearQuantRule.create_quant_node requires "
            "weights. Use QuantizationTransform (create_quant_node_with_weights)."
        )

    def create_quant_node_with_weights(self, node, weights: np.ndarray):
        if node.op_type != "linear":
            raise ValueError(
                f"DynamicInt4PerGroupLinearQuantRule only supports linear, got '{node.op_type}'"
            )
        from .ops.quant_int4_linear import DynamicInt4PerGroupQuantLinearNode

        return DynamicInt4PerGroupQuantLinearNode(
            original_node=node,
            group_size=self.group_size,
            weight_offset=self.weight_offset,
        )

    def quantize_weights(self, weights: np.ndarray, **kwargs) -> np.ndarray:
        ir_graph = kwargs.get("ir_graph")
        quant_node = kwargs.get("quant_node")
        if ir_graph is None or quant_node is None:
            raise ValueError(
                "DynamicInt4PerGroupLinearQuantRule requires ir_graph and quant_node"
            )
        return _pack_int4_per_group_weights(
            weights,
            group_size=self.group_size,
            weight_offset=self.weight_offset,
            rounding=self.rounding,
            calibration=self.calibration,
            ir_graph=ir_graph,
            quant_node=quant_node,
        )

    def get_quant_params(self) -> Dict[str, Any]:
        return {
            "dtype": "int8",
            "strategy": "dynamic_int4_per_group",
            "group_size": self.group_size,
            "rounding": self.rounding,
        }


class PaletteWeightRule(QuantRule):
    """Weight palettization via k-means codebook (weight-only, float activations)."""

    def __init__(self, pattern: str, num_centroids: int = 16):
        super().__init__(pattern, "palette")
        self.num_centroids = num_centroids

    def create_quant_node(self, node):
        if node.op_type != "linear":
            raise ValueError("PaletteWeightRule only supports linear")
        from .ops.quant_linear import PaletteWeightLinearNode

        return PaletteWeightLinearNode(
            original_node=node, num_centroids=self.num_centroids
        )

    def quantize_weights(self, weights: np.ndarray, **kwargs) -> np.ndarray:
        from .quant_helpers import kmeans_1d, pack_palette_indices

        ir_graph = kwargs.get("ir_graph")
        quant_node = kwargs.get("quant_node")
        if ir_graph is None or quant_node is None:
            raise ValueError("PaletteWeightRule requires ir_graph and quant_node")

        wn = quant_node.metadata.get("weight_name")
        codebook, indices = kmeans_1d(weights, self.num_centroids)
        packed = pack_palette_indices(indices, self.num_centroids)

        cb_name = f"{wn}_codebook"
        idx_name = f"{wn}_indices"
        ir_graph.parameters[cb_name] = codebook
        ir_graph.parameters[idx_name] = packed
        quant_node.metadata["codebook_param"] = cb_name
        quant_node.metadata["indices_param"] = idx_name
        quant_node.metadata["num_centroids"] = self.num_centroids
        quant_node.metadata["original_weight_shape"] = list(weights.shape)
        quant_node.metadata["weight_count"] = int(weights.size)
        return packed

    def get_quant_params(self) -> Dict[str, Any]:
        return {
            "dtype": "palette",
            "strategy": "palettization",
            "num_centroids": self.num_centroids,
        }


def _lqer_error_for_node(rule_name: str, error_matrix, node) -> np.ndarray:
    """Resolve the error matrix for a node: single array or {name: array} dict."""
    if isinstance(error_matrix, dict):
        if node.name not in error_matrix:
            raise ValueError(
                f"{rule_name}: no error matrix provided for matched node "
                f"'{node.name}'. Available: {sorted(error_matrix.keys())}"
            )
        return error_matrix[node.name]
    return error_matrix


class LQERStaticQuantRule(StaticQuantRule):
    """
    StaticQuantRule + LQER low-rank error correction (see ops/quant_LQER.py).

    The user supplies the full weight-quantization error matrix E and a rank;
    the node factorizes E at compile time (default SVD, pluggable) and the
    correction branch runs in parallel with the quantized path, joined by a
    float add after the dequantize.

    error_matrix: ndarray (pattern matches one layer) or {node_name: ndarray}.
    Linear layout: [in_features, out_features]. Conv2d layout: HWIO
    [kh, kw, in_c, out_c] or flattened [kh*kw*in_c, out_c].
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
        output_offset: int,
        error_matrix=None,
        rank: int = 8,
        factorizer=None,
    ):
        super().__init__(
            pattern, dtype,
            input_scale, input_offset,
            weight_scale, weight_offset,
            output_scale, output_offset,
        )
        if error_matrix is None:
            raise ValueError("LQERStaticQuantRule requires error_matrix")
        self.error_matrix = error_matrix
        self.rank = rank
        self.factorizer = factorizer

    def create_quant_node(self, node):
        error = _lqer_error_for_node("LQERStaticQuantRule", self.error_matrix, node)
        if node.op_type == 'linear':
            from .ops.quant_LQER import LQERStaticQuantLinearNode
            return LQERStaticQuantLinearNode(
                original_node=node,
                dtype=self.dtype,
                input_scale=self.input_scale,
                weight_scale=self.weight_scale,
                output_scale=self.output_scale,
                error_matrix=error,
                rank=self.rank,
                input_offset=self.input_offset,
                weight_offset=self.weight_offset,
                output_offset=self.output_offset,
                factorizer=self.factorizer,
            )
        elif node.op_type == 'conv2d':
            from .ops.quant_LQER import LQERStaticQuantConv2dNode
            return LQERStaticQuantConv2dNode(
                original_node=node,
                dtype=self.dtype,
                input_scale=self.input_scale,
                weight_scale=self.weight_scale,
                output_scale=self.output_scale,
                error_matrix=error,
                rank=self.rank,
                input_offset=self.input_offset,
                weight_offset=self.weight_offset,
                output_offset=self.output_offset,
                factorizer=self.factorizer,
            )
        else:
            raise ValueError(
                f"LQERStaticQuantRule supports linear and conv2d, got "
                f"'{node.op_type}' for node '{node.name}'"
            )

    def __repr__(self) -> str:
        return (f"LQERStaticQuantRule(pattern='{self.pattern}', dtype='{self.dtype}', "
                f"rank={self.rank})")


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
        elif node.op_type in ('conv2d', 'conv1d'):
            from .ops.quant_conv2d import DynamicQuantConv2dNode
            return DynamicQuantConv2dNode(
                original_node=node,
                dtype=self.dtype,
                weight_scale=self._computed_scale,
                offset=self._computed_offset
            )
        else:
            raise ValueError(f"Cannot quantize {node.op_type}")
    
    def quantize_weights(self, weights: np.ndarray, **kwargs) -> np.ndarray:
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
        
        Uses SYMMETRIC quantization (zero_point=0):
          scale = max(|w_min|, |w_max|) / q_max
        
        Activation scale is computed independently at runtime by
        compute_dynamic_scale_int8/int16 in the C headers.
        
        Args:
            weights: Float weights as numpy array
            
        Returns:
            Tuple of (scale, offset)
        """
        w_absmax = max(abs(float(np.min(weights))), abs(float(np.max(weights))))
        
        if self.dtype == 'int8':
            q_max = 127
        elif self.dtype == 'int16':
            q_max = 32767
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        
        if w_absmax == 0:
            scale = 1.0 / q_max
        else:
            scale = w_absmax / q_max
        
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


class LQERDynamicQuantRule(DynamicQuantRuleMinMaxPerTensor):
    """
    DynamicQuantRuleMinMaxPerTensor + LQER low-rank error correction
    (see ops/quant_LQER.py).

    The quantized path uses the float-output dynamic kernels, so the join is
    a plain float add of the layer output and the low-rank correction, and
    both chains run in parallel (OpenMP sections in the C backend).

    error_matrix: ndarray (pattern matches one layer) or {node_name: ndarray}.
    Linear layout: [in_features, out_features]. Conv2d layout: HWIO
    [kh, kw, in_c, out_c] or flattened [kh*kw*in_c, out_c].
    """

    def __init__(
        self,
        pattern: str,
        dtype: str,
        error_matrix=None,
        rank: int = 8,
        factorizer=None,
    ):
        super().__init__(pattern, dtype)
        if error_matrix is None:
            raise ValueError("LQERDynamicQuantRule requires error_matrix")
        self.error_matrix = error_matrix
        self.rank = rank
        self.factorizer = factorizer

    def create_quant_node_with_weights(self, node, weights: np.ndarray):
        self._computed_scale, self._computed_offset = self._compute_scale_offset(weights)
        error = _lqer_error_for_node("LQERDynamicQuantRule", self.error_matrix, node)

        if node.op_type == 'linear':
            from .ops.quant_LQER import LQERDynamicQuantLinearNode
            return LQERDynamicQuantLinearNode(
                original_node=node,
                dtype=self.dtype,
                weight_scale=self._computed_scale,
                error_matrix=error,
                rank=self.rank,
                offset=self._computed_offset,
                factorizer=self.factorizer,
            )
        elif node.op_type == 'conv2d':
            from .ops.quant_LQER import LQERDynamicQuantConv2dNode
            return LQERDynamicQuantConv2dNode(
                original_node=node,
                dtype=self.dtype,
                weight_scale=self._computed_scale,
                error_matrix=error,
                rank=self.rank,
                offset=self._computed_offset,
                factorizer=self.factorizer,
            )
        else:
            raise ValueError(
                f"LQERDynamicQuantRule supports linear and conv2d, got "
                f"'{node.op_type}' for node '{node.name}'"
            )

    def get_quant_params(self) -> Dict[str, Any]:
        params = super().get_quant_params()
        params['strategy'] = 'lqer_dynamic_minmax_per_tensor'
        params['lqer_rank'] = self.rank
        return params

    def __repr__(self) -> str:
        return (f"LQERDynamicQuantRule(pattern='{self.pattern}', dtype='{self.dtype}', "
                f"rank={self.rank})")

