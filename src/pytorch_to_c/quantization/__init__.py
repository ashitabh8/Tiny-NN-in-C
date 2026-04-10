"""Quantization module for PyTorch to C compiler"""

from .rules import (
    QuantRule,
    StaticQuantRule,
    StaticPerChannelLinearQuantRule,
    StaticPerChannelConvQuantRule,
    DynamicQuantRuleMinMaxPerTensor,
)
from .rule_matcher import RuleMatcher
from .graph_transform import QuantizationTransform
from .ops.quant_utils import QuantizeNode, DequantizeNode, DynamicQuantizeInputNode
from .ops.quant_linear import (
    StaticQuantLinearNode,
    StaticPerChannelQuantLinearNode,
    DynamicQuantLinearNode,
)
from .ops.quant_conv2d import (
    StaticQuantConv2dNode,
    StaticPerChannelQuantConv2dNode,
    DynamicQuantConv2dNode,
)

__all__ = [
    'QuantRule',
    'StaticQuantRule',
    'StaticPerChannelLinearQuantRule',
    'StaticPerChannelConvQuantRule',
    'DynamicQuantRuleMinMaxPerTensor',
    'RuleMatcher',
    'QuantizationTransform',
    'QuantizeNode',
    'DequantizeNode',
    'DynamicQuantizeInputNode',
    'StaticQuantLinearNode',
    'StaticPerChannelQuantLinearNode',
    'DynamicQuantLinearNode',
    'StaticQuantConv2dNode',
    'StaticPerChannelQuantConv2dNode',
    'DynamicQuantConv2dNode',
]
