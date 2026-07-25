"""Quantization module for PyTorch to C compiler"""

from .rules import (
    QuantRule,
    StaticQuantRule,
    StaticPerChannelLinearQuantRule,
    StaticPerChannelConvQuantRule,
    StaticPerGroupLinearQuantRule,
    Int8WeightOnlyLinearRule,
    StaticInt4PerGroupLinearQuantRule,
    DynamicInt4PerGroupLinearQuantRule,
    PaletteWeightRule,
    DynamicQuantRuleMinMaxPerTensor,
    LQERDynamicQuantRule,
    LQERStaticQuantRule,
)
from .rule_matcher import RuleMatcher
from .graph_transform import QuantizationTransform
from .ops.quant_utils import QuantizeNode, DequantizeNode, DynamicQuantizeInputNode
from .ops.quant_linear import (
    StaticQuantLinearNode,
    StaticPerChannelQuantLinearNode,
    StaticPerGroupQuantLinearNode,
    Int8WeightOnlyLinearNode,
    PaletteWeightLinearNode,
    DynamicQuantLinearNode,
)
from .ops.quant_int4_linear import (
    StaticInt4PerGroupQuantLinearNode,
    DynamicInt4PerGroupQuantLinearNode,
)
from .ops.quant_conv2d import (
    StaticQuantConv2dNode,
    StaticPerChannelQuantConv2dNode,
    DynamicQuantConv2dNode,
)
from .ops.quant_LQER import (
    svd_factorizer,
    LQERMatmulNode,
    LQERConvMatmulNode,
    LQERAddNode,
    LQERDynamicQuantLinearNode,
    LQERStaticQuantLinearNode,
    LQERDynamicQuantConv2dNode,
    LQERStaticQuantConv2dNode,
)
from .quant_helpers import BLOCK_K
from .gptq import gptq_quantize

__all__ = [
    'QuantRule',
    'StaticQuantRule',
    'StaticPerChannelLinearQuantRule',
    'StaticPerChannelConvQuantRule',
    'StaticPerGroupLinearQuantRule',
    'Int8WeightOnlyLinearRule',
    'StaticInt4PerGroupLinearQuantRule',
    'DynamicInt4PerGroupLinearQuantRule',
    'PaletteWeightRule',
    'DynamicQuantRuleMinMaxPerTensor',
    'RuleMatcher',
    'QuantizationTransform',
    'QuantizeNode',
    'DequantizeNode',
    'DynamicQuantizeInputNode',
    'StaticQuantLinearNode',
    'StaticPerChannelQuantLinearNode',
    'StaticPerGroupQuantLinearNode',
    'Int8WeightOnlyLinearNode',
    'PaletteWeightLinearNode',
    'DynamicQuantLinearNode',
    'StaticInt4PerGroupQuantLinearNode',
    'DynamicInt4PerGroupQuantLinearNode',
    'StaticQuantConv2dNode',
    'StaticPerChannelQuantConv2dNode',
    'DynamicQuantConv2dNode',
    'LQERDynamicQuantRule',
    'LQERStaticQuantRule',
    'svd_factorizer',
    'LQERMatmulNode',
    'LQERConvMatmulNode',
    'LQERAddNode',
    'LQERDynamicQuantLinearNode',
    'LQERStaticQuantLinearNode',
    'LQERDynamicQuantConv2dNode',
    'LQERStaticQuantConv2dNode',
    'BLOCK_K',
    'gptq_quantize',
]
