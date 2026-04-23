"""End-to-end verification for DeepSenseDWCleanBackbone (float32).

Exercises the new Conv1d / BatchNorm1d / 3D-permute / 3D-mean codegen paths
end-to-end: trace -> lower -> codegen -> gcc -> compare to PyTorch.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src2', 'models'))

from tools.verify_model import verify_model, _gcc_available  # noqa: E402
from DeepSenseDWClean import DeepSenseDWCleanBackbone        # noqa: E402


class _LogitsOnly(torch.nn.Module):
    """The verify harness expects a tensor output; the backbone returns a dict."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)['logits']


def _skip_no_gcc():
    if not _gcc_available():
        pytest.skip("gcc not available")


class _Conv1dStandardOnly(torch.nn.Module):
    """Single Conv1d (standard, k=3) — exercises INT8 conv1d wrapper end-to-end."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(4, 6, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        return self.conv(x)


class _Conv1dDepthwiseOnly(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(8, 8, kernel_size=3, padding=1, groups=8, bias=False)

    def forward(self, x):
        return self.conv(x)


def test_conv1d_standard_float():
    _skip_no_gcc()
    model = _Conv1dStandardOnly().eval()
    res = verify_model(model, torch.randn(1, 4, 8), num_samples=10, tolerance=1e-3)
    assert res.failed == 0, res.summary()


def test_conv1d_standard_dynamic_int8():
    _skip_no_gcc()
    from src.pytorch_to_c.quantization import DynamicQuantRuleMinMaxPerTensor
    rules = [DynamicQuantRuleMinMaxPerTensor(pattern=r'.*conv.*', dtype='int8')]
    model = _Conv1dStandardOnly().eval()
    res = verify_model(model, torch.randn(1, 4, 8), num_samples=10, tolerance=0.5,
                       quantization_rules=rules)
    assert res.failed == 0, res.summary()


def test_conv1d_depthwise_dynamic_int8():
    _skip_no_gcc()
    from src.pytorch_to_c.quantization import DynamicQuantRuleMinMaxPerTensor
    rules = [DynamicQuantRuleMinMaxPerTensor(pattern=r'.*conv.*', dtype='int8')]
    model = _Conv1dDepthwiseOnly().eval()
    res = verify_model(model, torch.randn(1, 8, 16), num_samples=10, tolerance=0.5,
                       quantization_rules=rules)
    assert res.failed == 0, res.summary()


def _make_deepsense():
    backbone = DeepSenseDWCleanBackbone(
        in_channels=2,
        in_spectrum_len=16,
        num_classes=4,
        channels_freq=[8, 16],
        kernel_sizes_freq=[(3, 3), (3, 3)],
        strides_freq=[(1, 1), (1, 1)],
        temporal_channels=8,
        num_temporal_layers=2,
        temporal_kernel=3,
        fc_dim=16,
        dropout_ratio=0.0,
    )
    return _LogitsOnly(backbone).eval()


def test_deepsense_dw_clean_float():
    _skip_no_gcc()
    model = _make_deepsense()
    example_input = torch.randn(1, 2, 8, 16)
    res = verify_model(model, example_input, num_samples=20, tolerance=0.1)
    assert res.failed == 0, res.summary()
    summary = res.summary()
    assert "Top-1 class match: 20/20" in summary, summary


def test_deepsense_dw_clean_dynamic_int8_conv():
    """Full DeepSense backbone with dynamic INT8 on every conv (1d + 2d, depthwise + pointwise)."""
    _skip_no_gcc()
    from src.pytorch_to_c.quantization import DynamicQuantRuleMinMaxPerTensor
    rules = [DynamicQuantRuleMinMaxPerTensor(pattern=r'.*conv.*', dtype='int8')]
    model = _make_deepsense()
    example_input = torch.randn(1, 2, 8, 16)
    res = verify_model(model, example_input, num_samples=20, tolerance=2.0,
                       quantization_rules=rules)
    assert res.failed == 0, res.summary()
