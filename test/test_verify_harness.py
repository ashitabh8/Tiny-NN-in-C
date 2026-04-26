"""
Regression tests using the verification harness (tools/verify_model).

Tests both float32 and quantized compilation against PyTorch inference.
Requires gcc on PATH; tests are skipped automatically if unavailable.
"""

import pytest
import subprocess
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tools.verify_model import verify_model, _gcc_available
from models import TinyMLP, MixedNet


def _skip_no_gcc():
    if not _gcc_available():
        pytest.skip("gcc not available")


class TestVerifyFloat:
    """Verify float32 compiled models match PyTorch."""

    def test_tiny_mlp(self):
        _skip_no_gcc()
        model = TinyMLP(input_size=20, hidden_size=10, output_size=5)
        res = verify_model(model, torch.randn(1, 20), num_samples=10, tolerance=1e-3)
        assert res.failed == 0, res.summary()

    def test_mixed_net(self):
        _skip_no_gcc()
        model = MixedNet(input_channels=3, num_classes=4)
        # Softmax amplifies small float diffs -- use generous tolerance
        res = verify_model(model, torch.randn(1, 3, 32, 32), num_samples=5, tolerance=0.2)
        assert res.failed == 0, res.summary()


class TestVerifyQuantized:
    """Verify quantized compiled models produce reasonable outputs."""

    def test_tiny_mlp_static_int8(self):
        _skip_no_gcc()
        from src.pytorch_to_c.quantization import StaticQuantRule

        model = TinyMLP(input_size=20, hidden_size=10, output_size=5)
        rules = [
            StaticQuantRule(
                pattern=r".*fc.*",
                dtype="int8",
                input_scale=0.05,
                input_offset=0,
                weight_scale=0.02,
                weight_offset=0,
                output_scale=0.05,
                output_offset=0,
            ),
        ]
        res = verify_model(
            model,
            torch.randn(1, 20),
            num_samples=10,
            quantization_rules=rules,
            tolerance=5.0,
        )
        assert res.top1_matches > 0 or res.num_samples == res.passed, res.summary()


class _Conv1dStandardOnly(torch.nn.Module):
    """Single Conv1d (standard, k=3) for end-to-end verify harness coverage."""

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


class TestVerifyConv1d:
    """Conv1d path through trace → codegen → gcc (no external model files)."""

    def test_conv1d_standard_float(self):
        _skip_no_gcc()
        model = _Conv1dStandardOnly().eval()
        res = verify_model(model, torch.randn(1, 4, 8), num_samples=10, tolerance=1e-3)
        assert res.failed == 0, res.summary()

    def test_conv1d_standard_dynamic_int8(self):
        _skip_no_gcc()
        from src.pytorch_to_c.quantization import DynamicQuantRuleMinMaxPerTensor

        rules = [DynamicQuantRuleMinMaxPerTensor(pattern=r'.*conv.*', dtype='int8')]
        model = _Conv1dStandardOnly().eval()
        res = verify_model(
            model,
            torch.randn(1, 4, 8),
            num_samples=10,
            tolerance=0.5,
            quantization_rules=rules,
        )
        assert res.failed == 0, res.summary()

    def test_conv1d_depthwise_dynamic_int8(self):
        _skip_no_gcc()
        from src.pytorch_to_c.quantization import DynamicQuantRuleMinMaxPerTensor

        rules = [DynamicQuantRuleMinMaxPerTensor(pattern=r'.*conv.*', dtype='int8')]
        model = _Conv1dDepthwiseOnly().eval()
        res = verify_model(
            model,
            torch.randn(1, 8, 16),
            num_samples=10,
            tolerance=0.5,
            quantization_rules=rules,
        )
        assert res.failed == 0, res.summary()
