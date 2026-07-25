"""Tests for calibration module."""

import sys
import os

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import TinyMLP
from src.pytorch_to_c.calibration import calibrate, make_static_rules, ActivationObserver


class TestCalibration:
    def test_observer_tracks_range(self):
        obs = ActivationObserver("fc1")
        obs.observe(torch.tensor([-2.0, 0.5, 3.0]))
        obs.stats.finalize_percentile()
        assert obs.stats.min_val == pytest.approx(-2.0)
        assert obs.stats.max_val == pytest.approx(3.0)
        scale = obs.symmetric_scale("int8")
        assert scale > 0

    def test_calibrate_tiny_mlp(self):
        model = TinyMLP(input_size=20, hidden_size=8, output_size=4).eval()
        data = [torch.randn(1, 20) for _ in range(5)]
        stats = calibrate(model, data, collect_hessian=True, max_batches=5)
        assert stats.num_batches == 5
        assert len(stats.layer_stats) >= 2

    def test_make_static_rules(self):
        model = TinyMLP(input_size=20, hidden_size=8, output_size=4).eval()
        stats = calibrate(model, [torch.randn(1, 20)], max_batches=1)
        rules = make_static_rules(stats, granularity="per_group")
        assert len(rules) >= 1
        assert any("PerGroup" in type(r).__name__ for r in rules)
