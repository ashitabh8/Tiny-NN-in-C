"""Tests for compression IR passes."""

import os
import sys

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import TinyMLP
from src.passes import PruneRule, StructuredPruningPass
from src.pytorch_to_c.compiler import compile_model
from tools.verify_model import verify_model, _gcc_available


def _skip_no_gcc():
    if not _gcc_available():
        pytest.skip("gcc not available")


class TestStructuredPruning:
    def test_prune_reduces_out_features(self):
        model = TinyMLP(input_size=20, hidden_size=10, output_size=5).eval()
        x = torch.randn(1, 20)
        ir = compile_model(model, x, return_ir=True, verbose=False)
        rules = [PruneRule(pattern=r"fc1", amount=0.3)]
        ir2 = StructuredPruningPass(rules).apply(ir)
        fc1 = next(n for n in ir2.nodes if n.name == "fc1")
        assert fc1.metadata["out_features"] < 10

    def test_prune_verify_c(self):
        _skip_no_gcc()
        model = TinyMLP(input_size=20, hidden_size=10, output_size=5).eval()
        x = torch.randn(1, 20)
        rules_prune = [PruneRule(pattern=r"fc1", amount=0.2)]
        res = verify_model(
            model, x, num_samples=5,
            passes=[StructuredPruningPass(rules_prune)],
            tolerance=2.0,
        )
        assert res.num_samples > 0
