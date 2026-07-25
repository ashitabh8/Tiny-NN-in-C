"""Tests for codegen backend registry."""

import warnings

import pytest

from src.pytorch_to_c.codegen.backend_registry import (
    TESTED_BACKENDS,
    check_backend,
    create_printer,
    get_printer_class,
    list_backends,
)
from src.pytorch_to_c.codegen.c_printer import CPrinter


def test_list_backends_includes_c_and_triton():
    names = list_backends()
    assert "c" in names
    assert "triton" in names


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown codegen backend"):
        check_backend("nonexistent")


def test_tested_backend_no_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_backend("c")
    assert not any(issubclass(w.category, UserWarning) for w in caught)


def test_untested_backend_warns():
    if "triton" in TESTED_BACKENDS:
        pytest.skip("triton already promoted to tested")
    with pytest.warns(UserWarning, match="NOT passed the project verification"):
        check_backend("triton")


def test_get_printer_class_c():
    assert get_printer_class("c") is CPrinter


def test_create_printer_c():
    from test.test_models import TinyMLP
    from src.pytorch_to_c.frontend.fx_tracer import trace_model
    from src.pytorch_to_c.lowering.lower import lower_fx_graph
    import torch

    model = TinyMLP(input_size=4, hidden_size=3, output_size=2)
    ir = lower_fx_graph(trace_model(model, torch.randn(1, 4)))
    printer = create_printer("c", ir)
    assert isinstance(printer, CPrinter)
