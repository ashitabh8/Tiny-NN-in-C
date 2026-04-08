"""Test model definitions -- imports canonical models from the models/ package."""

import torch
from models import TinyMLP, ResNetBlock, MixedNet


def get_test_models():
    """Get all test models with example inputs.

    Returns:
        List of tuples (model_name, model, example_input)
    """
    return [
        ("TinyMLP", TinyMLP(), torch.randn(1, 784)),
        ("ResNetBlock", ResNetBlock(), torch.randn(1, 64, 32, 32)),
        ("MixedNet", MixedNet(), torch.randn(1, 3, 32, 32)),
    ]
