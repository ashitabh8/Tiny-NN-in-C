"""Canonical model definitions for Tiny-NN-in-C examples and tests."""

from .tiny_mlp import TinyMLP
from .mixed_net import MixedNet
from .custom_mlp import CustomMLP
from .tiny_resnet import ResBlock1D, TinyResNet1D
from .mnist_cnn import MNISTConvNet
from .resnet_block import ResNetBlock
from .resnet_arch_early_exit import SimpleResNet, BasicBlock, BasicBlockDown

__all__ = [
    "TinyMLP",
    "MixedNet",
    "CustomMLP",
    "ResBlock1D",
    "TinyResNet1D",
    "MNISTConvNet",
    "ResNetBlock",
    "SimpleResNet",
    "BasicBlock",
    "BasicBlockDown",
]
