#!/usr/bin/env python3
"""
Test shape inference implementation
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.pytorch_to_c.frontend.fx_tracer import trace_model
from src.pytorch_to_c.lowering.lower import lower_fx_graph
from src.pytorch_to_c.compiler import compile_model


class TestModel(nn.Module):
    """Test model with different shapes."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 5)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def main():
    print("=" * 70)
    print("Testing Shape Inference")
    print("=" * 70)
    print()
    
    # Create model
    model = TestModel()
    model.eval()
    
    # Create input
    example_input = torch.randn(1, 20)
    
    print(f"Input shape: {example_input.shape}")
    print()
    
    # Step 1: Trace
    print("[1] Tracing model...")
    fx_graph = trace_model(model, example_input)
    print(f"  ✓ Traced {len(list(fx_graph.graph.nodes))} nodes")
    print()
    
    # Step 2: Lower with shape inference
    print("[2] Lowering with shape inference...")
    ir_graph = lower_fx_graph(fx_graph, example_input)
    print(f"  ✓ Created {len(ir_graph.nodes)} IR nodes")
    print()
    
    # Check shapes
    print("[3] Verifying shapes:")
    print()
    
    expected_shapes = {
        'x': (1, 20),
        'fc1': (1, 10),
        'relu': (1, 10),
        'fc2': (1, 5),
    }
    
    for node in ir_graph.nodes:
        shape = node.output_shape
        expected = expected_shapes.get(node.name, None)
        
        if shape:
            status = "✓" if shape == expected else "⚠"
            print(f"  {status} {node.name:10s} [{node.op_type:10s}] shape={shape}")
            
            if expected and shape != expected:
                print(f"      Expected: {expected}")
        else:
            print(f"  ✗ {node.name:10s} [{node.op_type:10s}] shape=None (missing!)")
    
    print()
    
    # Step 3: Test compilation with shapes
    print("[4] Testing full compilation...")
    os.makedirs("tmp", exist_ok=True)
    ir_graph = compile_model(model, example_input, "tmp", verbose=False)
    print(f"  ✓ Compilation successful")
    print()
    
    # Check generated code
    print("[5] Checking generated code:")
    with open("tmp/model.c", "r") as f:
        model_c = f.read()
    
    import re
    buffers = re.findall(r'float buf_(\w+)\[(\d+)\];', model_c)
    
    expected_sizes = {
        'fc1': 10,
        'relu': 10,
        'fc2': 5,
    }
    
    for buf_name, size_str in buffers:
        size = int(size_str)
        expected_size = expected_sizes.get(buf_name, None)
        
        if expected_size:
            status = "✓" if size == expected_size else "✗"
            print(f"  {status} buf_{buf_name}: size={size} (expected {expected_size})")
        else:
            print(f"  ? buf_{buf_name}: size={size}")
    
    print()
    print("=" * 70)
    print("Shape inference test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

