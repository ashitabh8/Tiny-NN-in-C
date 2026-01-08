"""
Example: Compiling a simple MLP to C code

This example demonstrates how to use the PyTorch-to-C compiler
to convert a simple Multi-Layer Perceptron to C code.
"""

import torch
import torch.nn as nn
import sys
import os

# Add src to path so we can import the compiler
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pytorch_to_c.compiler import compile_model


class SimpleMLP(nn.Module):
    """A simple 2-layer MLP for demonstration."""
    
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def main():
    """Main example function."""
    print("=" * 60)
    print("PyTorch to C Compiler - Example")
    print("=" * 60)
    print()
    
    # Create a simple model
    print("Creating model...")
    model = SimpleMLP(input_size=784, hidden_size=128, output_size=10)
    model.eval()
    
    # Create example input
    print("Creating example input...")
    example_input = torch.randn(1, 784)
    
    # Test the model in PyTorch
    print("\nRunning PyTorch inference...")
    with torch.no_grad():
        pytorch_output = model(example_input)
    print(f"PyTorch output shape: {pytorch_output.shape}")
    print(f"PyTorch output (first 5): {pytorch_output[0, :5].numpy()}")
    
    # Compile to C
    print("\n" + "=" * 60)
    print("Compiling to C...")
    print("=" * 60)
    
    output_dir = "generated"
    ir_graph = compile_model(
        model=model,
        example_input=example_input,
        output_dir=output_dir,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("Compilation Complete!")
    print("=" * 60)
    print(f"\nGenerated files are in: {output_dir}/")
    print("  - model.h    : Function declarations")
    print("  - model.c    : Model implementation")
    print("  - weights.h  : Trained weights")
    print()
    
    # Print IR graph summary
    print("IR Graph Summary:")
    print(f"  Nodes: {len(ir_graph.nodes)}")
    print(f"  Parameters: {len(ir_graph.parameters)}")
    print(f"  Inputs: {len(ir_graph.inputs)}")
    print(f"  Outputs: {len(ir_graph.outputs)}")
    print()
    
    print("Node Types:")
    node_types = {}
    for node in ir_graph.nodes:
        node_types[node.op_type] = node_types.get(node.op_type, 0) + 1
    for op_type, count in sorted(node_types.items()):
        print(f"  {op_type}: {count}")
    print()
    
    print("Next steps:")
    print("  1. Review the generated C code in the 'generated/' directory")
    print("  2. Integrate with your embedded project")
    print("  3. Compile with your target toolchain")
    print("  4. Call model_forward(input, output) to run inference")
    print()


if __name__ == "__main__":
    main()

