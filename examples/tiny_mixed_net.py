import torch
import torch.nn as nn
import sys
import os

# Add src to path so we can import the compiler
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pytorch_to_c.compiler import compile_model

class MixedNet(nn.Module):
    """
    Mixed network: Conv -> ReLU -> Linear -> Softmax
    Tests different operation types and datatype bridging (future quantization).
    """
    
    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32 * 16 * 16, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Input: [B, C, H, W]
        x = self.conv(x)  # [B, 32, 16, 16]
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        x = self.softmax(x)
        return x


def main():
    """Main example function."""
    print("=" * 60)
    print("PyTorch to C Compiler - Example")
    print("=" * 60)
    print()
    
    # Create a mixed network
    print("Creating model...")
    model = MixedNet(input_channels=3, num_classes=10)
    model.eval()

    # Create example input
    print("Creating example input...")
    example_input = torch.randn(1, 3, 32, 32)
    
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
    
    output_dir = "tmp/generated_mixed_net"
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
    
    print("\nIR Graph:")
    print(ir_graph.print_graph())
    print()

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

if __name__ == "__main__":
    main()