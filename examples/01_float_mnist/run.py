#!/usr/bin/env python3
"""
Float MNIST -- Train, compile to C, and verify.

End-to-end example with NO quantization:
  1. Train MNISTConvNet on MNIST (skip if checkpoint exists).
  2. Compile the trained model to standalone C code.
  3. Verify PyTorch vs C outputs match across random inputs.

Usage:
    python examples/01_float_mnist/run.py

Requirements:
    pip install torchvision   (for MNIST dataset)
    gcc                       (for verification harness)
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim

from models import MNISTConvNet
from src.pytorch_to_c.compiler import compile_model
from tools.verify_model import verify_model

CHECKPOINT = os.path.join(SCRIPT_DIR, "mnist_cnn.pth")
GENERATED_DIR = os.path.join(SCRIPT_DIR, "generated")
EPOCHS = 3
BATCH_SIZE = 128


def train_mnist(model: nn.Module) -> None:
    """Train MNISTConvNet on MNIST for a few epochs."""
    try:
        import torchvision
        import torchvision.transforms as T
    except ImportError:
        sys.exit(
            "torchvision is required for MNIST training.\n"
            "Install it:  pip install torchvision"
        )

    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    train_set = torchvision.datasets.MNIST(
        root=os.path.join(PROJECT_ROOT, "data"),
        train=True, download=True, transform=transform,
    )
    loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        acc = 100.0 * correct / total
        avg_loss = running_loss / total
        print(f"  Epoch {epoch}/{EPOCHS}  loss={avg_loss:.4f}  acc={acc:.1f}%")

    model.cpu().eval()
    torch.save(model.state_dict(), CHECKPOINT)
    print(f"  Saved checkpoint: {CHECKPOINT}")


def main() -> None:
    print("=" * 60)
    print("Example 01 -- Float MNIST: Train, Compile to C, Verify")
    print("=" * 60)

    model = MNISTConvNet()
    example_input = torch.randn(1, 1, 28, 28)

    # --- Step 1: Train (or load) ---
    if os.path.exists(CHECKPOINT):
        print(f"\n[1/3] Loading existing checkpoint: {CHECKPOINT}")
        model.load_state_dict(torch.load(CHECKPOINT, weights_only=True))
    else:
        print(f"\n[1/3] Training MNISTConvNet for {EPOCHS} epochs...")
        train_mnist(model)

    model.eval()

    # --- Step 2: Compile to C ---
    print(f"\n[2/3] Compiling to C -> {GENERATED_DIR}/")
    compile_model(model, example_input, output_dir=GENERATED_DIR, verbose=False)
    print("  Done.")

    # --- Step 3: Verify ---
    print("\n[3/3] Verifying PyTorch vs C (50 random samples)...")
    results = verify_model(
        model, example_input,
        num_samples=50,
        tolerance=1e-3,
        verbose=True,
    )
    print()
    print(results.summary())
    sys.exit(0 if results.failed == 0 else 1)


if __name__ == "__main__":
    main()
