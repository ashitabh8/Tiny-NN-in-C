#!/usr/bin/env python3
"""
QAT TinyResNet on MNIST -- Quantization-Aware Training, compile to C, verify.

End-to-end example using real MNIST data:
  1. Train TinyResNet (2D) on MNIST in two phases:
       a) Float warmup   -- normal training to reach good baseline accuracy.
       b) QAT fine-tune  -- enable fake-quantization observers so the model
          learns to tolerate quantization noise.
  2. Evaluate on MNIST test set.
  3. Extract learned scales from the observers.
  4. Compile using StaticQuantRule with calibrated scales.
  5. Verify PyTorch (float) vs quantized C.

The training loop lives entirely in this file -- the compiler library itself
knows nothing about QAT; it only sees the final scales you give it.

Usage:
    python examples/03_qat_resnet/run.py

Requirements:
    pip install torchvision   (for MNIST dataset)
    gcc                       (for verification harness)
"""

import os
import sys
import subprocess
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models import TinyResNet
from src.pytorch_to_c.compiler import compile_model
from src.pytorch_to_c.codegen.c_printer import CPrinter
from src.pytorch_to_c.quantization import StaticQuantRule, QuantizationTransform
from tools.verify_model import verify_model

GENERATED_DIR = os.path.join(SCRIPT_DIR, "generated")
CHECKPOINT = os.path.join(SCRIPT_DIR, "qat_resnet_mnist.pth")
WARMUP_EPOCHS = 5   # float-only training
QAT_EPOCHS = 5      # fine-tune with fake quantization
NUM_CLASSES = 10
BATCH_SIZE = 128


# ---------------------------------------------------------------------------
# MNIST data loaders
# ---------------------------------------------------------------------------

def get_mnist_loaders():
    """Load MNIST train and test sets."""
    try:
        import torchvision
        import torchvision.transforms as T
    except ImportError:
        sys.exit(
            "torchvision is required for MNIST.\n"
            "Install it:  pip install torchvision"
        )

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,)),
    ])

    train_set = torchvision.datasets.MNIST(
        root=os.path.join(PROJECT_ROOT, "data"),
        train=True, download=True, transform=transform,
    )
    test_set = torchvision.datasets.MNIST(
        root=os.path.join(PROJECT_ROOT, "data"),
        train=False, download=True, transform=transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
    )

    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Fake-quantization helper
# ---------------------------------------------------------------------------

class FakeQuantObserver:
    """Tracks running min/max of tensors flowing through a point in the graph."""

    def __init__(self):
        self.min_val = float("inf")
        self.max_val = float("-inf")

    def observe(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.min_val = min(self.min_val, float(x.min()))
            self.max_val = max(self.max_val, float(x.max()))
        return x

    @property
    def scale(self) -> float:
        absmax = max(abs(self.min_val), abs(self.max_val))
        return absmax / 127.0 if absmax > 0 else 1.0 / 127.0


class QATTinyResNet(nn.Module):
    """TinyResNet wrapped with fake-quant observers on every Conv2d/Linear I/O.

    Set ``self.qat_enabled = False`` to disable fake-quantization noise
    (observers still record stats).
    """

    def __init__(self, base_model: TinyResNet):
        super().__init__()
        self.model = base_model
        self.qat_enabled = False
        self.observers: dict[str, FakeQuantObserver] = {}
        self._register_observers()

    def _register_observers(self) -> None:
        for name, mod in self.model.named_modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                self.observers[f"{name}_input"] = FakeQuantObserver()
                self.observers[f"{name}_output"] = FakeQuantObserver()

    def _fake_quant(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        inv_scale = 1.0 / scale if scale > 0 else 1.0
        x_q = torch.clamp(torch.round(x * inv_scale), -128, 127)
        return x_q * scale

    def _observe_and_fq(self, x: torch.Tensor, key: str) -> torch.Tensor:
        x = self.observers[key].observe(x)
        if self.training and self.qat_enabled:
            x = self._fake_quant(x, self.observers[key].scale)
        return x

    def _run_resblock(self, block, block_name, x):
        identity = x

        x = self._observe_and_fq(x, f"{block_name}.conv1_input")
        x = block.conv1(x)
        x = self.observers[f"{block_name}.conv1_output"].observe(x)
        x = block.bn1(x)
        x = block.relu1(x)

        x = self._observe_and_fq(x, f"{block_name}.conv2_input")
        x = block.conv2(x)
        x = self.observers[f"{block_name}.conv2_output"].observe(x)
        x = block.bn2(x)
        x = x + identity
        x = block.relu2(x)
        return x

    def _run_conv_bn_relu(self, conv, bn, relu, conv_name, x):
        x = self._observe_and_fq(x, f"{conv_name}_input")
        x = conv(x)
        x = self.observers[f"{conv_name}_output"].observe(x)
        x = bn(x)
        x = relu(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.model

        x = self._run_conv_bn_relu(m.conv_stem, m.bn_stem, m.relu_stem, "conv_stem", x)
        x = self._run_resblock(m.block1, "block1", x)

        x = self._run_conv_bn_relu(m.conv_down1, m.bn_down1, m.relu_down1, "conv_down1", x)
        x = self._run_resblock(m.block2, "block2", x)

        x = self._run_conv_bn_relu(m.conv_down2, m.bn_down2, m.relu_down2, "conv_down2", x)
        x = self._run_resblock(m.block3, "block3", x)

        x = x.mean(dim=[2, 3])

        x = self._observe_and_fq(x, "fc_input")
        x = m.fc(x)
        x = self.observers["fc_output"].observe(x)
        return x


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, loader, device: torch.device) -> tuple[float, float]:
    """Return (loss, accuracy%) on the given data loader."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
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
    return running_loss / total, 100.0 * correct / total


def train_qat(
    qat_model: QATTinyResNet,
    train_loader,
    test_loader,
    device: torch.device,
) -> None:
    """Two-phase training: float warmup then QAT fine-tuning."""
    optimizer = optim.Adam(qat_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    total_epochs = WARMUP_EPOCHS + QAT_EPOCHS

    # Phase 1: Float warmup (no fake quantization, observers still track stats)
    print(f"\n  --- Phase 1: Float warmup ({WARMUP_EPOCHS} epochs) ---")
    qat_model.qat_enabled = False
    for epoch in range(1, WARMUP_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            qat_model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(qat_model, test_loader, device)
        print(f"  Epoch {epoch:2d}/{total_epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.1f}%  "
              f"test_loss={test_loss:.4f}  test_acc={test_acc:.1f}%")

    # Phase 2: QAT fine-tuning (fake quantization enabled)
    print(f"\n  --- Phase 2: QAT fine-tune ({QAT_EPOCHS} epochs) ---")
    qat_model.qat_enabled = True
    for param_group in optimizer.param_groups:
        param_group['lr'] = 3e-4
    for epoch in range(WARMUP_EPOCHS + 1, total_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            qat_model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(qat_model, test_loader, device)
        print(f"  Epoch {epoch:2d}/{total_epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.1f}%  "
              f"test_loss={test_loss:.4f}  test_acc={test_acc:.1f}%  [QAT]")


# ---------------------------------------------------------------------------
# Scale extraction
# ---------------------------------------------------------------------------

def extract_rules(qat_model: QATTinyResNet) -> list:
    """Build StaticQuantRule list from the observer statistics."""
    obs = qat_model.observers
    rules = []

    base = qat_model.model
    for name, mod in base.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            w = mod.weight.detach()
            w_absmax = float(w.abs().max())
            w_scale = w_absmax / 127.0 if w_absmax > 0 else 1.0 / 127.0
            escaped = name.replace(".", r"\.")
            rules.append(StaticQuantRule(
                pattern=f".*{escaped}.*",
                dtype="int8",
                input_scale=obs[f"{name}_input"].scale,
                input_offset=0,
                weight_scale=w_scale,
                weight_offset=0,
                output_scale=obs[f"{name}_output"].scale,
                output_offset=0,
            ))

    return rules


# ---------------------------------------------------------------------------
# Run the compiled C model on real MNIST images
# ---------------------------------------------------------------------------

def evaluate_c_on_mnist(
    test_loader,
    generated_dir: str,
    example_input: torch.Tensor,
) -> float:
    """Compile the generated C, feed every MNIST test image through it,
    and return the classification accuracy (%).

    The C model expects NHWC float input and produces float logits.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # -- 1. Copy generated C files into tmpdir --
        import shutil
        for fname in os.listdir(generated_dir):
            shutil.copy2(os.path.join(generated_dir, fname), tmpdir)

        # -- 2. Compute flat sizes from example_input --
        shape = list(example_input.shape)
        if shape[0] == 1:
            shape = shape[1:]
        input_size = int(np.prod(shape))
        output_size = NUM_CLASSES

        # -- 3. Write C harness --
        harness_src = os.path.join(tmpdir, "harness.c")
        with open(harness_src, "w") as f:
            f.write(f"""
#include <stdio.h>
#include <stdlib.h>
#include "model.h"

int main(int argc, char* argv[]) {{
    int num_samples = atoi(argv[1]);
    const int input_size  = {input_size};
    const int output_size = {output_size};

    float* input  = (float*)malloc(input_size  * sizeof(float));
    float* output = (float*)malloc(output_size * sizeof(float));

    FILE* f_in  = fopen(argv[2], "rb");
    FILE* f_out = fopen(argv[3], "wb");

    for (int s = 0; s < num_samples; ++s) {{
        fread(input, sizeof(float), input_size, f_in);
        model_forward(input, output);
        fwrite(output, sizeof(float), output_size, f_out);
    }}

    fclose(f_in); fclose(f_out);
    free(input); free(output);
    return 0;
}}
""")

        # -- 4. Compile --
        exe = os.path.join(tmpdir, "mnist_eval")
        model_c = os.path.join(tmpdir, "model.c")
        cmd = ["gcc", "-o", exe, harness_src, model_c,
               f"-I{tmpdir}", "-lm", "-std=c99", "-O2"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"gcc failed:\n{result.stderr}")

        # -- 5. Prepare all test images as NHWC float binary --
        all_inputs = []
        all_labels = []
        for images, labels in test_loader:
            for i in range(images.size(0)):
                img = images[i:i+1]                        # (1, C, H, W)
                img_nhwc = img.permute(0, 2, 3, 1)        # (1, H, W, C)
                all_inputs.append(img_nhwc.numpy().flatten().astype(np.float32))
                all_labels.append(int(labels[i]))
        num_samples = len(all_labels)

        input_bin = os.path.join(tmpdir, "inputs.bin")
        output_bin = os.path.join(tmpdir, "outputs.bin")
        np.concatenate(all_inputs).tofile(input_bin)

        # -- 6. Run C executable --
        print(f"  Running {num_samples} images through compiled C binary...")
        result = subprocess.run(
            [exe, str(num_samples), input_bin, output_bin],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"C executable failed:\n{result.stderr}")

        # -- 7. Read outputs and compute accuracy --
        c_raw = np.fromfile(output_bin, dtype=np.float32)
        c_outputs = c_raw.reshape(num_samples, output_size)
        c_preds = np.argmax(c_outputs, axis=1)
        all_labels = np.array(all_labels)
        correct = int(np.sum(c_preds == all_labels))
        accuracy = 100.0 * correct / num_samples
        print(f"  {correct}/{num_samples} correct")

        return accuracy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Example 03 -- QAT TinyResNet on MNIST")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = TinyResNet(in_channels=1, num_classes=NUM_CLASSES)
    example_input = torch.randn(1, 1, 28, 28)

    # --- Step 1: Train or load ---
    train_loader, test_loader = get_mnist_loaders()

    if os.path.exists(CHECKPOINT):
        print(f"\n[1/5] Loading QAT checkpoint: {CHECKPOINT}")
        base_model.load_state_dict(torch.load(CHECKPOINT, weights_only=True))
        qat_model = QATTinyResNet(base_model)
        qat_model.to(device)
        # Run one eval pass so observers collect activation stats
        evaluate(qat_model, test_loader, device)
    else:
        total = WARMUP_EPOCHS + QAT_EPOCHS
        print(f"\n[1/5] Training on MNIST (60k train, {total} epochs: "
              f"{WARMUP_EPOCHS} warmup + {QAT_EPOCHS} QAT)...")
        qat_model = QATTinyResNet(base_model)
        qat_model.to(device)
        train_qat(qat_model, train_loader, test_loader, device)
        base_model.cpu()
        torch.save(base_model.state_dict(), CHECKPOINT)
        print(f"\n  Saved checkpoint: {CHECKPOINT}")

    # --- Step 2: Final test accuracy ---
    print("\n[2/5] Evaluating on MNIST test set (10k samples)...")
    qat_model.to(device)
    test_loss, test_acc = evaluate(qat_model, test_loader, device)
    print(f"  Test accuracy (float, QAT-trained): {test_acc:.2f}%")

    # --- Step 3: Extract calibrated scales ---
    print("\n[3/5] Extracting observer scales...")
    qat_model.eval()
    rules = extract_rules(qat_model)
    print(f"  Created {len(rules)} StaticQuantRule(s) from observer statistics.")
    for r in rules:
        print(f"    {r.pattern:40s}  in_s={r.input_scale:.4f}  w_s={r.weight_scale:.4f}  "
              f"out_s={r.output_scale:.4f}")

    # --- Step 4: Compile ---
    print(f"\n[4/6] Compiling to quantized C -> {GENERATED_DIR}/")
    base_model.cpu().eval()
    ir_graph = compile_model(base_model, example_input, return_ir=True, verbose=False)
    ir_graph = QuantizationTransform(rules).apply(ir_graph)
    os.makedirs(GENERATED_DIR, exist_ok=True)
    CPrinter(ir_graph).generate_all(GENERATED_DIR)
    print("  Done.")

    # --- Step 5: Codegen correctness check (random noise) ---
    print("\n[5/6] Codegen sanity check: float PyTorch vs quantized C (50 random)...")
    results = verify_model(
        base_model, example_input,
        num_samples=50,
        quantization_rules=rules,
        tolerance=1.0,
        verbose=False,
    )
    print(f"  {results.passed}/{results.num_samples} passed  "
          f"max_err={results.overall_max_error:.2e}  "
          f"mean_err={results.overall_mean_error:.2e}")

    # --- Step 6: Run quantized C model on real MNIST test set ---
    print("\n[6/6] Running int8 C model on MNIST test set (10k images)...")
    c_acc = evaluate_c_on_mnist(test_loader, GENERATED_DIR, example_input)
    print(f"\n  Float PyTorch test accuracy : {test_acc:.2f}%")
    print(f"  Int8 C model test accuracy  : {c_acc:.2f}%")
    print(f"  Accuracy drop from quant    : {test_acc - c_acc:+.2f}%")

    ok = results.failed == 0 and c_acc >= 80.0
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
