#!/usr/bin/env python3
"""
Dynamic Depthwise-Separable CNN on MNIST — compile and measure accuracy.

Defines a depthwise-separable CNN, trains on MNIST for a few epochs,
applies DynamicQuantRuleMinMaxPerTensor to all conv/linear layers,
compiles to C, and compares classification accuracy between PyTorch
float32 and the generated C int8 model.

Usage:
    python examples/04_dynamic_depthwise/run.py
"""

import os
import sys
import tempfile
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.pytorch_to_c.compiler import compile_model
from src.pytorch_to_c.codegen.c_printer import CPrinter
from src.pytorch_to_c.quantization import (
    DynamicQuantRuleMinMaxPerTensor,
    QuantizationTransform,
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DepthwiseSeparableMNIST(nn.Module):
    """
    Depthwise-separable CNN for MNIST.

    Architecture:
      Conv2d(1,32, 3, pad=1)                       -> BN -> ReLU  [28x28]
      Conv2d(32,32, 3, pad=1, groups=32)  (dw)     -> BN -> ReLU  [28x28]
      Conv2d(32,64, 1)                    (pw)      -> BN -> ReLU  [28x28]
      Conv2d(64,64, 3, pad=1, s=2, groups=64) (dw) -> BN -> ReLU  [14x14]
      Conv2d(64,128, 1)                   (pw)      -> BN -> ReLU  [14x14]
      mean(dim=[2,3])                               -> [128]
      Linear(128, 10)                               -> [10]
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.dw1 = nn.Conv2d(32, 32, 3, padding=1, groups=32)
        self.bn_dw1 = nn.BatchNorm2d(32)

        self.pw1 = nn.Conv2d(32, 64, 1)
        self.bn_pw1 = nn.BatchNorm2d(64)

        self.dw2 = nn.Conv2d(64, 64, 3, padding=1, stride=2, groups=64)
        self.bn_dw2 = nn.BatchNorm2d(64)

        self.pw2 = nn.Conv2d(64, 128, 1)
        self.bn_pw2 = nn.BatchNorm2d(128)

        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn_dw1(self.dw1(x)))
        x = torch.relu(self.bn_pw1(self.pw1(x)))
        x = torch.relu(self.bn_dw2(self.dw2(x)))
        x = torch.relu(self.bn_pw2(self.pw2(x)))
        x = x.mean(dim=[2, 3])
        return self.fc(x)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_mnist():
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    data_dir = os.path.join(os.path.dirname(__file__), "..", ".mnist_cache")
    train = datasets.MNIST(data_dir, train=True,  download=True, transform=tf)
    test  = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    return train, test


def _train(model, train_ds, epochs=3, batch_size=256, lr=1e-3):
    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                         shuffle=True)
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        correct = total = 0
        for imgs, labels in loader:
            opt.zero_grad()
            logits = model(imgs)
            loss = crit(logits, labels)
            loss.backward()
            opt.step()
            correct += (logits.detach().argmax(1) == labels).sum().item()
            total += labels.size(0)
        print(f"    epoch {ep+1}/{epochs}  train acc {correct/total*100:.1f}%")
    model.eval()


def _get_test_batch(test_ds, n=500):
    loader = torch.utils.data.DataLoader(test_ds, batch_size=n, shuffle=False)
    imgs, labels = next(iter(loader))
    return imgs, labels


# ---------------------------------------------------------------------------
# C compilation / execution helpers
# ---------------------------------------------------------------------------

INPUT_ELEMS  = 1 * 28 * 28   # single MNIST image flattened
OUTPUT_ELEMS = 10


def _make_batch_harness(tmpdir, n_samples):
    code = f"""
#include <stdio.h>
#include <stdlib.h>
#include "model.h"

int main(int argc, char* argv[]) {{
    if (argc != 4) {{
        fprintf(stderr, "Usage: %s <n> <in.bin> <out.bin>\\n", argv[0]);
        return 1;
    }}
    int n = atoi(argv[1]);
    float* in_buf  = (float*)malloc(n * {INPUT_ELEMS} * sizeof(float));
    float* out_buf = (float*)malloc(n * {OUTPUT_ELEMS} * sizeof(float));
    if (!in_buf || !out_buf) {{ fprintf(stderr, "OOM\\n"); return 1; }}

    FILE* fi = fopen(argv[2], "rb");
    if (!fi || fread(in_buf, sizeof(float), n * {INPUT_ELEMS}, fi)
            != (size_t)(n * {INPUT_ELEMS})) {{
        fprintf(stderr, "read error\\n"); return 1;
    }}
    fclose(fi);

    for (int i = 0; i < n; ++i) {{
        model_forward(in_buf + i * {INPUT_ELEMS},
                      out_buf + i * {OUTPUT_ELEMS});
    }}

    FILE* fo = fopen(argv[3], "wb");
    if (!fo) {{ fprintf(stderr, "write error\\n"); return 1; }}
    fwrite(out_buf, sizeof(float), n * {OUTPUT_ELEMS}, fo);
    fclose(fo);
    free(in_buf); free(out_buf);
    return 0;
}}
"""
    path = os.path.join(tmpdir, "batch_harness.c")
    with open(path, "w") as f:
        f.write(code)
    return path


def _build(tmpdir):
    harness = _make_batch_harness(tmpdir, 0)
    exe = os.path.join(tmpdir, "dw_mnist")
    res = subprocess.run(
        ["gcc", "-o", exe, harness,
         os.path.join(tmpdir, "model.c"),
         f"-I{tmpdir}", "-lm", "-std=c99", "-O2"],
        capture_output=True, timeout=120, text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"gcc failed:\n{res.stderr}")
    return exe


def _run_batch(exe, tmpdir, inputs_np, n):
    in_path  = os.path.join(tmpdir, "in.bin")
    out_path = os.path.join(tmpdir, "out.bin")
    inputs_np.astype(np.float32).tofile(in_path)

    res = subprocess.run(
        [exe, str(n), in_path, out_path],
        capture_output=True, timeout=300, text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"C execution failed:\n{res.stderr}")

    return np.fromfile(out_path, dtype=np.float32).reshape(n, OUTPUT_ELEMS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Example 04 — Dynamic Depthwise-Separable CNN on MNIST")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Data
    print("\n[1/6] Loading MNIST ...")
    train_ds, test_ds = _load_mnist()

    # 2. Train
    print("\n[2/6] Training DepthwiseSeparableMNIST (3 epochs) ...")
    model = DepthwiseSeparableMNIST()
    _train(model, train_ds, epochs=3)

    # 3. PyTorch accuracy
    print("\n[3/6] Measuring PyTorch float32 accuracy ...")
    test_imgs, test_labels = _get_test_batch(test_ds, n=500)
    with torch.no_grad():
        pt_logits = model(test_imgs).numpy()
    pt_preds = np.argmax(pt_logits, axis=1)
    labels_np = test_labels.numpy()
    pt_acc = (pt_preds == labels_np).sum() / len(labels_np)
    print(f"    PyTorch float32 accuracy: {pt_acc*100:.1f}%")

    # 4. Dynamic quantize + compile
    print("\n[4/6] Applying dynamic int8 quantization and compiling to C ...")
    example_input = torch.randn(1, 1, 28, 28)
    rules = [
        DynamicQuantRuleMinMaxPerTensor(pattern=r"(conv|dw|pw|fc).*", dtype="int8"),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        ir = compile_model(model, example_input,
                           output_dir=None, verbose=False, return_ir=True)
        qir = QuantizationTransform(rules).apply(ir)
        CPrinter(qir).generate_all(tmpdir)

        # 5. Build + run
        print("\n[5/6] Building C model and running on test set ...")
        exe = _build(tmpdir)

        # C model expects NHWC layout; PyTorch is NCHW -> permute to NHWC
        inputs_nhwc = test_imgs.permute(0, 2, 3, 1).numpy()
        inputs_flat = inputs_nhwc.reshape(len(labels_np), -1)
        c_logits = _run_batch(exe, tmpdir, inputs_flat, len(labels_np))

    c_preds = np.argmax(c_logits, axis=1)
    c_acc = (c_preds == labels_np).sum() / len(labels_np)
    agree = (c_preds == pt_preds).sum() / len(labels_np)

    # 6. Report
    print("\n[6/6] Results")
    print("=" * 50)
    print(f"  PyTorch float32 accuracy : {pt_acc*100:.1f}%")
    print(f"  C int8 dynamic accuracy  : {c_acc*100:.1f}%")
    print(f"  C vs PyTorch agree       : {agree*100:.1f}%")
    print(f"  Accuracy drop            : {(pt_acc - c_acc)*100:+.1f}%")
    print("=" * 50)

    if c_acc < 0.70:
        print("\nWARNING: C accuracy below 70% — something may be wrong.")
        sys.exit(1)
    else:
        print("\nPASS: Dynamic depthwise quantization working correctly.")
        sys.exit(0)


if __name__ == "__main__":
    main()
