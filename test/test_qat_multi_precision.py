"""
MNIST dynamic quantization test — multi-precision INT8 / INT16.

Pipeline:
  1. Train a small MLP on real MNIST (float32)
  2. Measure PyTorch classification accuracy on a held-out test batch
  3. For each quant config (INT8 / INT16 / mixed):
       a. Dynamic-quantize the IR graph
       b. Compile to C, build with gcc
       c. Run the C binary on every test sample
       d. Measure classification accuracy of the C model
  4. Print a comparison table and assert accuracy thresholds

Only dynamic quantization is tested (no static rules).
"""

import pytest
import os
import tempfile
import subprocess

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.pytorch_to_c.compiler import compile_model
from src.pytorch_to_c.codegen.c_printer import CPrinter
from src.pytorch_to_c.quantization import (
    DynamicQuantRuleMinMaxPerTensor,
    QuantizationTransform,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MNISTNet(nn.Module):
    """
    Minimal MLP for MNIST (784 -> 128 -> 64 -> 32 -> 10).

    Layer names use encoder_ / head_ prefixes so that regex rules can
    assign different precisions to each group.
    """

    def __init__(self):
        super().__init__()
        self.encoder_fc1 = nn.Linear(784, 128)
        self.encoder_fc2 = nn.Linear(128, 64)
        self.head_fc1 = nn.Linear(64, 32)
        self.head_fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = torch.relu(self.encoder_fc1(x))
        x = torch.relu(self.encoder_fc2(x))
        x = torch.relu(self.head_fc1(x))
        return self.head_fc2(x)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_mnist():
    """Return (train_dataset, test_dataset) — downloads on first call."""
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    data_dir = os.path.join(os.path.dirname(__file__), "..", ".mnist_cache")
    train = datasets.MNIST(data_dir, train=True,  download=True, transform=tf)
    test  = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    return train, test


def _train_on_mnist(model, train_ds, epochs=3, batch_size=256, lr=1e-3):
    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                         shuffle=True)
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        correct = total = 0
        for imgs, labels in loader:
            x = imgs.view(imgs.size(0), -1)
            opt.zero_grad()
            loss = crit(model(x), labels)
            loss.backward()
            opt.step()
            correct += (model(x).argmax(1) == labels).sum().item()
            total += labels.size(0)
        print(f"    epoch {ep+1}/{epochs}  train acc {correct/total*100:.1f}%")
    model.eval()


def _get_test_batch(test_ds, n=200):
    """Return (images_flat [n, 784], labels [n]) from the test set."""
    loader = torch.utils.data.DataLoader(test_ds, batch_size=n, shuffle=False)
    imgs, labels = next(iter(loader))
    return imgs.view(n, -1), labels


# ---------------------------------------------------------------------------
# C compilation / execution helpers
# ---------------------------------------------------------------------------

def _check_gcc():
    try:
        subprocess.run(["gcc", "--version"], check=True,
                       capture_output=True, timeout=5)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired):
        return False


def _make_batch_harness(tmpdir, input_size, output_size, n_samples):
    """
    C harness that reads N flattened inputs, runs model_forward on each,
    and writes N flattened outputs — all in one process.
    """
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
    float* in_buf  = (float*)malloc(n * {input_size} * sizeof(float));
    float* out_buf = (float*)malloc(n * {output_size} * sizeof(float));
    if (!in_buf || !out_buf) {{ fprintf(stderr, "OOM\\n"); return 1; }}

    FILE* fi = fopen(argv[2], "rb");
    if (!fi || fread(in_buf, sizeof(float), n * {input_size}, fi)
            != (size_t)(n * {input_size})) {{
        fprintf(stderr, "read error\\n"); return 1;
    }}
    fclose(fi);

    for (int i = 0; i < n; ++i) {{
        model_forward(in_buf + i * {input_size},
                      out_buf + i * {output_size});
    }}

    FILE* fo = fopen(argv[3], "wb");
    if (!fo) {{ fprintf(stderr, "write error\\n"); return 1; }}
    fwrite(out_buf, sizeof(float), n * {output_size}, fo);
    fclose(fo);
    free(in_buf); free(out_buf);
    return 0;
}}
"""
    path = os.path.join(tmpdir, "batch_harness.c")
    with open(path, "w") as f:
        f.write(code)
    return path


def _build_c_model(tmpdir):
    """Compile model.c + batch harness, return path to executable."""
    harness = _make_batch_harness(tmpdir, 784, 10, 0)  # n read from argv
    exe = os.path.join(tmpdir, "mnist_model")
    res = subprocess.run(
        ["gcc", "-o", exe, harness,
         os.path.join(tmpdir, "model.c"),
         f"-I{tmpdir}", "-lm", "-std=c99", "-O2"],
        capture_output=True, timeout=60, text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"gcc failed:\n{res.stderr}")
    return exe


def _run_c_batch(exe, tmpdir, inputs_np, n_samples):
    """Run the compiled C model on a batch of inputs, return [n, 10] logits."""
    in_path  = os.path.join(tmpdir, "in.bin")
    out_path = os.path.join(tmpdir, "out.bin")
    inputs_np.astype(np.float32).tofile(in_path)

    res = subprocess.run(
        [exe, str(n_samples), in_path, out_path],
        capture_output=True, timeout=120, text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"C execution failed:\n{res.stderr}")

    return np.fromfile(out_path, dtype=np.float32).reshape(n_samples, 10)


# ---------------------------------------------------------------------------
# Core pipeline: quantize -> compile -> run batch -> accuracy
# ---------------------------------------------------------------------------

def _quantize_compile_and_measure(model, example_input, rules,
                                  test_x, test_y, label):
    """
    Quantize *model* with *rules*, compile to C, run on test_x,
    compute classification accuracy, and return a results dict.
    """
    n = test_x.shape[0]
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        ir = compile_model(model, example_input,
                           output_dir=None, verbose=False, return_ir=True)
        qir = QuantizationTransform(rules).apply(ir)
        CPrinter(qir).generate_all(tmpdir)

        exe = _build_c_model(tmpdir)
        c_logits = _run_c_batch(exe, tmpdir, test_x.numpy(), n)

    c_preds = np.argmax(c_logits, axis=1)
    labels_np = test_y.numpy()
    c_acc = (c_preds == labels_np).sum() / n

    with torch.no_grad():
        pt_logits = model(test_x).numpy()
    pt_preds = np.argmax(pt_logits, axis=1)
    pt_acc = (pt_preds == labels_np).sum() / n

    pred_match = (c_preds == pt_preds).sum() / n

    print(f"\n  {label}")
    print(f"    PyTorch accuracy     : {pt_acc*100:.1f}%")
    print(f"    C (quantized) acc    : {c_acc*100:.1f}%")
    print(f"    C vs PyTorch agree   : {pred_match*100:.1f}%")

    return dict(label=label, pt_acc=pt_acc, c_acc=c_acc,
                pred_match=pred_match)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMNISTDynamicQuantization:
    """
    Train on real MNIST, then quantize with three dynamic configs and
    compare classification accuracy of the generated C code.
    """

    @pytest.fixture(autouse=True)
    def _require_gcc(self):
        if not _check_gcc():
            pytest.skip("gcc not available")

    @pytest.fixture(scope="class")
    def trained_model_and_data(self):
        """Train once, reuse across all tests in this class."""
        torch.manual_seed(42)
        train_ds, test_ds = _load_mnist()
        model = MNISTNet()
        print("\n  Training MNISTNet on MNIST ...")
        _train_on_mnist(model, train_ds, epochs=3)
        test_x, test_y = _get_test_batch(test_ds, n=200)
        example_input = torch.randn(1, 784)
        return model, test_x, test_y, example_input

    # -- individual config tests -------------------------------------------

    def test_dynamic_int8(self, trained_model_and_data):
        """All layers dynamic INT8."""
        model, test_x, test_y, ex = trained_model_and_data
        rules = [DynamicQuantRuleMinMaxPerTensor(r".*fc.*", "int8")]
        r = _quantize_compile_and_measure(
            model, ex, rules, test_x, test_y, "Dynamic INT8 (all layers)")
        assert r["c_acc"] > 0.70, f"INT8 accuracy too low: {r['c_acc']}"

    def test_dynamic_int16(self, trained_model_and_data):
        """All layers dynamic INT16."""
        model, test_x, test_y, ex = trained_model_and_data
        rules = [DynamicQuantRuleMinMaxPerTensor(r".*fc.*", "int16")]
        r = _quantize_compile_and_measure(
            model, ex, rules, test_x, test_y, "Dynamic INT16 (all layers)")
        assert r["c_acc"] > 0.80, f"INT16 accuracy too low: {r['c_acc']}"

    def test_multi_precision_int8_int16(self, trained_model_and_data):
        """Encoder INT8, head INT16 — mixed precision."""
        model, test_x, test_y, ex = trained_model_and_data
        rules = [
            DynamicQuantRuleMinMaxPerTensor(r"encoder_fc.*", "int8"),
            DynamicQuantRuleMinMaxPerTensor(r"head_fc.*", "int16"),
        ]
        r = _quantize_compile_and_measure(
            model, ex, rules, test_x, test_y,
            "Mixed: encoder INT8, head INT16")
        assert r["c_acc"] > 0.70

    def test_selective_encoder_only(self, trained_model_and_data):
        """Only encoder quantized (INT8); head stays float32."""
        model, test_x, test_y, ex = trained_model_and_data
        rules = [DynamicQuantRuleMinMaxPerTensor(r"encoder_fc.*", "int8")]
        r = _quantize_compile_and_measure(
            model, ex, rules, test_x, test_y,
            "Selective: encoder INT8, head float32")
        assert r["c_acc"] > 0.80

    # -- comparison test ---------------------------------------------------

    def test_precision_ladder(self, trained_model_and_data):
        """
        Run all three configs on the same batch, print a summary table,
        and verify INT16 >= INT8 accuracy (with margin).
        """
        model, test_x, test_y, ex = trained_model_and_data

        configs = [
            ("int8",  [DynamicQuantRuleMinMaxPerTensor(r".*fc.*", "int8")]),
            ("mixed", [DynamicQuantRuleMinMaxPerTensor(r"encoder_fc.*", "int8"),
                       DynamicQuantRuleMinMaxPerTensor(r"head_fc.*", "int16")]),
            ("int16", [DynamicQuantRuleMinMaxPerTensor(r".*fc.*", "int16")]),
        ]

        results = {}
        for name, rules in configs:
            results[name] = _quantize_compile_and_measure(
                model, ex, rules, test_x, test_y,
                f"Ladder — {name}")

        print(f"\n  {'='*56}")
        print(f"  MNIST Classification Accuracy — Precision Ladder")
        print(f"  {'='*56}")
        print(f"  {'Config':<12} {'PyTorch':>10} {'C (quant)':>10} "
              f"{'C==PT':>10}")
        print(f"  {'-'*56}")
        for name in ["int8", "mixed", "int16"]:
            r = results[name]
            print(f"  {name:<12} {r['pt_acc']*100:>9.1f}% "
                  f"{r['c_acc']*100:>9.1f}% "
                  f"{r['pred_match']*100:>9.1f}%")
        print()

        assert results["int16"]["c_acc"] >= results["int8"]["c_acc"] - 0.05, \
            "INT16 should be at least as accurate as INT8 (within 5% margin)"
