"""
Compare pytorch_outputs.txt vs c_outputs.txt (raw logits per line).

Usage:
    python compare.py [--tol 1e-3]
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PYTORCH_FILE = os.path.join(_THIS_DIR, "pytorch_outputs.txt")
C_FILE = os.path.join(_THIS_DIR, "c_outputs.txt")


def _parse_line(line: str) -> tuple[list[float], str | None]:
    """Strip inline comment; return 10 floats and optional trailing comment."""
    if "#" in line:
        main, comment = line.split("#", 1)
        comment = comment.strip()
    else:
        main, comment = line, None
    parts = main.strip().split()
    vals = [float(x) for x in parts]
    if len(vals) != 10:
        raise ValueError(f"Expected 10 floats, got {len(vals)}: {line!r}")
    return vals, comment


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-3, help="Max abs error per sample")
    args = ap.parse_args()
    tol = args.tol

    if not os.path.isfile(PYTORCH_FILE):
        print(f"ERROR: missing {PYTORCH_FILE}", file=sys.stderr)
        return 1
    if not os.path.isfile(C_FILE):
        print(f"ERROR: missing {C_FILE}", file=sys.stderr)
        return 1

    with open(PYTORCH_FILE, encoding="utf-8") as f:
        pt_lines = [ln for ln in f.read().splitlines() if ln.strip()]
    with open(C_FILE, encoding="utf-8") as f:
        c_lines = [ln for ln in f.read().splitlines() if ln.strip()]

    if len(pt_lines) != len(c_lines):
        print(
            f"ERROR: line count mismatch pytorch={len(pt_lines)} c={len(c_lines)}",
            file=sys.stderr,
        )
        return 1

    print("=" * 72)
    print("MNIST C vs PyTorch logit comparison")
    print("=" * 72)
    print(f"Tolerance (max abs error per sample): {tol:g}")
    print()

    all_ok = True
    print(
        f"{'i':>3} {'label':>5} {'pt_pred':>7} {'c_pred':>7} {'max|err|':>12} {'status':>8}"
    )
    print("-" * 72)

    for i, (pt_ln, c_ln) in enumerate(zip(pt_lines, c_lines)):
        pt, pt_comment = _parse_line(pt_ln)
        c_vals, _ = _parse_line(c_ln)

        errs = [abs(a - b) for a, b in zip(pt, c_vals)]
        max_err = max(errs)
        pt_pred = max(range(10), key=lambda k: pt[k])
        c_pred = max(range(10), key=lambda k: c_vals[k])

        label = None
        if pt_comment:
            m = re.search(r"label=(\d+)", pt_comment)
            if m:
                label = int(m.group(1))

        ok = max_err <= tol
        if not ok:
            all_ok = False
        status = "PASS" if ok else "FAIL"

        lab_str = str(label) if label is not None else "?"

        print(
            f"{i:3d} {lab_str:>5} {pt_pred:7d} {c_pred:7d} {max_err:12.6e} {status:>8}"
        )

    print("-" * 72)
    if all_ok:
        print("RESULT: all samples within tolerance.")
        return 0
    print("RESULT: one or more samples FAILED tolerance — debug nn_ops_float / codegen.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
