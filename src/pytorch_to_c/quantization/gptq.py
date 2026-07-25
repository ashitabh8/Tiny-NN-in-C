"""GPTQ-style error-compensated weight quantization."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .quant_helpers import (
    quantize_affine_per_group,
    symmetric_scales_per_group,
)


def gptq_quantize(
    weights: np.ndarray,
    hessian: Optional[np.ndarray] = None,
    group_size: int = 128,
    dtype: str = "int8",
    weight_offset: int = 0,
    damp_percent: float = 0.01,
) -> np.ndarray:
    """
    Column-sequential GPTQ-style quantization with error redistribution.

    weights: [in_features, out_features] (input-major, as in IR)
    hessian: optional [in_features, in_features] from calibration (X^T X)
    """
    w = weights.astype(np.float64).copy()
    in_features, out_features = w.shape

    if hessian is None:
        hessian = np.eye(in_features, dtype=np.float64)

    h = hessian.astype(np.float64)
    diag = np.diag(h).copy()
    damp = damp_percent * float(np.mean(diag))
    h = h + np.eye(in_features) * damp

    try:
        h_inv = np.linalg.cholesky(h)
        h_inv = np.linalg.inv(h_inv)
        h_inv = h_inv @ h_inv.T
    except np.linalg.LinAlgError:
        h_inv = np.linalg.pinv(h)

    scales = symmetric_scales_per_group(w, group_size, dtype, weight_offset)
    num_groups = in_features // group_size
    q_max = 127.0 if dtype == "int8" else 32767.0

    w_q = np.zeros_like(w)

    for o in range(out_features):
        w_col = w[:, o].copy()
        err = np.zeros(in_features, dtype=np.float64)

        for g in range(num_groups):
            base = g * group_size
            scale = scales[g, o]
            for i in range(group_size):
                idx = base + i
                q_val = int(np.clip(np.round(w_col[idx] / scale) + weight_offset, -q_max, q_max))
                w_q[idx, o] = q_val
                q_float = (q_val - weight_offset) * scale
                delta = w_col[idx] - q_float
                w_col[idx] = q_float
                err[idx] = delta
                if idx + 1 < in_features:
                    w_col[idx + 1 :] -= err[idx] * h_inv[idx, idx + 1 :] / max(h_inv[idx, idx], 1e-8)

    if dtype == "int8":
        return np.clip(w_q, -128, 127).astype(np.int8)
    if dtype == "int16":
        return np.clip(w_q, -32768, 32767).astype(np.int16)
    raise ValueError(f"Unsupported dtype: {dtype}")


def gptq_quantize_per_group(
    weights: np.ndarray,
    hessian: Optional[np.ndarray],
    group_size: int,
    dtype: str,
    weight_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """GPTQ quantize and return (weights_q, scales)."""
    scales = symmetric_scales_per_group(weights, group_size, dtype, weight_offset)
    wq = gptq_quantize(weights, hessian, group_size, dtype, weight_offset)
    return wq, scales
