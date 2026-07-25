"""Shared quantization helpers for rules and passes."""

from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

# Tile alignment constant for Triton/C kernels
BLOCK_K = 32


def q_max_for_dtype(dtype: str) -> float:
    if dtype == "int8":
        return 127.0
    if dtype == "int16":
        return 32767.0
    if dtype == "int4":
        return 7.0
    raise ValueError(f"Unsupported dtype: {dtype}")


def candidate_group_sizes(in_features: int, block_k: int = BLOCK_K) -> List[int]:
    """Return tile-aligned group sizes that divide in_features."""
    sizes: List[int] = []
    g = block_k
    while g < in_features:
        if in_features % g == 0:
            sizes.append(g)
        g += block_k
    sizes.append(in_features)
    return sorted(set(sizes))


def symmetric_scales_per_group(
    weights: np.ndarray,
    group_size: int,
    dtype: str,
    weight_offset: int = 0,
) -> np.ndarray:
    """
    Compute symmetric scales for groups along input axis.

    weights: [in_features, out_features]
    returns: [num_groups, out_features]
    """
    in_features, out_features = weights.shape
    num_groups = in_features // group_size
    q_max = q_max_for_dtype(dtype)
    scales = np.empty((num_groups, out_features), dtype=np.float32)
    for g in range(num_groups):
        sl = weights[g * group_size : (g + 1) * group_size]
        for o in range(out_features):
            amax = float(np.max(np.abs(sl[:, o]))) if sl.size else 0.0
            scales[g, o] = (amax / q_max) if amax > 0.0 else (1.0 / q_max)
    return scales


def quantize_affine_per_group(
    weights: np.ndarray,
    scales: np.ndarray,
    group_size: int,
    weight_offset: int,
    dtype: str,
) -> np.ndarray:
    """Quantize weights using per-group scales along input axis."""
    in_features, out_features = weights.shape
    num_groups = in_features // group_size
    assert scales.shape == (num_groups, out_features)
    acc = np.zeros_like(weights, dtype=np.float64)
    for g in range(num_groups):
        for o in range(out_features):
            sl = weights[g * group_size : (g + 1) * group_size, o]
            acc[g * group_size : (g + 1) * group_size, o] = (
                np.round(sl / scales[g, o]) + weight_offset
            )
    if dtype == "int8":
        return np.clip(acc, -128, 127).astype(np.int8)
    if dtype == "int16":
        return np.clip(acc, -32768, 32767).astype(np.int16)
    raise ValueError(f"Unsupported dtype: {dtype}")


def group_quant_mse(weights: np.ndarray, group_size: int, dtype: str) -> float:
    """MSE between float weights and round-trip group quantization."""
    scales = symmetric_scales_per_group(weights, group_size, dtype)
    wq = quantize_affine_per_group(weights, scales, group_size, 0, dtype)
    recon = np.zeros_like(weights, dtype=np.float64)
    num_groups = weights.shape[0] // group_size
    for g in range(num_groups):
        for o in range(weights.shape[1]):
            sl = wq[g * group_size : (g + 1) * group_size, o].astype(np.float64)
            recon[g * group_size : (g + 1) * group_size, o] = sl * scales[g, o]
    return float(np.mean((weights - recon) ** 2))


def select_group_size(
    weights: np.ndarray,
    dtype: str,
    group_size: Union[int, str] = "auto",
    error_budget: float | None = None,
    block_k: int = BLOCK_K,
) -> int:
    """Pick group size: explicit int, or auto with optional error budget."""
    in_features = weights.shape[0]
    if isinstance(group_size, int):
        if in_features % group_size == 0:
            return group_size
        candidates = candidate_group_sizes(in_features, block_k)
        valid = [g for g in candidates if g <= group_size]
        if valid:
            return valid[-1]
        return in_features
    candidates = candidate_group_sizes(in_features, block_k)
    if error_budget is None:
        return candidates[-1]
    for g in reversed(candidates):
        if group_quant_mse(weights, g, dtype) <= error_budget:
            return g
    return candidates[0]


def pack_int4_nibbles(values: np.ndarray) -> np.ndarray:
    """Pack signed int4 values (even length) into int8 bytes (2 nibbles/byte)."""
    flat = values.reshape(-1).astype(np.int8)
    if flat.size % 2 != 0:
        flat = np.append(flat, 0)
    low = flat[0::2] & 0x0F
    high = (flat[1::2] & 0x0F) << 4
    return (low | high).astype(np.int8)


def unpack_int4_nibbles(packed: np.ndarray, count: int) -> np.ndarray:
    """Unpack int4 values from packed int8 bytes."""
    flat = packed.reshape(-1)
    low = (flat & 0x0F).astype(np.int8)
    high = ((flat >> 4) & 0x0F).astype(np.int8)
    # sign extend 4-bit
    low = np.where(low >= 8, low - 16, low)
    high = np.where(high >= 8, high - 16, high)
    out = np.empty(flat.size * 2, dtype=np.int8)
    out[0::2] = low
    out[1::2] = high
    return out[:count]


def kmeans_1d(values: np.ndarray, k: int, max_iter: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Simple 1D k-means for weight palettization."""
    flat = values.reshape(-1).astype(np.float64)
    if flat.size == 0:
        return np.zeros(k, dtype=np.float32), np.zeros(0, dtype=np.uint8)
    k = min(k, flat.size)
    rng = np.random.default_rng(42)
    centroids = np.linspace(float(flat.min()), float(flat.max()), k)
    for _ in range(max_iter):
        dist = np.abs(flat[:, None] - centroids[None, :])
        labels = np.argmin(dist, axis=1)
        new_centroids = centroids.copy()
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                new_centroids[i] = flat[mask].mean()
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return centroids.astype(np.float32), labels.astype(np.uint8)


def pack_palette_indices(indices: np.ndarray, num_centroids: int) -> np.ndarray:
    """Pack palette indices into 4-bit nibbles when K<=16."""
    if num_centroids <= 16:
        return pack_int4_nibbles(indices.astype(np.int8))
    return indices.astype(np.uint8)
