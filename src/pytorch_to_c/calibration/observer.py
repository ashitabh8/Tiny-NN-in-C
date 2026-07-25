"""Activation observers for calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class LayerActivationStats:
    """Per-layer activation statistics from calibration data."""

    name: str
    min_val: float = float("inf")
    max_val: float = float("-inf")
    absmax: float = 0.0
    percentile_999: float = 0.0
    num_samples: int = 0
    values_buffer: List[np.ndarray] = field(default_factory=list, repr=False)
    output_min_val: float = float("inf")
    output_max_val: float = float("-inf")
    output_absmax: float = 0.0
    output_percentile_999: float = 0.0
    output_num_samples: int = 0
    output_values_buffer: List[np.ndarray] = field(default_factory=list, repr=False)

    def update(self, x: np.ndarray, store_values: bool = False) -> None:
        flat = np.asarray(x, dtype=np.float64).reshape(-1)
        if flat.size == 0:
            return
        self.min_val = min(self.min_val, float(flat.min()))
        self.max_val = max(self.max_val, float(flat.max()))
        self.absmax = max(self.absmax, float(np.max(np.abs(flat))))
        self.num_samples += 1
        if store_values:
            self.values_buffer.append(flat.copy())

    def update_output(self, x: np.ndarray, store_values: bool = False) -> None:
        flat = np.asarray(x, dtype=np.float64).reshape(-1)
        if flat.size == 0:
            return
        self.output_min_val = min(self.output_min_val, float(flat.min()))
        self.output_max_val = max(self.output_max_val, float(flat.max()))
        self.output_absmax = max(self.output_absmax, float(np.max(np.abs(flat))))
        self.output_num_samples += 1
        if store_values:
            self.output_values_buffer.append(flat.copy())

    def finalize_percentile(self, percentile: float = 99.9) -> None:
        if self.values_buffer:
            all_vals = np.concatenate(self.values_buffer)
            self.percentile_999 = float(np.percentile(np.abs(all_vals), percentile))
        else:
            self.percentile_999 = self.absmax
        if self.output_values_buffer:
            all_vals = np.concatenate(self.output_values_buffer)
            self.output_percentile_999 = float(
                np.percentile(np.abs(all_vals), percentile)
            )
        else:
            self.output_percentile_999 = self.output_absmax


class ActivationObserver:
    """Tracks running activation ranges for a single module."""

    def __init__(self, name: str, store_values: bool = False):
        self.name = name
        self.store_values = store_values
        self.stats = LayerActivationStats(name=name)

    def observe(self, tensor) -> None:
        if hasattr(tensor, "detach"):
            arr = tensor.detach().cpu().numpy()
        else:
            arr = np.asarray(tensor)
        self.stats.update(arr, store_values=self.store_values)

    def observe_output(self, tensor) -> None:
        if hasattr(tensor, "detach"):
            arr = tensor.detach().cpu().numpy()
        else:
            arr = np.asarray(tensor)
        self.stats.update_output(arr, store_values=self.store_values)

    def symmetric_scale(self, dtype: str = "int8", use_percentile: bool = True) -> float:
        q_max = 127.0 if dtype == "int8" else 32767.0
        bound = self.stats.percentile_999 if use_percentile else self.stats.absmax
        if bound <= 0.0:
            return 1.0 / q_max
        return bound / q_max

    def output_symmetric_scale(self, dtype: str = "int8", use_percentile: bool = True) -> float:
        q_max = 127.0 if dtype == "int8" else 32767.0
        bound = (
            self.stats.output_percentile_999
            if use_percentile
            else self.stats.output_absmax
        )
        if bound <= 0.0:
            return 1.0 / q_max
        return bound / q_max

    def mse_optimal_scale(self, dtype: str = "int8", num_candidates: int = 100) -> float:
        if not self.stats.values_buffer:
            return self.symmetric_scale(dtype, use_percentile=False)
        q_max = 127.0 if dtype == "int8" else 32767.0
        vals = np.concatenate(self.stats.values_buffer)
        absmax = float(np.max(np.abs(vals)))
        if absmax <= 0.0:
            return 1.0 / q_max
        candidates = np.linspace(absmax / q_max * 0.5, absmax / q_max * 1.5, num_candidates)
        best_scale = candidates[0]
        best_mse = float("inf")
        for scale in candidates:
            q = np.clip(np.round(vals / scale), -q_max, q_max)
            recon = q * scale
            mse = float(np.mean((vals - recon) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_scale = float(scale)
        return best_scale
