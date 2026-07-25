"""Run calibration over sample inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from .observer import ActivationObserver, LayerActivationStats


@dataclass
class CalibrationStats:
    """Collected calibration statistics."""

    layer_stats: Dict[str, LayerActivationStats] = field(default_factory=dict)
    hessians: Dict[str, np.ndarray] = field(default_factory=dict)
    num_batches: int = 0

    def get_input_scale(self, layer_name: str, dtype: str = "int8") -> float:
        stats = self.layer_stats.get(layer_name)
        if stats is None:
            q_max = 127.0 if dtype == "int8" else 32767.0
            return 1.0 / q_max
        obs = ActivationObserver(layer_name)
        obs.stats = stats
        return obs.symmetric_scale(dtype, use_percentile=True)

    def get_output_scale(self, layer_name: str, dtype: str = "int8") -> float:
        stats = self.layer_stats.get(layer_name)
        if stats is None:
            q_max = 127.0 if dtype == "int8" else 32767.0
            return 1.0 / q_max
        obs = ActivationObserver(layer_name)
        obs.stats = stats
        if stats.output_num_samples > 0 or stats.output_absmax > 0.0:
            return obs.output_symmetric_scale(dtype, use_percentile=True)
        return obs.symmetric_scale(dtype, use_percentile=True)


def _iter_batches(data: Union[Iterator, List], max_batches: Optional[int]) -> Iterator:
    count = 0
    for batch in data:
        yield batch
        count += 1
        if max_batches is not None and count >= max_batches:
            break


def calibrate(
    model: nn.Module,
    data_iter: Union[Iterator[torch.Tensor], List[torch.Tensor]],
    collect_inputs: bool = True,
    collect_hessian: bool = False,
    max_batches: Optional[int] = 32,
    store_values: bool = False,
) -> CalibrationStats:
    """
    Calibrate activation ranges (and optionally Hessians) from sample inputs.

    Hooks are registered on Linear, Conv2d, and Conv1d modules by module name.
    """
    model = model.eval()
    stats = CalibrationStats()
    observers: Dict[str, ActivationObserver] = {}
    hooks: List[torch.utils.hooks.RemovableHandle] = []
    hessian_acc: Dict[str, np.ndarray] = {}

    def _register(module: nn.Module, name: str) -> None:
        if name in observers:
            return
        obs = ActivationObserver(name, store_values=store_values)
        observers[name] = obs
        stats.layer_stats[name] = obs.stats

        def pre_hook(_mod, inp):
            if collect_inputs and inp and inp[0] is not None:
                obs.observe(inp[0])
            if collect_hessian and isinstance(_mod, nn.Linear) and inp and inp[0] is not None:
                x = inp[0].detach()
                if x.dim() > 2:
                    x = x.reshape(-1, x.shape[-1])
                x_np = x.cpu().double().numpy()
                batch_xt_x = x_np.T @ x_np
                if name not in hessian_acc:
                    hessian_acc[name] = batch_xt_x
                else:
                    hessian_acc[name] = hessian_acc[name] + batch_xt_x

        def post_hook(_mod, _inp, out):
            if collect_inputs and isinstance(_mod, (nn.Conv2d, nn.Conv1d)) and out is not None:
                obs.observe_output(out)
            elif not collect_inputs:
                obs.observe(out)

        hooks.append(module.register_forward_pre_hook(pre_hook))
        hooks.append(module.register_forward_hook(post_hook))

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            _register(module, name if name else "root")

    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in _iter_batches(data_iter, max_batches):
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)
            model(x)
            stats.num_batches += 1

    for hook in hooks:
        hook.remove()

    for obs in observers.values():
        obs.stats.finalize_percentile()

    if collect_hessian:
        stats.hessians = {k: v.astype(np.float64) for k, v in hessian_acc.items()}

    return stats
