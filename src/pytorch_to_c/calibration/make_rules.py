"""Generate quantization rules from calibration statistics."""

from __future__ import annotations

from typing import List, Literal, Optional, Union

from ..quantization.rules import (
    QuantRule,
    StaticPerChannelConvQuantRule,
    StaticPerChannelLinearQuantRule,
    StaticPerGroupLinearQuantRule,
    StaticQuantRule,
)
from .calibrate import CalibrationStats


Granularity = Literal["per_tensor", "per_channel", "per_group"]


def make_static_rules(
    stats: CalibrationStats,
    dtype: str = "int8",
    granularity: Granularity = "per_channel",
    pattern: str = r".*(conv|fc|linear).*",
    default_input_scale: Optional[float] = None,
    default_output_scale: Optional[float] = None,
    group_size: Union[int, str] = "auto",
    error_budget: Optional[float] = None,
) -> List[QuantRule]:
    """
    Build StaticQuantRule list with calibrated activation scales.

    Uses layer name patterns; input/output scales come from calibration stats
    when available, else symmetric defaults.
    """
    q_max = 127.0 if dtype == "int8" else 32767.0
    fallback = 1.0 / q_max
    in_scale = default_input_scale or fallback
    out_scale = default_output_scale or fallback

    if stats.layer_stats:
        first = next(iter(stats.layer_stats.values()))
        in_scale = stats.get_input_scale(first.name, dtype)
        out_scale = in_scale

    rules: List[QuantRule] = []

    if granularity == "per_tensor":
        rules.append(
            StaticQuantRule(
                pattern=pattern,
                dtype=dtype,
                input_scale=in_scale,
                input_offset=0,
                weight_scale=in_scale,
                weight_offset=0,
                output_scale=out_scale,
                output_offset=0,
            )
        )
    elif granularity == "per_channel":
        rules.extend(
            [
                StaticPerChannelLinearQuantRule(
                    pattern=r".*(fc|linear).*",
                    dtype=dtype,
                    input_scale=in_scale,
                    input_offset=0,
                    output_scale=out_scale,
                    output_offset=0,
                ),
                StaticPerChannelConvQuantRule(
                    pattern=r".*conv.*",
                    dtype=dtype,
                    input_scale=in_scale,
                    input_offset=0,
                    output_scale=out_scale,
                    output_offset=0,
                ),
            ]
        )
    elif granularity == "per_group":
        rules.append(
            StaticPerGroupLinearQuantRule(
                pattern=r".*(fc|linear).*",
                dtype=dtype,
                input_scale=in_scale,
                input_offset=0,
                output_scale=out_scale,
                output_offset=0,
                group_size=group_size,
                error_budget=error_budget,
            )
        )
        rules.append(
            StaticPerChannelConvQuantRule(
                pattern=r".*conv.*",
                dtype=dtype,
                input_scale=in_scale,
                input_offset=0,
                output_scale=out_scale,
                output_offset=0,
            )
        )
    else:
        raise ValueError(f"Unknown granularity: {granularity}")

    return rules
