"""QuantLinearConfig, legality/availability validation, and capability registry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple

BLOCK_K = 32


class AGran(Enum):
    PER_TENSOR = "per_tensor"
    PER_TOKEN = "per_token"


@dataclass(frozen=True)
class QuantLinearConfig:
    w_bits: int
    a_bits: int
    input_group_size: int
    per_out_column: bool
    a_gran: AGran
    w_symmetric: bool = True
    a_symmetric: bool = False
    dynamic_act: bool = False
    rounding: str = "rtn"


class IllegalConfig(Exception):
    """Permanent: config is unsound and must never be accepted."""


class UnimplementedConfig(Exception):
    """Temporary: config is legal but no codegen path exists on this backend yet."""


# Registry: backend -> list of (matcher, phase_label)
_CapabilityRegistry: Dict[str, List[Tuple[Callable[[QuantLinearConfig], bool], str]]] = {
    "c": [],
    "triton": [],
}


def register_capability(
    backend: str,
    matcher: Callable[[QuantLinearConfig], bool],
    phase_label: str = "implemented",
) -> None:
    """Register a config cell that a backend can emit."""
    _CapabilityRegistry.setdefault(backend, []).append((matcher, phase_label))


def _register_phase1_capabilities() -> None:
    """Seed registry with cells implemented in the current repo (Phase 1)."""

    def _w8a16_w16(w: int, a: int) -> Callable[[QuantLinearConfig], bool]:
        def m(c: QuantLinearConfig) -> bool:
            return c.w_bits == w and c.a_bits == a

        return m

    def _affine_dense(c: QuantLinearConfig) -> bool:
        return (
            (c.w_bits, c.a_bits) in ((8, 8), (16, 16))
            and c.a_gran == AGran.PER_TENSOR
            and c.rounding in ("rtn", "gptq")
        )

    def _w4a8(c: QuantLinearConfig) -> bool:
        return (
            c.w_bits == 4
            and c.a_bits == 8
            and c.per_out_column
            and c.a_gran == AGran.PER_TENSOR
            and c.rounding in ("rtn", "gptq")
        )

    for backend in ("c", "triton"):
        register_capability(backend, _affine_dense, "Phase 1")
        register_capability(backend, _w4a8, "Phase 1 (Triton: ref loop)" if backend == "triton" else "Phase 1")


_register_phase1_capabilities()


def _num_groups(config: QuantLinearConfig, in_features: int) -> int:
    if in_features <= 0:
        return 1
    g = config.input_group_size
    if g <= 0 or in_features % g != 0:
        raise IllegalConfig(
            f"input_group_size={g} must divide in_features={in_features}"
        )
    return in_features // g


def is_legal(config: QuantLinearConfig, *, in_features: int = 128) -> bool:
    try:
        _check_legality(config, in_features=in_features)
        return True
    except IllegalConfig:
        return False


def _check_legality(config: QuantLinearConfig, *, in_features: int = 128) -> None:
    if config.w_bits not in (4, 8, 16):
        raise IllegalConfig(f"w_bits={config.w_bits} must be 4, 8, or 16")
    if config.a_bits not in (4, 8, 16):
        raise IllegalConfig(f"a_bits={config.a_bits} must be 4, 8, or 16")
    if config.rounding not in ("rtn", "gptq"):
        raise IllegalConfig(f"rounding={config.rounding!r} must be 'rtn' or 'gptq'")
    if config.a_gran not in (AGran.PER_TENSOR, AGran.PER_TOKEN):
        raise IllegalConfig(
            f"a_gran={config.a_gran} illegal: only PER_TENSOR and PER_TOKEN are allowed"
        )

    # Pathological: activation precision below weight precision.
    if config.w_bits == 16 and config.a_bits in (4, 8):
        raise IllegalConfig(
            f"W{config.w_bits}A{config.a_bits} illegal: activation precision lower than weight"
        )

    if config.w_bits == 4 and config.a_bits not in (4, 8, 16):
        raise IllegalConfig(f"W4A{config.a_bits} not supported")
    if config.w_bits == 8 and config.a_bits not in (8, 16):
        raise IllegalConfig(f"W8A{config.a_bits} not supported")
    if config.w_bits == 16 and config.a_bits != 16:
        raise IllegalConfig(f"W16A{config.a_bits} not supported")

    if config.input_group_size <= 0:
        raise IllegalConfig("input_group_size must be positive")

    num_groups = _num_groups(config, in_features)

    # Per-group along input axis: int8/16 require BLOCK_K alignment; int4 exempt (MIGRATION §6).
    if num_groups > 1 and config.w_bits in (8, 16):
        if config.input_group_size % BLOCK_K != 0:
            raise IllegalConfig(
                f"input_group_size={config.input_group_size} must be a multiple of "
                f"BLOCK_K={BLOCK_K} for w_bits={config.w_bits} per-group quantization"
            )

    if num_groups > 1 and not config.per_out_column:
        raise IllegalConfig(
            "per-group weight scales require per_out_column=True "
            "(scale shape [num_groups, out_features])"
        )


def is_implemented(config: QuantLinearConfig, backend: str) -> bool:
    if backend not in _CapabilityRegistry:
        return False
    return any(m(config) for m, _ in _CapabilityRegistry[backend])


def _implemented_phase(config: QuantLinearConfig, backend: str) -> Optional[str]:
    for matcher, phase in _CapabilityRegistry.get(backend, []):
        if matcher(config):
            return phase
    return None


def validate(
    config: QuantLinearConfig,
    backend: str,
    *,
    in_features: int = 128,
) -> None:
    """Legality first, then per-backend availability. No silent fallback."""
    _check_legality(config, in_features=in_features)
    if config.a_gran == AGran.PER_TOKEN:
        raise UnimplementedConfig(
            f"{config} legal but PER_TOKEN not implemented on {backend}; target: Phase 4+"
        )
    if not is_implemented(config, backend):
        phase = _implemented_phase(config, backend)
        raise UnimplementedConfig(
            f"{config} not implemented on {backend}"
            + (f"; nearest: {phase}" if phase else "; no registered capability")
        )


def _bits_from_dtype(dtype: str) -> int:
    if dtype == "int8":
        return 8
    if dtype == "int16":
        return 16
    raise ValueError(f"Unsupported dtype for QuantLinearConfig: {dtype}")


def static_per_tensor_config(
    dtype: str,
    *,
    input_offset: int = 0,
    weight_offset: int = 0,
    in_features: int,
) -> QuantLinearConfig:
    bits = _bits_from_dtype(dtype)
    return QuantLinearConfig(
        w_bits=bits,
        a_bits=bits,
        input_group_size=in_features,
        per_out_column=False,
        a_gran=AGran.PER_TENSOR,
        w_symmetric=(weight_offset == 0),
        a_symmetric=(input_offset == 0),
        dynamic_act=False,
        rounding="rtn",
    )


def static_per_channel_config(
    dtype: str,
    *,
    input_offset: int = 0,
    weight_offset: int = 0,
    in_features: int,
) -> QuantLinearConfig:
    bits = _bits_from_dtype(dtype)
    return QuantLinearConfig(
        w_bits=bits,
        a_bits=bits,
        input_group_size=in_features,
        per_out_column=True,
        a_gran=AGran.PER_TENSOR,
        w_symmetric=(weight_offset == 0),
        a_symmetric=(input_offset == 0),
        dynamic_act=False,
        rounding="rtn",
    )


def static_per_group_config(
    dtype: str,
    *,
    input_offset: int = 0,
    weight_offset: int = 0,
    in_features: int,
    group_size: int = 32,
) -> QuantLinearConfig:
    bits = _bits_from_dtype(dtype)
    return QuantLinearConfig(
        w_bits=bits,
        a_bits=bits,
        input_group_size=group_size,
        per_out_column=True,
        a_gran=AGran.PER_TENSOR,
        w_symmetric=(weight_offset == 0),
        a_symmetric=(input_offset == 0),
        dynamic_act=False,
        rounding="rtn",
    )


def dynamic_per_tensor_config(
    dtype: str,
    *,
    weight_offset: int = 0,
    in_features: int,
) -> QuantLinearConfig:
    bits = _bits_from_dtype(dtype)
    return QuantLinearConfig(
        w_bits=bits,
        a_bits=bits,
        input_group_size=in_features,
        per_out_column=False,
        a_gran=AGran.PER_TENSOR,
        w_symmetric=(weight_offset == 0),
        a_symmetric=True,
        dynamic_act=True,
        rounding="rtn",
    )


def static_w4a8_config(
    *,
    input_offset: int = 0,
    weight_offset: int = 0,
    group_size: int,
) -> QuantLinearConfig:
    return QuantLinearConfig(
        w_bits=4,
        a_bits=8,
        input_group_size=group_size,
        per_out_column=True,
        a_gran=AGran.PER_TENSOR,
        w_symmetric=(weight_offset == 0),
        a_symmetric=(input_offset == 0),
        dynamic_act=False,
        rounding="rtn",
    )


def dynamic_w4a8_config(
    *,
    weight_offset: int = 0,
    group_size: int,
) -> QuantLinearConfig:
    return QuantLinearConfig(
        w_bits=4,
        a_bits=8,
        input_group_size=group_size,
        per_out_column=True,
        a_gran=AGran.PER_TENSOR,
        w_symmetric=(weight_offset == 0),
        a_symmetric=True,
        dynamic_act=True,
        rounding="rtn",
    )


def config_label(config: QuantLinearConfig) -> str:
    gran = "per_group" if config.input_group_size < 128 else (
        "per_channel" if config.per_out_column else "per_tensor"
    )
    return (
        f"W{config.w_bits}A{config.a_bits}_{gran}_"
        f"{'dyn' if config.dynamic_act else 'static'}_"
        f"{config.a_gran.value}"
    )


def iter_config_grid(
    in_features: int = 128,
) -> List[Tuple[QuantLinearConfig, str, str, str]]:
    """Yield (config, legality, c_impl, triton_impl) for doc generation."""
    rows: List[Tuple[QuantLinearConfig, str, str, str]] = []

    candidates = [
        QuantLinearConfig(8, 8, in_features, False, AGran.PER_TENSOR),
        QuantLinearConfig(8, 8, in_features, True, AGran.PER_TENSOR),
        QuantLinearConfig(8, 8, 32, True, AGran.PER_TENSOR),
        QuantLinearConfig(16, 16, in_features, False, AGran.PER_TENSOR),
        QuantLinearConfig(16, 16, in_features, True, AGran.PER_TENSOR),
        QuantLinearConfig(16, 16, 32, True, AGran.PER_TENSOR),
        QuantLinearConfig(8, 8, in_features, False, AGran.PER_TENSOR, dynamic_act=True),
        QuantLinearConfig(16, 16, in_features, False, AGran.PER_TENSOR, dynamic_act=True),
        QuantLinearConfig(4, 8, 32, True, AGran.PER_TENSOR),
        QuantLinearConfig(8, 8, in_features, False, AGran.PER_TOKEN),
        QuantLinearConfig(4, 16, 32, True, AGran.PER_TENSOR),
        QuantLinearConfig(8, 16, in_features, False, AGran.PER_TENSOR),
        QuantLinearConfig(16, 8, in_features, False, AGran.PER_TENSOR),
        QuantLinearConfig(4, 4, 32, True, AGran.PER_TENSOR),
        QuantLinearConfig(8, 8, 16, True, AGran.PER_TENSOR),  # illegal: non-32 per-group
    ]

    for cfg in candidates:
        try:
            _check_legality(cfg, in_features=in_features)
            legal = "legal"
        except IllegalConfig as e:
            rows.append((cfg, f"illegal: {e}", "—", "—"))
            continue

        c_status = "implemented" if is_implemented(cfg, "c") else "unimplemented"
        t_status = "implemented" if is_implemented(cfg, "triton") else "unimplemented"
        if cfg.a_gran == AGran.PER_TOKEN:
            c_status = "unimplemented (Phase 4+)"
            t_status = "unimplemented (Phase 4+)"
        rows.append((cfg, legal, c_status, t_status))

    return rows
