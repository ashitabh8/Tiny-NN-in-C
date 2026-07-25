#!/usr/bin/env python3
"""Generate legal/implemented quant config markdown from the validator."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pytorch_to_c.quantization.quant_config import (  # noqa: E402
    iter_config_grid,
    config_label,
)

OUTPUT = ROOT / "docs" / "legal_quant_config.md"


def render_doc(in_features: int = 128) -> str:
    lines = [
        "# Legal QuantLinearConfig cells",
        "",
        "Auto-generated from `quant_config.py` — do not edit by hand.",
        "Regenerate: `python -m tools.gen_legal_config_doc`",
        "",
        f"Reference `in_features={in_features}` for per-group legality checks.",
        "",
        "int4 (`w_bits=4`) is exempt from `BLOCK_K=32` alignment on `input_group_size`",
        "(see unit test `test_int4_non_32_group_legal`).",
        "",
        "| Config | Legality | C | Triton |",
        "|--------|----------|---|--------|",
    ]
    for cfg, legal, c_impl, t_impl in iter_config_grid(in_features=in_features):
        label = config_label(cfg)
        lines.append(f"| `{label}` | {legal} | {c_impl} | {t_impl} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render_doc(), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
