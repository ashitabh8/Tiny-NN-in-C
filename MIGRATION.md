# MIGRATION.md — Parameterized Quantization + TensorFacts

Phase 0 audit (no code changes). Baseline for Phases 1–8 of the v2 migration plan.

**Audit date:** 2026-07-10  
**Branch:** current workspace  
**Spec:** Cursor Command v2 (parameterized quant + fact-carrying IR)

---

## 1. Executive summary

Tiny-NN-in-C already centralizes quantization wiring in the right places:

- **Rules** create nodes (`QuantRule.create_quant_node` / `create_quant_node_with_weights`)
- **Nodes** own pre/post (`get_pre_nodes` / `get_post_nodes`) and kernel dispatch (`generate_c_code` / `generate_triton_code`)
- **`QuantizationTransform`** inserts parallel branches → pre/post → quantizes weights → validates

The migration consolidates **affine linear** quantization into one `QuantLinearNode` + `QuantLinearConfig` with template-generated C/Triton kernels. **Carve-outs** (weight-only, palette, LQER composition, conv through Phase 7) stay on existing node families until Phase 8.

**Regression contract frozen:** 43 tests in three files, all passing at audit time.

```bash
pytest test/test_compression_quant.py test/test_lqer.py test/test_quantization_e2e.py
# 43 passed in 3.95s
```

---

## 2. Regression baseline

### 2.1 Test files and tolerances

| File | Tests | Primary gate | Tolerance |
|------|-------|--------------|-----------|
| [`test/test_compression_quant.py`](test/test_compression_quant.py) | 14 | C `verify_model` for per-group, int4, palette; graph structure | `5.0` max abs error per sample |
| [`test/test_lqer.py`](test/test_lqer.py) | 20 | Graph wiring, OpenMP, buffer aliasing; relative LQER improvement | `1e9` (relative comparison, not absolute pass/fail) |
| [`test/test_quantization_e2e.py`](test/test_quantization_e2e.py) | 9 | Custom harness: SimpleMLP + TinyResNet static/dynamic/mixed | `1e-1` quant; `1e-5` float ResNet |

### 2.2 `verify_model` defaults

| Parameter | Default | Quant tests use |
|-----------|---------|-----------------|
| `tolerance` | `1e-3` | `5.0` (compression_quant, palette) |
| `num_samples` | 20 | 5–20 |
| `openmp` | `False` | `True` in LQER OpenMP equivalence test |

### 2.3 Formats exercised by regression suite

| Format | Rule(s) | Backend tested |
|--------|---------|----------------|
| W8A8 static per-tensor | `StaticQuantRule` | C (e2e, LQER, fuse) |
| W16A16 static per-tensor | `StaticQuantRule` | C (e2e) |
| W8A8 dynamic per-tensor | `DynamicQuantRuleMinMaxPerTensor` | C (e2e, LQER) |
| W16A16 dynamic per-tensor | `DynamicQuantRuleMinMaxPerTensor` | C (e2e) |
| Mixed W8A8 + W16A16 | two `StaticQuantRule` | C (e2e) |
| W8A8 per-group | `StaticPerGroupLinearQuantRule` | C |
| W4A8 static per-group | `StaticInt4PerGroupLinearQuantRule` | C |
| W4A8 dynamic per-group | `DynamicInt4PerGroupLinearQuantRule` | C |
| Weight palettization | `PaletteWeightRule` | C |
| GPTQ helper | `gptq_quantize` unit test | Python only |
| LQER static/dynamic linear+conv | `LQERStaticQuantRule`, `LQERDynamicQuantRule` | C |
| Conv static/dynamic (ResNet e2e) | `StaticQuantRule`, `DynamicQuantRuleMinMaxPerTensor` | C |

**Not in regression suite today:** `StaticPerChannelLinearQuantRule`, `Int8WeightOnlyLinearRule`, Triton `verify_model_triton` for any quant format.

### 2.4 Parity gate for later phases

Every phase that touches numerics must reproduce this baseline:

```bash
python3 -m pytest test/test_compression_quant.py test/test_lqer.py test/test_quantization_e2e.py -q
```

Additional gates per phase are listed in §10.

---

## 3. Current inventory

### 3.1 Public rules (`src/pytorch_to_c/quantization/rules.py`)

| Rule | `op_type` | Affine linear target? | Phase 2 wrapper? |
|------|-----------|----------------------|------------------|
| `StaticQuantRule` | `linear`, `conv2d`/`conv1d` | linear only → `QuantLinearNode`; conv → old node | Partial (linear only) |
| `StaticPerChannelLinearQuantRule` | `linear` | Yes | Yes |
| `StaticPerChannelConvQuantRule` | `conv2d`/`conv1d` | No (conv carve-out) | Phase 8 |
| `StaticPerGroupLinearQuantRule` | `linear` | Yes | Yes |
| `StaticInt4PerGroupLinearQuantRule` | `linear` | Yes (`w_bits=4`) | Yes |
| `DynamicInt4PerGroupLinearQuantRule` | `linear` | Yes (`w_bits=4`, dynamic) | Yes |
| `DynamicQuantRuleMinMaxPerTensor` | `linear`, `conv2d`/`conv1d` | linear only; conv → old node | Partial |
| `Int8WeightOnlyLinearRule` | `linear` | **Carve-out** | No |
| `PaletteWeightRule` | `linear` | **Carve-out** | No |
| `LQERStaticQuantRule` | `linear`, `conv2d` | Compositional diamond | Phase 2 implicit (linear quant leg) |
| `LQERDynamicQuantRule` | `linear`, `conv2d` | Compositional diamond | Phase 2 implicit (linear quant leg) |

### 3.2 Node classes (`src/pytorch_to_c/quantization/ops/`)

**Affine linear (migrate → `QuantLinearNode`):**

| Node | Rule(s) | `quant_strategy` |
|------|---------|------------------|
| `StaticQuantLinearNode` | `StaticQuantRule` | `static` |
| `StaticPerChannelQuantLinearNode` | `StaticPerChannelLinearQuantRule` | `static` |
| `StaticPerGroupQuantLinearNode` | `StaticPerGroupLinearQuantRule` | `static` |
| `DynamicQuantLinearNode` | `DynamicQuantRuleMinMaxPerTensor` | `dynamic` |
| `StaticInt4PerGroupQuantLinearNode` | `StaticInt4PerGroupLinearQuantRule` | `static_int4_per_group` |
| `DynamicInt4PerGroupQuantLinearNode` | `DynamicInt4PerGroupLinearQuantRule` | `dynamic_int4_per_group` |

**Carve-outs (unchanged through Phase 7):**

| Node | Rule |
|------|------|
| `Int8WeightOnlyLinearNode` | `Int8WeightOnlyLinearRule` |
| `PaletteWeightLinearNode` | `PaletteWeightRule` |
| `StaticQuantConv2dNode`, `StaticPerChannelQuantConv2dNode`, `DynamicQuantConv2dNode` | conv rules |
| `LQER*QuantLinearNode`, `LQER*QuantConv2dNode` | LQER rules |

**Pre/post utility nodes (unchanged):**

| Node | Role |
|------|------|
| `QuantizeNode` | static activation quant |
| `DynamicQuantizeInputNode` | runtime activation scale + quant |
| `DequantizeNode` | static output dequant |

### 3.3 C kernels (`src/c_ops/`)

**Linear affine (template target in Phase 3):**

| Kernel | Node(s) | Scale layout |
|--------|---------|--------------|
| `dense_int8` / `dense_int16` | `StaticQuantLinearNode` | scalar `weight_scale` |
| `dense_int8_per_channel` / `dense_int16_per_channel` | `StaticPerChannelQuantLinearNode` | `[out_features]` |
| `dense_int8_per_group` / `dense_int16_per_group` | `StaticPerGroupQuantLinearNode` | `[num_groups × out_features]` |
| `dense_int8_to_float` / `dense_int16_to_float` | `DynamicQuantLinearNode` | scalar |
| `dense_int8_int4w_per_group` | `StaticInt4PerGroupQuantLinearNode` | packed int4 + per-group scales |
| `dense_int8_int4w_per_group_to_float` | `DynamicInt4PerGroupQuantLinearNode` | packed int4 + per-group scales |

**Carve-out C kernels (not in Phase 3 template):**

| Kernel | Node |
|--------|------|
| `dense_float_input_int8_weight_per_channel` | `Int8WeightOnlyLinearNode` |
| `dense_float_palettized` | `PaletteWeightLinearNode` |
| `conv2d_nhwc_*`, `depthwise_conv2d_nhwc_*` (+ per_channel, to_float) | conv nodes |

All affine dense kernels use the **four-term zero-point cross-expansion** when any of `input_zp`, `weight_zp`, `output_zp` are non-zero (see `dense_int8_per_group` in `nn_ops_int8.h`).

### 3.4 Triton kernels (`src/triton_ops/nn_ops_quant.py`)

| Category | Functions | Implementation |
|----------|-----------|----------------|
| Pre/post | `quantize_float_to_int*`, `dequantize_*`, `compute_dynamic_scale_*` | Real `@triton.jit` |
| Dense W8A8/W16A16 | `dense_int8`, `dense_int16`, `*_per_channel`, `*_per_group`, `*_to_float` | Real `@triton.jit` |
| Dense W4A8 | `dense_int8_int4w_per_group`, `dense_int8_int4w_per_group_to_float` | **Python reference loops** |
| Conv / depthwise | `conv2d_nhwc_*`, `depthwise_conv2d_nhwc_*` | **Python reference loops** (`_conv2d_quant_ref`) |
| Palette | `dense_float_palettized` | **Python reference loop** |
| Weight-only | `dense_float_input_int8_weight_per_channel` | **MISSING** (C node emits call; Triton file has no definition) |

Fixed `BLOCK=256` grid on most Triton launchers. No `tl.make_block_ptr`, `tl.assume`, `tl.multiple_of`, or `@triton.autotune` today.

---

## 4. `QuantLinearConfig` mapping

Proposed config (Phase 1):

```python
@dataclass(frozen=True)
class QuantLinearConfig:
    w_bits: int             # 4 | 8 | 16
    a_bits: int             # 4 | 8 | 16
    input_group_size: int   # in_features => 1 group; else multiple of 32 (int8/16 per-group)
    per_out_column: bool    # output-axis scale vector or [group, col] scales
    a_gran: AGran           # PER_TENSOR | PER_TOKEN
    w_symmetric: bool = True
    a_symmetric: bool = False
    dynamic_act: bool = False
    rounding: str = "rtn"   # "rtn" | "gptq"
```

### 4.1 Rule → config (affine linear)

| Rule | `w_bits` | `a_bits` | `input_group_size` | `per_out_column` | `a_gran` | `dynamic_act` | `rounding` | Notes |
|------|----------|----------|-------------------|------------------|----------|---------------|------------|-------|
| `StaticQuantRule` (linear) | 8/16 | 8/16 | `in_features` | `False` | `PER_TENSOR` | `False` | `rtn` | scalar `weight_scale` from rule |
| `StaticPerChannelLinearQuantRule` | 8/16 | 8/16 | `in_features` | `True` | `PER_TENSOR` | `False` | `rtn` | compile-time per-column scales |
| `StaticPerGroupLinearQuantRule` | 8/16 | 8/16 | `g` (from `select_group_size`) | `True` | `PER_TENSOR` | `False` | `rtn`/`gptq` | `g % 32 == 0` enforced |
| `DynamicQuantRuleMinMaxPerTensor` (linear) | 8/16 | 8/16 | `in_features` | `False` | `PER_TENSOR` | `True` | `rtn` | weight scale from absmax |
| `StaticInt4PerGroupLinearQuantRule` | 4 | 8 | `g` (see §6) | `True` | `PER_TENSOR` | `False` | `rtn`/`gptq` | packed weights in `quantize_weights` |
| `DynamicInt4PerGroupLinearQuantRule` | 4 | 8 | `g` (see §6) | `True` | `PER_TENSOR` | `True` | `rtn`/`gptq` | |

`input_group_size` and `per_out_column` at runtime come from `metadata['group_size']` (per-group/int4) or `in_features` (per-tensor/per-channel). Wrappers set config fields; node stores resolved values after `quantize_weights`.

### 4.2 Weight granularity as two knobs

| Named granularity | `input_group_size` | `per_out_column` | Scale shape | Today's C kernel |
|-------------------|-------------------|------------------|-------------|------------------|
| PER_TENSOR | `in_features` | `False` | scalar | `dense_int8` |
| PER_CHANNEL | `in_features` | `True` | `[out_features]` | `dense_int8_per_channel` |
| PER_GROUP(g) | `g` | `True` | `[num_groups, out_features]` | `dense_int8_per_group` |

Phase 3 must prove one template covers all three via scale-indexing, without numerics change.

---

## 5. Offset audit → `w_symmetric` / `a_symmetric`

### 5.1 API surface (all rules accept offsets)

| Rule | `input_offset` | `weight_offset` | `output_offset` | Default |
|------|----------------|-----------------|-----------------|---------|
| `StaticQuantRule` | required | required | required | user-supplied |
| `StaticPerChannelLinearQuantRule` | required | optional | required | `weight_offset=0` |
| `StaticPerGroupLinearQuantRule` | required | optional | required | `weight_offset=0` |
| `StaticInt4PerGroupLinearQuantRule` | required | optional | required | `weight_offset=0` |
| `DynamicInt4PerGroupLinearQuantRule` | — | optional | — | `weight_offset=0` |
| `Int8WeightOnlyLinearRule` | — | optional | — | `weight_offset=0` |
| `LQERStaticQuantRule` | inherits `StaticQuantRule` | | | |
| `DynamicQuantRuleMinMaxPerTensor` | N/A (runtime act) | computed | N/A | weight `offset=0` always |

### 5.2 Test / docs usage

**All regression tests use offset 0** for input, weight, and output.

The only non-zero offset test is [`test/test_quantization.py::test_get_quant_params`](test/test_quantization.py) which sets `input_offset=weight_offset=output_offset=5` to verify `get_quant_params()` serialization — **no E2E numerics test for asymmetric offsets**.

### 5.3 Wrapper symmetric-flag derivation (Phase 2 contract)

```python
w_symmetric = (weight_offset == 0)
a_symmetric = (input_offset == 0)   # static only; dynamic uses runtime symmetric scale (offset 0)
```

When any flag is `False`, the template must emit the **existing four-term ZP expansion** (current `dense_*` behavior). When all True, the fast symmetric path may skip cross-terms (new configs only; existing wrappers pass offsets through unchanged).

**LQER residual gate (post-Phase 4):** for each migrated rule, assert `max(abs(W - dequant(quant(W))))` unchanged before/after template swap.

---

## 6. Int4 `BLOCK_K` asymmetry (preserve)

| Path | Group-size selection | `BLOCK_K=32` enforced? |
|------|---------------------|------------------------|
| `StaticPerGroupLinearQuantRule` | `select_group_size()` | **Yes** — candidates are multiples of 32 |
| `StaticInt4PerGroupLinearQuantRule` / `DynamicInt4PerGroupLinearQuantRule` | `_pack_int4_per_group_weights()` | **No** — if `in_features % group_size != 0`, falls back to `g = in_features` |

Phase 2 int4 wrappers must **not** introduce `select_group_size()` for int4. Document in validator: int4 `input_group_size` need not be 32-aligned.

---

## 7. Pre/post derivation (Phase 1 invariant)

The general node derives pre/post; `QuantizationTransform` does not decide.

| Config | Pre node | Post node | Node `dtype` after matmul |
|--------|----------|-----------|---------------------------|
| static, `a_bits ∈ {8,16}`, `dynamic_act=False` | `QuantizeNode(a_bits)` | `DequantizeNode(a_bits)` | `int8` / `int16` |
| `dynamic_act=True` | `DynamicQuantizeInputNode(a_bits)` | none | `float32` |

Weight-only and palette: **no pre/post** (float activations throughout).

LQER: inherits pre/post from quantized leg; parallel branch taps **original float input** (before pre nodes); static join is after `DequantizeNode`.

---

## 8. Carve-outs (explicit)

| Format | Stays outside `QuantLinearConfig` | Through Phase 7 | Phase 8 |
|--------|-----------------------------------|-----------------|---------|
| `Int8WeightOnlyLinearRule` | float activations, per-channel weight scales | unchanged | optional future `encoding=weight_only` |
| `PaletteWeightRule` | codebook/index encoding | unchanged | stay separate |
| Conv quant rules/nodes | HWIO, depthwise, groups | unchanged | `QuantConvNode` sibling |
| LQER | `get_parallel_branch()` diamond | quant leg on `QuantLinearNode` (Phase 2); conv leg Phase 8 | full rebase + e2e |

---

## 9. LQER two-stage rebase

| Stage | When | Action |
|-------|------|--------|
| **Implicit** | Phase 2 | LQER linear rules construct `QuantLinearNode` (or shell delegating to it) for the quantized leg. Kernels unchanged. Branch/join unchanged. |
| **Explicit** | Post-Phase 4 | Remove shells; LQER mixins target `QuantLinearNode` only. Gate: `dequant(quant(W))` residual + full regression suite. |
| **Conv-LQER** | Phase 8 | `LQER*QuantConv2dNode` on `QuantConvNode`. |

---

## 10. Phase 2 codegen delegation contract

Until Phase 3/4 template land, `QuantLinearNode.generate_c_code()` and `generate_triton_code()` must emit **byte-identical call strings** to today's specialized nodes for each config point.

Example mappings (W8A8):

| Config point | C kernel string |
|--------------|-----------------|
| per-tensor static | `dense_int8(...)` |
| per-channel static | `dense_int8_per_channel(...)` |
| per-group static | `dense_int8_per_group(...)` |
| dynamic | `dense_int8_to_float(...)` |
| W4A8 static | `dense_int8_int4w_per_group(...)` |
| W4A8 dynamic | `dense_int8_int4w_per_group_to_float(...)` |

Phase 2 parity gate = regression suite green with delegation, zero kernel file changes.

---

## 11. Triton gap inventory (Phase 4 scope)

| Cell | C status | Triton status | Phase 4 action |
|------|----------|---------------|----------------|
| W8A8/W16A16 dense (all granularities) | mature | real `@triton.jit` | restructure into template |
| W4A8 dense | mature | Python ref loop | **build** `@triton.jit` at ref parity |
| Conv / depthwise quant | mature | Python ref loop | template in Phase 8 (conv carve-out) |
| Weight-only | C only | **missing function** | build or keep carve-out |
| Palette | C only | Python ref loop | carve-out; optional later |

---

## 12. Legal vs implemented matrix (Phase 1 validator preview)

Validator exposes **two predicates**: `is_legal(config)` and `is_implemented(config)`.

| Cell | Legal | Implemented (today) | Notes |
|------|-------|---------------------|-------|
| W8A8, per-tensor/channel/group, static | yes | yes (C); yes (Triton dense) | primary |
| W16A16, same | yes | yes | |
| W4A8, per-group, static/dynamic | yes | yes (C); ref (Triton) | |
| W4A16 / W8A16 | yes | partial | higher-precision act paths |
| W4A4 | conditional | no | tensor-core availability |
| W16A8 / W16A4 | no (default) | no | pathological; override flag later |
| `a_gran=PER_TENSOR` | yes | yes | |
| `a_gran=PER_TOKEN` | yes | **no** | legal, not implemented |
| reduction-axis `a_gran` | **no** | no | illegal — cannot hoist from int accum |
| per-group activation (reduction) | no | no | deferred `NotImplementedError` |

---

## 13. `StaticQuantRule` linear/conv split (Phase 2)

`StaticQuantRule.create_quant_node()` and `DynamicQuantRuleMinMaxPerTensor.create_quant_node_with_weights()` branch on `op_type`:

```python
if node.op_type == 'linear':
    return QuantLinearNode(...)      # Phase 2
elif node.op_type in ('conv2d', 'conv1d'):
    return StaticQuantConv2dNode(...)  # unchanged through Phase 7
```

Same pattern for `LQERStaticQuantRule` / `LQERDynamicQuantRule`: linear → new quant leg; conv2d → old conv LQER nodes until Phase 8.

---

## 14. Phase roadmap and gates

| Phase | Deliverable | Gate |
|-------|-------------|------|
| **0** | This document | Review approved |
| **1** | `QuantLinearConfig`, `validate()`, `QuantLinearNode` shell, pre/post, legal doc | validator unit tests green |
| **2** | Affine-linear rule wrappers + LQER linear implicit rebase | 43 regression tests green; delegation strings match §10 |
| **3** | C template; delete redundant `dense_*` after parity | regression + `evaluate_compression.py` |
| **4** | Triton template; real JIT for W4A8 ref cells | `verify_model_triton` on CUDA |
| **5** | `TensorFacts` populate + debug asserts | fact unit tests |
| **6** | Fact lowering + lint | numerics vs Phase 4; TTGIR inspection |
| **7** | Autotune cache | `benchmark_triton` ≥ baseline |
| **8** | `QuantConvNode`, conv-LQER, docs, e2e prune→quant→LQER | conv bit-parity + `results.summary()` clean |

---

## 15. Files touched by migration (reference)

| Phase | Primary files |
|-------|---------------|
| 1 | new `src/pytorch_to_c/quantization/quant_config.py`, `ops/quant_linear_general.py` (or similar) |
| 2 | `rules.py` (wrapper bodies only) |
| 3 | `src/c_ops/nn_ops_int{8,16,4}.h`, node `generate_c_code` |
| 4 | `src/triton_ops/nn_ops_quant.py`, node `generate_triton_code` |
| 5–6 | new `src/pytorch_to_c/ir/facts.py`, `codegen/*_printer.py` |
| 7 | `tools/benchmark_triton.py` |
| 8 | `ops/quant_conv2d.py`, `ops/quant_LQER.py`, `README_FEATURES.md` |

---

## 16. Open questions resolved in v2

| Question | Decision |
|----------|----------|
| Weight granularity model | Two knobs: `input_group_size` + `per_out_column` |
| `w_symmetric` default | Per-config; wrappers mirror offset behavior |
| `PER_TOKEN` | Legal but not implemented; separate validator predicate |
| LQER timing | Phase 2 implicit on linear; explicit post-Phase 4; conv Phase 8 |
| Conv scope | Untouched Phases 1–7; consolidate Phase 8 |
| `TechniqueRegistry` | Out of scope (not in repo) |

**Phase 0 complete. Stop for review before Phase 1.**

---

## Addendum A — Phase 1 characterization baselines (captured 2026-07-10)

Part A baselines were captured against **current (old-node) code** on this machine before any Phase 1 B work. Use `torch.manual_seed(42)`, `TinyMLP(32,16,4)`, `num_samples=20`.

### A1 — Per-channel linear C (`StaticPerChannelLinearQuantRule`)

| Cell | `BASELINE` (max abs error) | `MARGIN` | Suite tol | C kernel delegation target |
|------|---------------------------|----------|-----------|---------------------------|
| W8A8 per-channel | `0.033286` | `0.1` | `5.0` | `dense_int8_per_channel(...)` |
| W16A16 per-channel | `0.004679` | `0.1` | `5.0` | `dense_int16_per_channel(...)` |

**Phase 3 fidelity gate:** `max_abs_error <= BASELINE + MARGIN` per cell (stricter than suite tolerance).

**Test file:** `test/test_perchannel_baseline.py` (to be added in Part A commit).

### A2 — Triton quant static dense (`verify_model_triton`, real `@triton.jit`)

| Cell | `SNAPSHOT` (max abs error) | `MARGIN` | Status |
|------|---------------------------|----------|--------|
| W8A8 static per-tensor | `0.10852346` | `0.1` | passes |
| W16A16 static per-tensor | `0.01339865` | `0.1` | passes |
| W8A8 static per-channel | `0.03282639` | `0.1` | passes |
| W16A16 static per-channel | `0.00341088` | `0.1` | passes |
| W8A8 static per-group | `0.03282639` | `0.1` | passes |
| W16A16 static per-group | `0.00341088` | `0.1` | passes |
| W8A8 dynamic `*_to_float` | — | — | **pre-existing kernel bug** |
| W16A16 dynamic `*_to_float` | — | — | **pre-existing kernel bug** |

**Dynamic Triton bug (pre-Phase 1):** `_dense_quant_to_float_kernel` fails at compile time:
`Loop-carried variable acc has initial type int32 but is re-assigned to int64 in loop`.
Phase 4 must fix this before a dynamic snapshot can be recorded. Test should assert the
failure today; inverts to fidelity gate once fixed.

**Excluded from Triton baseline (by design):** W4A8, conv, palette — Python ref loops.

**Phase 4 fidelity gate:** `template_error <= SNAPSHOT + MARGIN` per static cell.

**Test file:** `test/test_quant_triton_baseline.py` (to be added in Part A commit).

### Updated regression contract (post-Part A)

| Suite | Tests |
|-------|-------|
| Original baseline | 43 |
| `test_perchannel_baseline.py` | 2 |
| `test_quant_triton_baseline.py` | 8 (6 static + 2 dynamic-bug) |
| `test_quant_config.py` (Part B) | 18 |
| **Total after Phase 1** | **71** |

**Phase 1 acceptance (2026-07-10):** all 71 tests green; `docs/legal_quant_config.md` generated.


