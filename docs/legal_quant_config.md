# Legal QuantLinearConfig cells

Auto-generated from `quant_config.py` — do not edit by hand.
Regenerate: `python -m tools.gen_legal_config_doc`

Reference `in_features=128` for per-group legality checks.

int4 (`w_bits=4`) is exempt from `BLOCK_K=32` alignment on `input_group_size`
(see unit test `test_int4_non_32_group_legal`).

| Config | Legality | C | Triton |
|--------|----------|---|--------|
| `W8A8_per_tensor_static_per_tensor` | legal | implemented | implemented |
| `W8A8_per_channel_static_per_tensor` | legal | implemented | implemented |
| `W8A8_per_group_static_per_tensor` | legal | implemented | implemented |
| `W16A16_per_tensor_static_per_tensor` | legal | implemented | implemented |
| `W16A16_per_channel_static_per_tensor` | legal | implemented | implemented |
| `W16A16_per_group_static_per_tensor` | legal | implemented | implemented |
| `W8A8_per_tensor_dyn_per_tensor` | legal | implemented | implemented |
| `W16A16_per_tensor_dyn_per_tensor` | legal | implemented | implemented |
| `W4A8_per_group_static_per_tensor` | legal | implemented | implemented |
| `W8A8_per_tensor_static_per_token` | legal | unimplemented (Phase 4+) | unimplemented (Phase 4+) |
| `W4A16_per_group_static_per_tensor` | legal | unimplemented | unimplemented |
| `W8A16_per_tensor_static_per_tensor` | legal | unimplemented | unimplemented |
| `W16A8_per_tensor_static_per_tensor` | illegal: W16A8 illegal: activation precision lower than weight | — | — |
| `W4A4_per_group_static_per_tensor` | legal | unimplemented | unimplemented |
| `W8A8_per_group_static_per_tensor` | illegal: input_group_size=16 must be a multiple of BLOCK_K=32 for w_bits=8 per-group quantization | — | — |
