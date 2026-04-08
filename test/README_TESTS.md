# Test Suite

## Test Files

| File | What it tests |
|------|---------------|
| `test_models.py` | Canonical model imports (`TinyMLP`, `ResNetBlock`, `MixedNet`) |
| `test_frontend.py` | `torch.fx` tracing |
| `test_lowering.py` | FX graph to IR conversion |
| `test_codegen.py` | C code generation (`CPrinter`) |
| `test_integration.py` | End-to-end compile + gcc build + PyTorch vs C comparison |
| `test_passes.py` | `FuseDequantQuantPass` |
| `test_quantization.py` | Quantization rules and matcher |
| `test_quantization_integration.py` | Quantization transform + codegen + gcc compile |
| `test_quantization_e2e.py` | Full quantized pipeline (MLP + ResNet) |
| `test_verify_harness.py` | Verification tool on float and quantized models |
| `test_c_ops.c` | Standalone C unit tests for `nn_ops_float.h` |
| `test_c_ops_int8.c` | Standalone C unit tests for `nn_ops_int8.h` |
| `test_c_ops_int16.c` | Standalone C unit tests for `nn_ops_int16.h` |

## Running Tests

```bash
# All Python tests
pytest test/ -v

# Just the verification harness (requires gcc)
pytest test/test_verify_harness.py -v

# Quantization end-to-end
pytest test/test_quantization_e2e.py -v

# Skip gcc-dependent tests
pytest test/ -v -k "not TestPyTorchCComparison and not TestVerify"
```

### Standalone C tests

```bash
gcc -o test_c_ops test/test_c_ops.c -Isrc/c_ops -lm && ./test_c_ops
gcc -o test_c_ops_int8 test/test_c_ops_int8.c -Isrc/c_ops -lm && ./test_c_ops_int8
gcc -o test_c_ops_int16 test/test_c_ops_int16.c -Isrc/c_ops -lm && ./test_c_ops_int16
```

## Requirements

- Python 3.8+, PyTorch >= 2.0.0, numpy, pytest
- `gcc` for comparison and verification tests (auto-skipped if missing)
