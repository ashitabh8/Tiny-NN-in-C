# Test Suite Documentation

This directory contains the comprehensive test suite for the PyTorch-to-C compiler.

## Test Files

### `test_models.py`
Defines test models used across the test suite:
- **TinyMLP**: Linear → ReLU → Linear (basic flow)
- **ResNetBlock**: Conv → BatchNorm → ReLU → Add (skip connections)
- **MixedNet**: Conv → ReLU → Linear → Softmax (mixed operations)

### `test_frontend.py`
Tests the torch.fx frontend:
- Model tracing validation
- Output preservation
- Graph printing
- Graph validation

### `test_lowering.py`
Tests FX graph to IR conversion:
- Node type mapping
- Double-linking verification
- Parameter extraction
- Layer name preservation
- Topological ordering

### `test_codegen.py`
Tests C code generation:
- weights.h generation
- model.h generation
- model.c generation
- Name sanitization
- Buffer allocation

### `test_integration.py`
End-to-end integration tests, including:

#### Basic Integration Tests
- Compilation of all test models
- File generation verification
- Parameter preservation

#### **PyTorch vs C Output Comparison** (NEW!)
The most important test for correctness validation:

**Class: `TestPyTorchCComparison`**

These tests validate the Phase 1 success criteria:
> "C output matches PyTorch output within floating-point tolerance"

**What it does:**
1. Compiles a PyTorch model to C
2. Runs PyTorch inference with test input
3. Compiles the generated C code with gcc
4. Runs the C executable with the same input
5. Compares outputs element-wise
6. Validates errors are within tolerance

**Tests:**
- `test_compare_tiny_mlp_outputs`: Detailed comparison for TinyMLP
- `test_compare_all_models`: Runs comparison for all test models

**Error Tolerances:**
- Max absolute error: < 1e-3
- Mean absolute error: < 1e-4

**Requirements:**
- gcc must be installed and in PATH
- Tests are automatically skipped if gcc is not available

**Run with:**
```bash
# Run just comparison tests
pytest test/test_integration.py::TestPyTorchCComparison -v -s

# Or use convenience script
python run_comparison_test.py
```

## Running Tests

### Run All Tests
```bash
pytest test/ -v
```

### Run Specific Test File
```bash
pytest test/test_frontend.py -v
pytest test/test_lowering.py -v
pytest test/test_codegen.py -v
pytest test/test_integration.py -v
```

### Run Specific Test Class
```bash
pytest test/test_integration.py::TestIntegration -v
pytest test/test_integration.py::TestPyTorchCComparison -v
```

### Run Specific Test
```bash
pytest test/test_integration.py::TestPyTorchCComparison::test_compare_tiny_mlp_outputs -v -s
```

### Show Print Statements
Add `-s` flag to see print output:
```bash
pytest test/test_integration.py::TestPyTorchCComparison -v -s
```

## Test Coverage

### Unit Tests (test_frontend.py, test_lowering.py, test_codegen.py)
- Test individual components in isolation
- Fast execution
- No external dependencies (except PyTorch)

### Integration Tests (test_integration.py)
- Test the complete pipeline
- End-to-end compilation
- File generation verification

### Comparison Tests (test_integration.py::TestPyTorchCComparison) ⭐
- **Most important for correctness validation**
- Tests actual execution of C code
- Requires gcc
- Compares numerical outputs
- Validates Phase 1 success criteria

## Test Requirements

### Required for All Tests
- Python 3.8+
- PyTorch >= 2.0.0
- pytest >= 7.3.0
- numpy >= 1.24.0

### Required for Comparison Tests Only
- gcc (C compiler)
- Linux/macOS (for subprocess execution)

## Adding New Tests

### Adding a New Test Model
Edit `test_models.py`:
```python
class MyNewModel(nn.Module):
    def __init__(self):
        super().__init__()
        # ... your model ...
    
    def forward(self, x):
        # ... forward pass ...
        return x

# Add to get_test_models()
def get_test_models():
    return [
        # ... existing models ...
        ("MyNewModel", MyNewModel(), torch.randn(1, 784)),
    ]
```

### Adding a Comparison Test
The comparison test framework automatically works with any model. Just add your model to the `test_cases` list in `test_compare_all_models()`:

```python
test_cases = [
    ("ModelName", model_instance, example_input, input_size, output_size),
]
```

## Understanding Test Output

### Comparison Test Output Example
```
TinyMLP Comparison:
  Max error: 2.38e-06
  Mean error: 4.12e-07
  PyTorch output (first 5): [ 0.234  -0.156   0.891  -0.423   0.067]
  C output (first 5):       [ 0.234  -0.156   0.891  -0.423   0.067]
  ✓ TinyMLP passed comparison test
```

### What the Errors Mean
- **Max error**: Largest absolute difference between any output element
- **Mean error**: Average absolute difference across all output elements
- Both should be very small (< 1e-3 for max, < 1e-4 for mean)

### Why Errors Occur
Small numerical errors are expected due to:
- Floating-point precision differences
- Order of operations
- Compiler optimizations
- Different math library implementations

Errors > 1e-3 indicate a problem and the test will fail.

## CI/CD Integration

For continuous integration, run:
```bash
# All tests except comparison (no gcc requirement)
pytest test/ -v -k "not TestPyTorchCComparison"

# Or with gcc available
pytest test/ -v
```

## Troubleshooting

### "gcc not available" - Tests Skipped

### Compilation Errors
Check that:
1. `src/c_ops/nn_ops_float.h` exists
2. Generated files are in the temp directory
3. Include paths are correct

### Large Errors in Comparison
If errors > 1e-3:
1. Check the generated C code for correctness
2. Verify weight loading is correct
3. Check for uninitialized buffers
4. Verify operation implementations match PyTorch

### Test Timeouts
Comparison tests have 30-second timeouts for compilation and execution. If tests timeout:
1. Check for infinite loops in C code
2. Verify model size is reasonable
3. Check system resources

## Future Improvements

Planned test enhancements:
- [ ] Add tests with different input shapes
- [ ] Add tests with edge cases (zeros, very large/small values)
- [ ] Add tests for quantized models (Phase 2)
- [ ] Add memory usage tests (Phase 3)
- [ ] Add performance benchmarks

