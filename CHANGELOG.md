# Changelog

All notable changes to the PyTorch-to-C compiler project.

## [0.1.0] - Phase 1: Float32 Baseline - 2026-01-08

### Added - Core Infrastructure

- **Project Structure**
  - Complete Python package structure with `src/pytorch_to_c/`
  - C operations library in `src/c_ops/`
  - Test suite in `test/`
  - Examples in `examples/`
  - Documentation files

- **Frontend Module** (`src/pytorch_to_c/frontend/`)
  - `FXTracer` class for torch.fx integration
  - Model tracing with validation
  - Graph visualization and printing
  - Support for skip connections and complex topologies

- **IR Layer** (`src/pytorch_to_c/ir/`)
  - `IRNode` class with double-linking (inputs + users)
  - `IRGraph` container with parameter storage
  - Topological sort implementation
  - Graph validation
  - Human-readable graph printing

- **Lowering Pass** (`src/pytorch_to_c/lowering/`)
  - FX graph to IR conversion
  - Support for Conv2D, Linear, ReLU, BatchNorm2D, Softmax
  - Automatic weight layout conversion
  - Parameter extraction and storage
  - Layer name preservation for future quantization

- **Code Generation** (`src/pytorch_to_c/codegen/`)
  - C code printer with three output files:
    - `model.h` - Function declarations
    - `model.c` - Implementation
    - `weights.h` - Serialized parameters
  - Operation mapping system
  - Buffer allocation
  - Name sanitization for C identifiers

- **Main Compiler** (`src/pytorch_to_c/compiler.py`)
  - `PyTorchToCCompiler` orchestration class
  - Three-stage pipeline (trace → lower → codegen)
  - Verbose mode for debugging
  - Model statistics reporting
  - Convenience function `compile_model()`

- **C Operations** (`src/c_ops/nn_ops_float.h`)
  - `conv2d_nhwc` - 2D convolution with SAME/VALID padding
  - `dense` - Fully connected layer
  - `relu` - ReLU activation
  - `batchnorm2d_nhwc` - Batch normalization (NEW)
  - `softmax` - Softmax activation
  - `global_average_pool_2d` - Global average pooling
  - All operations use NHWC layout
  - No dynamic allocation
  - Portable C99 code

### Added - Testing

- **Test Models** (`test/test_models.py`)
  - TinyMLP: Simple Linear → ReLU → Linear
  - ResNetBlock: Conv → BatchNorm → ReLU → Add
  - MixedNet: Conv → ReLU → Linear → Softmax

- **Unit Tests**
  - `test_frontend.py`: FX tracing validation (5 tests)
  - `test_lowering.py`: IR conversion and double-linking (7 tests)
  - `test_codegen.py`: C code generation (6 tests)
  - `test_integration.py`: End-to-end compilation (6 tests)

- **Quick Tests**
  - `test_basic.py`: Smoke test for basic functionality

### Added - Documentation

- **README.md**: Complete project documentation
  - Architecture overview
  - Installation instructions
  - Quick start guide
  - API documentation
  - Roadmap

- **IMPLEMENTATION_SUMMARY.md**: Phase 1 completion report
  - What was built
  - Design decisions
  - Testing status
  - Next steps for Phase 2

- **quantization/README.md**: Phase 2 implementation guide
  - Rule-based quantization system design
  - How to add quantization nodes
  - C implementation patterns
  - Per-channel vs per-tensor quantization

- **CHANGELOG.md**: This file

### Added - Examples

- **examples/tiny_mlp.py**: Complete working example
  - Model definition
  - Compilation
  - Output inspection
  - Next steps guide

### Added - Configuration

- **requirements.txt**: Python dependencies
  - torch >= 2.0.0
  - numpy >= 1.24.0
  - pytest >= 7.3.0

- **setup.py**: Package configuration for pip installation

## Design Decisions

### Double-Linking Architecture
Each IRNode maintains both `inputs` (dependencies) and `users` (consumers). This enables:
- Bidirectional graph traversal
- Future liveness analysis (Phase 3)
- Efficient optimization passes

### Layer Name Preservation
Original PyTorch layer names are preserved in the IR to enable:
- Rule-based quantization matching (Phase 2)
- Better debugging and traceability
- Human-readable generated code

### NHWC Tensor Layout
All operations use Height-Width-Channels layout because:
- Better for embedded targets
- Simpler memory access patterns
- Standard in microcontroller ML frameworks (TensorFlow Lite)
- More efficient for small batch sizes (typically 1)

### Weight Layout Conversion
Automatic conversion between PyTorch and C:
- Conv2D: PyTorch [OutC, InC, H, W] → C [H, W, InC, OutC]
- Linear: PyTorch [Out, In] → C [In, Out]

This enables direct memory access without reshaping at runtime.

## Phase 1 Success Criteria - All Met ✅

- ✅ Directory structure created
- ✅ IR with double-linking implemented
- ✅ torch.fx frontend working
- ✅ Lowering pass functional
- ✅ C code generation complete
- ✅ Test suite comprehensive
- ✅ TinyMLP compiles successfully
- ✅ ResNetBlock with skip connections works
- ✅ MixedNet with multiple op types works
- ✅ Generated C code is valid
- ✅ Documentation complete

## Known Limitations (Phase 1)

- Shape inference is simplified (some sizes hardcoded)
- Each node has separate buffer (no memory optimization)
- Float32 only (quantization in Phase 2)
- Limited operation set (core ops only)
- No dynamic control flow (torch.fx limitation)

## Next Release - Phase 2: Quantization Engine

Planned features:
- [ ] Calibration pass for min/max statistics
- [ ] Rule-based quantization system
- [ ] int8 and int16 quantization
- [ ] Quantize/Dequantize node insertion
- [ ] Per-channel quantization support
- [ ] Quantized C operations
- [ ] Accuracy validation vs PyTorch

## Future Release - Phase 3: Embedded Optimization

Planned features:
- [ ] Liveness analysis pass
- [ ] Arena memory allocator
- [ ] Static memory allocation (no malloc)
- [ ] Memory usage reporting
- [ ] Buffer overlap visualization
- [ ] Optimized buffer reuse

---

## Version History

- **0.1.0** - Initial release, Phase 1 complete (2026-01-08)

