# PyTorch to C Compiler

A modular compiler that converts PyTorch models (nn.Module) into standalone, dependency-free C code suitable for microcontrollers (Cortex-M, ESP32, etc.).

## Project Status

**Phase 1: Float32 Baseline** - ✅ **COMPLETED**
- Full compilation pipeline working
- Shape inference implemented
- PyTorch vs C output comparison tests passing
- Self-contained C output

**Phase 2: Quantization Engine** - 📝 **IN DESIGN**
- Comprehensive design documents completed
- See [DESIGN_DOCS_INDEX.md](DESIGN_DOCS_INDEX.md) for all design docs
- Ready for implementation pending final design decisions
- Original plan: [quantization/README.md](quantization/README.md) (to be updated)

**Phase 3: Embedded Optimization** - 📝 Planned

## Features

- **Graph-First Architecture**: Operates on an optimized intermediate representation (IR)
- **Rule-Based Quantization** (Phase 2): Decouple model architecture from quantization logic
- **Embedded-Native**: Optimized for static memory allocation (Phase 3)
- **Supported Operations**: Conv2D, Linear, ReLU, BatchNorm2D, Softmax, Add

## Architecture

The system follows a standard Frontend → Middle-end → Backend compiler flow:

```
PyTorch Model → torch.fx Graph → Custom IR → C Code
```

### Components

1. **Frontend** (`src/pytorch_to_c/frontend/`): Traces PyTorch models using torch.fx
2. **IR** (`src/pytorch_to_c/ir/`): Double-linked intermediate representation
3. **Lowering** (`src/pytorch_to_c/lowering/`): Converts FX graph to custom IR
4. **Codegen** (`src/pytorch_to_c/codegen/`): Generates C code (model.c, model.h, weights.h)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Phase 2 Design Documentation

Comprehensive design documents for the **Quantization Engine** (Phase 2) are now available:

- **[DESIGN_DOCS_INDEX.md](DESIGN_DOCS_INDEX.md)** - Start here! Navigation guide to all design docs
- **[QUANTIZATION_DESIGN.md](QUANTIZATION_DESIGN.md)** - Complete technical specification
- **[QUANTIZATION_ARCHITECTURE.md](QUANTIZATION_ARCHITECTURE.md)** - Visual architecture guide
- **[QUANTIZATION_DESIGN_SUMMARY.md](QUANTIZATION_DESIGN_SUMMARY.md)** - Executive summary
- **[PHASE1_VS_PHASE2.md](PHASE1_VS_PHASE2.md)** - What changes from Phase 1 to Phase 2
- **[DESIGN_DECISIONS_NEEDED.md](DESIGN_DECISIONS_NEEDED.md)** - Open questions to resolve

**Key Design Principles:**
- ✅ **Modular**: Logic in rules/nodes, not in c_printer
- ✅ **Extensible**: Easy to add new quantization strategies
- ✅ **User Control**: Users provide scale/offset (no calibration in compiler)
- ✅ **Backward Compatible**: Phase 1 code continues to work

## Quick Start

```python
import torch
import torch.nn as nn
from src.pytorch_to_c.compiler import compile_model

# Define your model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create model and example input
model = MyModel()
example_input = torch.randn(1, 784)

# Compile to C
compile_model(
    model=model,
    example_input=example_input,
    output_dir="generated"
)
```

This generates:
- `generated/model.h` - Function declarations
- `generated/model.c` - Model implementation
- `generated/weights.h` - Serialized weights
- `generated/nn_ops_float.h` - C operations (copied for self-contained deployment)

## Example

Run the included example:

```bash
python examples/tiny_mlp.py
```

## Testing

Run the test suite:

```bash
pytest test/
```

Run PyTorch vs C output comparison tests (requires gcc):

```bash
python run_comparison_test.py
# or
pytest test/test_integration.py::TestPyTorchCComparison -v -s
```

Test coverage includes:
- ✅ Frontend: torch.fx tracing
- ✅ Lowering: FX to IR conversion with double-linking
- ✅ Codegen: C code generation
- ✅ Integration: End-to-end compilation
- ✅ **Output Comparison: PyTorch vs C model outputs (validates correctness)**

### Test Models

The test suite includes three reference models:

1. **TinyMLP**: Linear → ReLU → Linear (basic flow)
2. **ResNetBlock**: Conv → BatchNorm → ReLU → Add (skip connections)
3. **MixedNet**: Conv → ReLU → Linear → Softmax (mixed operations)

## Project Structure

```
Tiny-NN-in-C/
├── src/
│   ├── pytorch_to_c/          # Main compiler package
│   │   ├── frontend/          # torch.fx tracing
│   │   ├── ir/                # Intermediate representation
│   │   ├── lowering/          # FX to IR conversion
│   │   ├── codegen/           # C code generation
│   │   └── compiler.py        # Main compiler entry point
│   └── c_ops/                 # C operation implementations
│       └── nn_ops_float.h     # Float32 operations
├── quantization/              # Quantization strategies (Phase 2)
│   ├── README.md              # Quantization guide
│   └── strategies/            # Future implementations
├── test/                      # Test suite
│   ├── test_models.py         # Test model definitions
│   ├── test_frontend.py       # Frontend tests
│   ├── test_lowering.py       # Lowering tests
│   ├── test_codegen.py        # Code generation tests
│   └── test_integration.py    # End-to-end tests
├── examples/                  # Usage examples
│   └── tiny_mlp.py
├── generated/                 # Generated C code output
├── requirements.txt
├── setup.py
└── README.md
```

## Design Decisions

1. **Layer Name Matching**: IR nodes preserve PyTorch layer names for future rule-based quantization
2. **Double-Linking**: IR nodes maintain both inputs and users lists for Phase 3 liveness analysis
3. **Memory Strategy**: Phase 1 uses simple per-node arrays; Phase 3 will optimize with arena allocator
4. **Tensor Layout**: NHWC format (matches TensorFlow Lite convention)
5. **C Code Structure**: Each node gets its own output buffer; function calls are sequential

## Supported Operations (Phase 1)

| Operation | PyTorch | C Function | Status |
|-----------|---------|------------|--------|
| Conv2D | `nn.Conv2d` | `conv2d_nhwc` | ✅ |
| Linear | `nn.Linear` | `dense` | ✅ |
| ReLU | `nn.ReLU` | `relu` | ✅ |
| BatchNorm2D | `nn.BatchNorm2d` | `batchnorm2d_nhwc` | ✅ |
| Softmax | `nn.Softmax` | `softmax` | ✅ |
| Add | `torch.add` | element-wise add | ✅ |

## Limitations (Phase 1)

- No dynamic control flow (if/while in forward())
- Fixed input shapes (determined at compile time)
- No automatic shape inference (some shapes hardcoded)
- No memory optimization (each node has separate buffer)
- Float32 only (int8/int16 quantization in Phase 2)

## Roadmap

### Phase 1: Float32 Baseline ✅
- [x] IR design with double-linking
- [x] torch.fx frontend
- [x] Lowering pass
- [x] C code generation
- [x] Test suite
- [x] Example usage

### Phase 2: Quantization Engine 📝
- [ ] Calibration pass
- [ ] Rule-based quantization system
- [ ] int8/int16 quantization
- [ ] Quantize/Dequantize node insertion
- [ ] Per-channel quantization support

### Phase 3: Embedded Optimization 📝
- [ ] Liveness analysis
- [ ] Arena memory allocator
- [ ] Static memory allocation (no malloc)
- [ ] Memory usage reporting
- [ ] Buffer overlap visualization

## Contributing

This is a research/educational project. Contributions are welcome!

## References

- Design document: [DesignDoc](DesignDoc)
- Quantization guide: [quantization/README.md](quantization/README.md)
- PyTorch FX: https://pytorch.org/docs/stable/fx.html
- TensorFlow Lite: https://www.tensorflow.org/lite

## License

[Your License Here]

## Authors

[Your Name]
