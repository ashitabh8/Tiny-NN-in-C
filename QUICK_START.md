# Quick Start Guide

Get started with the PyTorch-to-C compiler in 5 minutes!

## Installation

```bash
cd /home/misra8/Tiny-NN-in-C

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Run Your First Compilation

### Option 1: Run the Example

```bash
python examples/tiny_mlp.py
```

This will:
1. Create a simple MLP model
2. Compile it to C
3. Generate files in `generated/` directory
4. Show you model statistics

### Option 2: Quick Smoke Test

```bash
python test_basic.py
```

This runs a minimal smoke test to verify everything works.

### Option 3: Run Full Test Suite

```bash
pytest test/ -v
```

This runs all unit tests and integration tests.

### Option 4: Run Output Comparison Test

This test compiles models to C, runs both PyTorch and C versions, and compares outputs:

```bash
python run_comparison_test.py
```

**Requires gcc to be installed.** This validates that the C model produces the same outputs as PyTorch within error tolerance (< 1e-3).

## Use in Your Own Code

Create a file `my_model.py`:

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

# Compile it!
model = MyModel()
example_input = torch.randn(1, 784)

compile_model(
    model=model,
    example_input=example_input,
    output_dir="my_output",
    verbose=True
)
```

Run it:
```bash
python my_model.py
```

## Check Generated Files

After compilation, you'll find four C files (self-contained, ready to deploy):

```
generated/
├── model.h          # Function declarations
├── model.c          # Model implementation  
├── weights.h        # Serialized weights
└── nn_ops_float.h   # C operations (automatically copied)
```

All necessary headers are included - just copy the entire `generated/` folder to your embedded project!

### Using the Generated Code

In your C project:

```c
#include "model.h"

int main() {
    float input[784] = { /* your input data */ };
    float output[10];
    
    // Run inference
    model_forward(input, output);
    
    // Use output...
    return 0;
}
```

Compile with:
```bash
gcc -o mymodel model.c main.c -lm
```

## Supported Operations

Currently supported (Phase 1):
- ✅ Linear (fully connected)
- ✅ Conv2D
- ✅ ReLU
- ✅ BatchNorm2D
- ✅ Softmax
- ✅ Element-wise Add

## Project Structure

```
.
├── src/pytorch_to_c/      # Main compiler package
│   ├── frontend/          # torch.fx tracing
│   ├── ir/                # Intermediate representation
│   ├── lowering/          # FX to IR conversion
│   ├── codegen/           # C code generation
│   └── compiler.py        # Main entry point
├── src/c_ops/             # C operation implementations
├── test/                  # Test suite
├── examples/              # Usage examples
└── quantization/          # Phase 2 docs (future)
```

## Next Steps

1. **Read the README**: Full documentation in `README.md`
2. **Check the Example**: See `examples/tiny_mlp.py` for detailed usage
3. **Explore Tests**: Look at `test/test_models.py` for example models
4. **Phase 2 Preview**: See `quantization/README.md` for upcoming features

## Troubleshooting

### Import Errors

If you get import errors, make sure you've installed the package:
```bash
pip install -e .
```

Or add the src directory to your Python path:
```python
import sys
sys.path.insert(0, '/home/misra8/Tiny-NN-in-C')
```

### PyTorch Version

Requires PyTorch >= 2.0.0 for torch.fx support:
```bash
pip install torch>=2.0.0
```

### Missing Dependencies

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Getting Help

- **Design Document**: Read `DesignDoc` for architecture details
- **Implementation Summary**: See `IMPLEMENTATION_SUMMARY.md` for what's implemented
- **Changelog**: Check `CHANGELOG.md` for version history
- **Quantization Guide**: See `quantization/README.md` for Phase 2 plans

## Common Use Cases

### Case 1: Simple MLP
```python
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```

### Case 2: CNN
```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32 * 32 * 32, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

### Case 3: ResNet-style with Skip Connection
```python
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        identity = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = out + identity  # Skip connection
        return out
```

## Status

✅ **Phase 1 Complete**: Float32 baseline working
📝 **Phase 2 Planned**: Quantization (int8/int16)
📝 **Phase 3 Planned**: Memory optimization

Enjoy compiling PyTorch to C! 🚀

