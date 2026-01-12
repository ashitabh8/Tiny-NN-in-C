# PyTorch to C Compiler

A compiler that converts PyTorch models into standalone, dependency-free C code for microcontrollers.

## Quick Start

### 1. Setup

```bash
pip install -r requirements.txt
```

### 2. Run the TinyResNet Example

```bash
python examples/tiny_resnet.py
```

This generates C code in `tmp/tiny_resnet_for_embedded_device/`.

### 3. Generated Files

```
tmp/tiny_resnet_for_embedded_device/
├── model.h         # Function declarations
├── model.c         # Model implementation
├── weights.h       # Quantized weights (int8)
└── nn_ops_*.h      # C operation kernels
```

### 4. Using the Generated C Code

**Include in your project:**
```c
#include "model.h"
```

**Prepare input data:**
```c
// Input: (1, 1, 200, 10) in NHWC layout = 2000 floats
// Note: PyTorch uses NCHW (1, 10, 1, 200), C uses NHWC (1, 1, 200, 10)
float input_data[2000];
// Fill with your sensor data...
```

**Run inference:**
```c
float output[10];  // 10 classes
model_forward(input_data, output);
```

**Get prediction:**
```c
int predicted_class = argmax(output, 10);
```

### 5. Compile for Your Target

```bash
# Host testing:
gcc -O2 -o model_test main.c model.c -lm

# ARM Cortex-M:
arm-none-eabi-gcc -mcpu=cortex-m4 -O2 -c model.c -o model.o
```

## Model Details

- **Input shape:** `(1, 10, 1, 200)` NCHW → `(1, 1, 200, 10)` NHWC in C
- **Output shape:** `(10,)` - 10-class scores
- **Model size:** ~100KB (int8 quantized)
- **Architecture:** ResNet-style with skip connections

## Testing

```bash
# Run all tests
pytest test/

# Run quantization tests
pytest test/test_quantization_e2e.py -v
```

## License

MIT
