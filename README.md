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

**Prepare input data and run inference:**
```c
float input_data[2000];   // 1 * 1 * 200 * 10 = 2000 floats
float output[10];         // 10 classes

// Fill input_data (see "Input Layout" section below)
model_forward(input_data, output);

// Find predicted class
int predicted_class = 0;
for (int i = 1; i < 10; i++) {
    if (output[i] > output[predicted_class]) predicted_class = i;
}
```

### 5. Compile for Your Target

```bash
# Host testing:
gcc -O2 -o model_test main.c model.c -lm

# ARM Cortex-M:
arm-none-eabi-gcc -mcpu=cortex-m4 -O2 -c model.c -o model.o
```

---

## Input Layout (NHWC)

The generated C code uses **NHWC layout** (channels-last), while PyTorch uses **NCHW** (channels-first).

### TinyResNet Example

| | PyTorch (NCHW) | C Code (NHWC) |
|---|----------------|---------------|
| **Shape** | `(1, 10, 1, 200)` | `(1, 1, 200, 10)` |
| **Meaning** | batch, freq_bins, height, time_steps | batch, height, time_steps, freq_bins |
| **Total floats** | 2000 | 2000 |

### How to Fill the Input Array

For the TinyResNet model with **10 frequency bins** and **200 time steps** (spectrogram input):

```
NHWC layout: input[time_step * num_freq_bins + freq_bin]

  input[0]   = freq bin 0 at time 0
  input[1]   = freq bin 1 at time 0
  ...
  input[9]   = freq bin 9 at time 0
  input[10]  = freq bin 0 at time 1
  input[11]  = freq bin 1 at time 1
  ...
  input[1999] = freq bin 9 at time 199
```

**C code example:**
```c
#define NUM_FREQ_BINS 10
#define NUM_TIMESTEPS 200

float input_data[NUM_TIMESTEPS * NUM_FREQ_BINS];

// Fill from spectrogram (magnitude values)
for (int t = 0; t < NUM_TIMESTEPS; t++) {
    for (int f = 0; f < NUM_FREQ_BINS; f++) {
        input_data[t * NUM_FREQ_BINS + f] = spectrogram[f][t];
    }
}
```

### Converting from PyTorch

If you have a PyTorch tensor in NCHW format, convert it to NHWC:

```python
# PyTorch tensor: shape (1, 10, 1, 200) in NCHW
pytorch_input = torch.randn(1, 10, 1, 200)

# Convert to NHWC: shape (1, 1, 200, 10)
nhwc_input = pytorch_input.permute(0, 2, 3, 1)

# Flatten to 1D array for C
c_input = nhwc_input.numpy().flatten()  # shape: (2000,)
```

---

## Model Details

- **Input:** Spectrogram with 10 frequency bins × 200 time steps
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
