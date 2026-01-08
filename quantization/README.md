# Quantization Strategies Guide

**Status:** Phase 2 - To Be Implemented

This document describes the rule-based quantization system and how to add quantization nodes and strategies to the PyTorch-to-C compiler.

## Overview

The compiler uses a **rule-based quantization system** that decouples model architecture from quantization logic. This approach is inspired by QWIX/Jax and allows flexible experimentation without modifying the model code.

## Key Principles

1. **Rules Match Layer Names**: Quantization rules match against the layer's `name` parameter (as assigned in PyTorch), NOT the class name. This allows fine-grained control.
   
2. **Supported Data Types**: 
   - `float32` - Default floating point
   - `int8` - 8-bit integer quantization
   - `int16` - 16-bit integer quantization

3. **Quantization Schemes**:
   - **Static Quantization**: Weights and activations quantized at compile time
   - **Dynamic Quantization**: Only weights quantized; activations computed in float then quantized per-operation

## Rule Structure

A quantization rule consists of:

```python
class QuantRule:
    pattern: str           # Regex pattern to match layer names
    weight_dtype: str      # Data type for weights ('int8', 'int16', 'float32')
    activation_dtype: str  # Data type for activations
    scheme: str            # 'static' or 'dynamic'
    per_channel: bool      # Per-channel vs per-tensor quantization
```

### Example Rules

```python
rules = [
    # Quantize all layers with 'linear' in their name to int8
    QuantRule(
        pattern=r'.*linear.*',
        weight_dtype='int8',
        activation_dtype='int8',
        scheme='static',
        per_channel=True
    ),
    
    # Keep skip connections in float32
    QuantRule(
        pattern=r'.*skip.*',
        weight_dtype='float32',
        activation_dtype='float32',
        scheme='static',
        per_channel=False
    ),
    
    # Quantize convolutional layers with int16
    QuantRule(
        pattern=r'conv\d+',
        weight_dtype='int16',
        activation_dtype='int16',
        scheme='static',
        per_channel=True
    ),
]
```

## How Quantization Works

### Phase 2.1: Calibration Pass

1. Run representative data through the float32 IR graph
2. Observe and record `[min, max]` values for each activation
3. Store statistics in `IRNode.metadata['calibration_stats']`

### Phase 2.2: Rule Application

1. Iterate through all nodes in the IR graph
2. For each node, check if its name matches any rule pattern
3. If matched:
   - Set `node.dtype` to the rule's data type
   - Calculate quantization parameters:
     - `scale = (max - min) / (2^bits - 1)`
     - `zero_point = -round(min / scale)`
   - Store in `node.metadata['quant_params']`

### Phase 2.3: Graph Transformation

Insert `Quantize` and `Dequantize` nodes at precision boundaries:

```
[float32] -> [int8] : Insert Quantize node
[int8] -> [float32] : Insert Dequantize node
```

Example transformation:
```
Before:
  conv1 (float32) -> relu1 (float32) -> linear1 (float32)

After (with rule matching '.*linear.*' to int8):
  conv1 (float32) -> relu1 (float32) -> quantize -> linear1 (int8) -> dequantize
```

## Adding Quantize/Dequantize Nodes

### Quantize Node

```python
IRNode(
    name=f"{node.name}_quantize",
    op_type="quantize",
    dtype=target_dtype,
    metadata={
        'scale': scale,
        'zero_point': zero_point,
        'target_dtype': target_dtype
    }
)
```

### Dequantize Node

```python
IRNode(
    name=f"{node.name}_dequantize",
    op_type="dequantize",
    dtype="float32",
    metadata={
        'scale': scale,
        'zero_point': zero_point,
        'source_dtype': source_dtype
    }
)
```

## C Implementation

### Quantization Operations

For int8:
```c
int8_t quantize_float_to_int8(float x, float scale, int zero_point) {
    int32_t val = (int32_t)roundf(x / scale) + zero_point;
    return (int8_t)clamp_int(val, -128, 127);
}

float dequantize_int8_to_float(int8_t x, float scale, int zero_point) {
    return scale * (float)(x - zero_point);
}
```

For int16:
```c
int16_t quantize_float_to_int16(float x, float scale, int zero_point) {
    int32_t val = (int32_t)roundf(x / scale) + zero_point;
    return (int16_t)clamp_int(val, -32768, 32767);
}

float dequantize_int16_to_float(int16_t x, float scale, int zero_point) {
    return scale * (float)(x - zero_point);
}
```

### Quantized Operations

Quantized operations perform integer arithmetic:

```c
// Quantized Linear: y_q = (W_q * x_q + b_q)
// Requires careful handling of scales and zero points
void dense_int8(
    const int8_t* x, int in_features,
    const int8_t* W, const int32_t* b,
    int out_features,
    float x_scale, int x_zero,
    float w_scale, int w_zero,
    float y_scale, int y_zero,
    int8_t* y)
{
    for (int o = 0; o < out_features; ++o) {
        int32_t acc = b ? b[o] : 0;
        for (int i = 0; i < in_features; ++i) {
            acc += (int32_t)(x[i] - x_zero) * (int32_t)(W[i * out_features + o] - w_zero);
        }
        // Rescale: acc * (x_scale * w_scale) / y_scale
        float result = (float)acc * x_scale * w_scale / y_scale + y_zero;
        y[o] = (int8_t)clamp_int((int)roundf(result), -128, 127);
    }
}
```

## Per-Channel vs Per-Tensor Quantization

### Per-Tensor
- Single scale and zero_point for entire tensor
- Simpler, faster
- Less accurate

### Per-Channel
- Different scale and zero_point for each output channel
- More complex
- Better accuracy, especially for Conv2D weights

Example for per-channel Conv2D:
```python
# weights shape: [K_h, K_w, C_in, C_out]
# C_out scales and zero_points
for c_out in range(C_out):
    w_channel = weights[:, :, :, c_out]
    scale[c_out] = (w_channel.max() - w_channel.min()) / 255
    zero_point[c_out] = -round(w_channel.min() / scale[c_out])
```

## Integration with Lowering Pass

The lowering pass should:

1. Preserve layer names from PyTorch FX graph:
   ```python
   ir_node.name = fx_node.name  # This is the layer's name parameter
   ```

2. Store original layer type for rule matching:
   ```python
   ir_node.metadata['original_type'] = fx_node.target
   ```

3. Make layer names available for quantization rules

## Directory Structure

```
quantization/
├── README.md              # This file
├── strategies/            # Quantization strategy implementations
│   ├── static_int8.py     # Static int8 quantization
│   ├── static_int16.py    # Static int16 quantization
│   ├── dynamic_int8.py    # Dynamic int8 quantization
│   └── calibrator.py      # Calibration pass
├── rules.py               # Rule definition and matching
└── transforms.py          # Graph transformation passes
```

## Future Work (Phase 2)

- [ ] Implement calibration pass
- [ ] Implement rule matcher
- [ ] Add Quantize/Dequantize node insertion
- [ ] Implement int8/int16 C operations
- [ ] Add per-channel quantization support
- [ ] Validate accuracy vs PyTorch quantized models

## References

- PyTorch Quantization: https://pytorch.org/docs/stable/quantization.html
- TensorFlow Lite Quantization: https://www.tensorflow.org/lite/performance/quantization_spec
- QWIX Paper: Rule-based quantization for JAX

