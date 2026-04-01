// Auto-generated model header
// DO NOT EDIT

#ifndef MODEL_H_
#define MODEL_H_

#include <stddef.h>

// =============================================================================
// IMPORTANT: Input Layout
// =============================================================================
// This model expects input in NHWC format (batch, height, width, channels).
// PyTorch uses NCHW format. Convert before calling:
//   PyTorch: input.permute(0, 2, 3, 1).numpy().flatten()
// =============================================================================

// Input: x
// Output: final_softmax

// Main model inference function
void model_forward(const float* input, float* output);

#endif // MODEL_H_