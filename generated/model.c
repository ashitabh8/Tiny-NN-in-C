// Auto-generated model implementation
// DO NOT EDIT

#include "model.h"
#include "weights.h"
#include "nn_ops_float.h"

#include <string.h>

void model_forward(const float* input, float* output) {
    // Intermediate buffers
    float buf_fc1[128];
    float buf_relu[128];
    float buf_fc2[10];

    // Forward pass

    // fc1 [linear]
    dense(input, 784, fc1_weight, fc1_bias, 128, buf_fc1);

    // relu [relu]
    memcpy(buf_relu, buf_fc1, 128 * sizeof(float));
    relu(buf_relu, 128);

    // fc2 [linear]
    dense(buf_relu, 128, fc2_weight, fc2_bias, 10, buf_fc2);

    // Copy output
    memcpy(output, buf_fc2, 10 * sizeof(float));
}
