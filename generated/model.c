// Auto-generated model implementation
// DO NOT EDIT

#include "model.h"
#include "weights.h"
#include "../src/c_ops/nn_ops_float.h"

#include <string.h>

void model_forward(const float* input, float* output) {
    // Intermediate buffers
    float buf_fc1[5];
    float buf_relu[5];
    float buf_fc2[2];

    // Forward pass

    // fc1 [linear]
    dense(input, 10, fc1_weight, fc1_bias, 5, buf_fc1);

    // relu [relu]
    memcpy(buf_relu, buf_fc1, 1024 * sizeof(float));
    relu(buf_relu, 1024);

    // fc2 [linear]
    dense(buf_relu, 5, fc2_weight, fc2_bias, 2, buf_fc2);

    // Copy output
    memcpy(output, buf_fc2, 2 * sizeof(float));
}
