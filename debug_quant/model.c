// Auto-generated model implementation
// DO NOT EDIT

#include "model.h"
#include "weights.h"
#include "nn_ops_float.h"
#include "nn_ops_int8.h"

#include <string.h>

void model_forward(const float* input, float* output) {
    // Intermediate buffers
    int8_t buf_x_to_fc1_q[16];
    int8_t buf_fc1[8];
    float buf_fc1_to_relu_dq[8];
    float buf_relu[8];
    int8_t buf_relu_to_fc2_q[8];
    int8_t buf_fc2[4];
    float buf_fc2_output_dq[4];

    // Forward pass

    // x_to_fc1_q [quantize]
    quantize_float_to_int8(input, 16, 0.01f, 0, buf_x_to_fc1_q);

    // fc1 [linear]
    dense_int8(buf_x_to_fc1_q, 16, fc1_weight, fc1_bias, 8, 0.01f, 0, buf_fc1);

    // fc1_to_relu_dq [dequantize]
    dequantize_int8_to_float(buf_fc1, 8, 0.01f, 0, buf_fc1_to_relu_dq);

    // relu [relu]
    memcpy(buf_relu, buf_fc1_to_relu_dq, 8 * sizeof(float));
    relu(buf_relu, 8);

    // relu_to_fc2_q [quantize]
    quantize_float_to_int8(buf_relu, 8, 0.01f, 0, buf_relu_to_fc2_q);

    // fc2 [linear]
    dense_int8(buf_relu_to_fc2_q, 8, fc2_weight, fc2_bias, 4, 0.01f, 0, buf_fc2);

    // fc2_output_dq [dequantize]
    dequantize_int8_to_float(buf_fc2, 4, 0.01f, 0, buf_fc2_output_dq);

    // Copy output
    memcpy(output, buf_fc2_output_dq, 4 * sizeof(float));
}
