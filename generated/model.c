// Auto-generated model implementation
// DO NOT EDIT

#include "model.h"
#include "weights.h"
#include "nn_ops_float.h"

#include <string.h>

void model_forward(const float* input, float* output) {
    // Intermediate buffers
    float buf_conv[8192];
    float buf_relu[8192];
    float buf_size[1024];
    float buf_view[8192];
    float buf_fc[10];
    float buf_softmax[10];

    // Forward pass

    // conv [conv2d]
    int in_h = 32, in_w = 32;  // TODO: Infer from previous layer
    conv2d_nhwc(input, in_h, in_w, 3, conv_weight, 3, 3, 32, conv_bias, 2, 2, 1, buf_conv);

    // relu [relu]
    memcpy(buf_relu, buf_conv, 8192 * sizeof(float));
    relu(buf_relu, 8192);

    // size [method_size]
    // Unsupported operation: method_size

    // view [method_view]
    // Unsupported operation: method_view

    // fc [linear]
    dense(buf_view, 8192, fc_weight, fc_bias, 10, buf_fc);

    // softmax [softmax]
    memcpy(buf_softmax, buf_fc, 10 * sizeof(float));
    softmax(buf_softmax, 10);

    // Copy output
    memcpy(output, buf_softmax, 10 * sizeof(float));
}
