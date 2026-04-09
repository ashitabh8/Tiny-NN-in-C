// Auto-generated model implementation
// DO NOT EDIT

// Input shape  (NCHW): [1, 6, 7, 256]
// Output shape: [1, 5]

#include <Arduino.h>

#include "model.h"
#include "weights.h"
#include "nn_ops_float.h"

#include <string.h>

void model_forward(const float* input, float* output) {
    unsigned long _t_start, _t_end;
    _t_start = micros();
    static float slot_0[10752];
    static float slot_1[10752];
    static float slot_2[10752];

    // stem_conv [conv2d]
    conv2d_nhwc(input, 7, 256, 6, stem_conv_weight, 5, 5, 2, NULL, 1, 1, 1, slot_0);
    // stem_bn [batchnorm]
    batchnorm2d_nhwc(slot_0, 7, 256, 2, stem_bn_gamma, stem_bn_beta, stem_bn_mean, stem_bn_var, 1e-05f, slot_1);
    // stem_relu [relu]
    relu(slot_1, 3584);
    // stage0_block0_conv1 [conv2d]
    conv2d_nhwc(slot_1, 7, 256, 2, stage0_block0_conv1_weight, 3, 3, 2, NULL, 1, 1, 1, slot_0);
    // stage0_block0_bn1 [batchnorm]
    batchnorm2d_nhwc(slot_0, 7, 256, 2, stage0_block0_bn1_gamma, stage0_block0_bn1_beta, stage0_block0_bn1_mean, stage0_block0_bn1_var, 1e-05f, slot_2);
    // stage0_block0_relu1 [relu]
    relu(slot_2, 3584);
    // stage0_block0_conv2 [conv2d]
    conv2d_nhwc(slot_2, 7, 256, 2, stage0_block0_conv2_weight, 3, 3, 2, NULL, 1, 1, 1, slot_0);
    // stage0_block0_bn2 [batchnorm]
    batchnorm2d_nhwc(slot_0, 7, 256, 2, stage0_block0_bn2_gamma, stage0_block0_bn2_beta, stage0_block0_bn2_mean, stage0_block0_bn2_var, 1e-05f, slot_2);
    // add [add]
    for (int i = 0; i < 3584; ++i) {
        slot_0[i] = slot_2[i] + slot_1[i];
    }
    // stage0_block0_relu2 [relu]
    relu(slot_0, 3584);
    // stage0_block1_conv1 [conv2d]
    conv2d_nhwc(slot_0, 7, 256, 2, stage0_block1_conv1_weight, 3, 3, 2, NULL, 1, 1, 1, slot_1);
    // stage0_block1_bn1 [batchnorm]
    batchnorm2d_nhwc(slot_1, 7, 256, 2, stage0_block1_bn1_gamma, stage0_block1_bn1_beta, stage0_block1_bn1_mean, stage0_block1_bn1_var, 1e-05f, slot_2);
    // stage0_block1_relu1 [relu]
    relu(slot_2, 3584);
    // stage0_block1_conv2 [conv2d]
    conv2d_nhwc(slot_2, 7, 256, 2, stage0_block1_conv2_weight, 3, 3, 2, NULL, 1, 1, 1, slot_1);
    // stage0_block1_bn2 [batchnorm]
    batchnorm2d_nhwc(slot_1, 7, 256, 2, stage0_block1_bn2_gamma, stage0_block1_bn2_beta, stage0_block1_bn2_mean, stage0_block1_bn2_var, 1e-05f, slot_2);
    // add_1 [add]
    for (int i = 0; i < 3584; ++i) {
        slot_1[i] = slot_2[i] + slot_0[i];
    }
    // stage0_block1_relu2 [relu]
    relu(slot_1, 3584);
    // stage1_block0_shortcut_conv [conv2d]
    conv2d_nhwc(slot_1, 7, 256, 2, stage1_block0_shortcut_conv_weight, 1, 1, 6, NULL, 1, 1, 0, slot_0);
    // stage1_block0_shortcut_bn [batchnorm]
    batchnorm2d_nhwc(slot_0, 7, 256, 6, stage1_block0_shortcut_bn_gamma, stage1_block0_shortcut_bn_beta, stage1_block0_shortcut_bn_mean, stage1_block0_shortcut_bn_var, 1e-05f, slot_2);
    // stage1_block0_conv1 [conv2d]
    conv2d_nhwc(slot_1, 7, 256, 2, stage1_block0_conv1_weight, 3, 3, 6, NULL, 1, 1, 1, slot_0);
    // stage1_block0_bn1 [batchnorm]
    batchnorm2d_nhwc(slot_0, 7, 256, 6, stage1_block0_bn1_gamma, stage1_block0_bn1_beta, stage1_block0_bn1_mean, stage1_block0_bn1_var, 1e-05f, slot_1);
    // stage1_block0_relu1 [relu]
    relu(slot_1, 10752);
    // stage1_block0_conv2 [conv2d]
    conv2d_nhwc(slot_1, 7, 256, 6, stage1_block0_conv2_weight, 3, 3, 6, NULL, 1, 1, 1, slot_0);
    // stage1_block0_bn2 [batchnorm]
    batchnorm2d_nhwc(slot_0, 7, 256, 6, stage1_block0_bn2_gamma, stage1_block0_bn2_beta, stage1_block0_bn2_mean, stage1_block0_bn2_var, 1e-05f, slot_1);
    // add_2 [add]
    for (int i = 0; i < 10752; ++i) {
        slot_0[i] = slot_1[i] + slot_2[i];
    }
    // stage1_block0_relu2 [relu]
    relu(slot_0, 10752);
    // stage1_block1_conv1 [conv2d]
    conv2d_nhwc(slot_0, 7, 256, 6, stage1_block1_conv1_weight, 3, 3, 6, NULL, 1, 1, 1, slot_1);
    // stage1_block1_bn1 [batchnorm]
    batchnorm2d_nhwc(slot_1, 7, 256, 6, stage1_block1_bn1_gamma, stage1_block1_bn1_beta, stage1_block1_bn1_mean, stage1_block1_bn1_var, 1e-05f, slot_2);
    // stage1_block1_relu1 [relu]
    relu(slot_2, 10752);
    // stage1_block1_conv2 [conv2d]
    conv2d_nhwc(slot_2, 7, 256, 6, stage1_block1_conv2_weight, 3, 3, 6, NULL, 1, 1, 1, slot_1);
    // stage1_block1_bn2 [batchnorm]
    batchnorm2d_nhwc(slot_1, 7, 256, 6, stage1_block1_bn2_gamma, stage1_block1_bn2_beta, stage1_block1_bn2_mean, stage1_block1_bn2_var, 1e-05f, slot_2);
    // add_3 [add]
    for (int i = 0; i < 10752; ++i) {
        slot_1[i] = slot_2[i] + slot_0[i];
    }
    // stage1_block1_relu2 [relu]
    relu(slot_1, 10752);
    // stage1_block2_conv1 [conv2d]
    conv2d_nhwc(slot_1, 7, 256, 6, stage1_block2_conv1_weight, 3, 3, 6, NULL, 1, 1, 1, slot_0);
    // stage1_block2_bn1 [batchnorm]
    batchnorm2d_nhwc(slot_0, 7, 256, 6, stage1_block2_bn1_gamma, stage1_block2_bn1_beta, stage1_block2_bn1_mean, stage1_block2_bn1_var, 1e-05f, slot_2);
    // stage1_block2_relu1 [relu]
    relu(slot_2, 10752);
    // stage1_block2_conv2 [conv2d]
    conv2d_nhwc(slot_2, 7, 256, 6, stage1_block2_conv2_weight, 3, 3, 6, NULL, 1, 1, 1, slot_0);
    // stage1_block2_bn2 [batchnorm]
    batchnorm2d_nhwc(slot_0, 7, 256, 6, stage1_block2_bn2_gamma, stage1_block2_bn2_beta, stage1_block2_bn2_mean, stage1_block2_bn2_var, 1e-05f, slot_2);
    // add_4 [add]
    for (int i = 0; i < 10752; ++i) {
        slot_0[i] = slot_2[i] + slot_1[i];
    }
    // stage1_block2_relu2 [relu]
    relu(slot_0, 10752);
    // exit_conv [conv2d]
    conv2d_nhwc(slot_0, 7, 256, 6, exit_conv_weight, 5, 5, 6, NULL, 2, 2, 1, slot_1);
    // exit_bn [batchnorm]
    batchnorm2d_nhwc(slot_1, 4, 128, 6, exit_bn_gamma, exit_bn_beta, exit_bn_mean, exit_bn_var, 1e-05f, slot_0);
    // exit_relu [relu]
    relu(slot_0, 3072);
    // exit_pool [adaptive_avg_pool]
    global_average_pool_2d(slot_0, 4, 128, 6, slot_1);
    // view [method_view]
    memcpy(slot_0, slot_1, 6 * sizeof(float));
    // exit_fc [linear]
    dense(slot_0, 6, exit_fc_weight, exit_fc_bias, 5, slot_1);
    // exit_softmax [softmax]
    memcpy(slot_0, slot_1, 5 * sizeof(float));
    softmax(slot_0, 5);
    _t_end = micros();
    Serial.print("Final Exit: ");
    Serial.print((_t_end - _t_start) / 1000.0f, 2);
    Serial.println(" ms");
    memcpy(output, slot_0, 5 * sizeof(float));
}
