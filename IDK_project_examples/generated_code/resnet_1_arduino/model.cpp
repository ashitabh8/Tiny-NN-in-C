// Auto-generated model implementation
// DO NOT EDIT

// Input shape  (NCHW): [1, 6, 7, 256]
// Output shape: [1, 10]

#include <Arduino.h>

#include "model.h"
#include "weights.h"
#include "nn_ops_float.h"

#include <string.h>

void model_forward(const float* input, float* output) {
    unsigned long _t_start, _t_end;
    _t_start = micros();
    float slot_0[8192];
    float slot_1[8192];
    float slot_2[8192];

    // stem_conv [conv2d]
    conv2d_nhwc(input, 7, 256, 6, stem_conv_weight, 5, 5, 16, NULL, 2, 2, 1, slot_0);
    // stem_bn [batchnorm]
    batchnorm2d_nhwc(slot_0, 4, 128, 16, stem_bn_gamma, stem_bn_beta, stem_bn_mean, stem_bn_var, 1e-05f, slot_1);
    // stem_relu [relu]
    relu(slot_1, 8192);
    // stage0_block0_conv1 [conv2d]
    conv2d_nhwc(slot_1, 4, 128, 16, stage0_block0_conv1_weight, 3, 3, 16, NULL, 1, 1, 1, slot_0);
    // stage0_block0_bn1 [batchnorm]
    batchnorm2d_nhwc(slot_0, 4, 128, 16, stage0_block0_bn1_gamma, stage0_block0_bn1_beta, stage0_block0_bn1_mean, stage0_block0_bn1_var, 1e-05f, slot_2);
    // stage0_block0_relu1 [relu]
    relu(slot_2, 8192);
    // stage0_block0_conv2 [conv2d]
    conv2d_nhwc(slot_2, 4, 128, 16, stage0_block0_conv2_weight, 3, 3, 16, NULL, 1, 1, 1, slot_0);
    // stage0_block0_bn2 [batchnorm]
    batchnorm2d_nhwc(slot_0, 4, 128, 16, stage0_block0_bn2_gamma, stage0_block0_bn2_beta, stage0_block0_bn2_mean, stage0_block0_bn2_var, 1e-05f, slot_2);
    // add [add]
    for (int i = 0; i < 8192; ++i) {
        slot_0[i] = slot_2[i] + slot_1[i];
    }
    // stage0_block0_relu2 [relu]
    relu(slot_0, 8192);
    // stage0_block1_conv1 [conv2d]
    conv2d_nhwc(slot_0, 4, 128, 16, stage0_block1_conv1_weight, 3, 3, 16, NULL, 1, 1, 1, slot_1);
    // stage0_block1_bn1 [batchnorm]
    batchnorm2d_nhwc(slot_1, 4, 128, 16, stage0_block1_bn1_gamma, stage0_block1_bn1_beta, stage0_block1_bn1_mean, stage0_block1_bn1_var, 1e-05f, slot_2);
    // stage0_block1_relu1 [relu]
    relu(slot_2, 8192);
    // stage0_block1_conv2 [conv2d]
    conv2d_nhwc(slot_2, 4, 128, 16, stage0_block1_conv2_weight, 3, 3, 16, NULL, 1, 1, 1, slot_1);
    // stage0_block1_bn2 [batchnorm]
    batchnorm2d_nhwc(slot_1, 4, 128, 16, stage0_block1_bn2_gamma, stage0_block1_bn2_beta, stage0_block1_bn2_mean, stage0_block1_bn2_var, 1e-05f, slot_2);
    // add_1 [add]
    for (int i = 0; i < 8192; ++i) {
        slot_1[i] = slot_2[i] + slot_0[i];
    }
    // stage0_block1_relu2 [relu]
    relu(slot_1, 8192);
    // stage1_block0_conv1 [conv2d]
    conv2d_nhwc(slot_1, 4, 128, 16, stage1_block0_conv1_weight, 3, 3, 16, NULL, 1, 1, 1, slot_0);
    // stage1_block0_bn1 [batchnorm]
    batchnorm2d_nhwc(slot_0, 4, 128, 16, stage1_block0_bn1_gamma, stage1_block0_bn1_beta, stage1_block0_bn1_mean, stage1_block0_bn1_var, 1e-05f, slot_2);
    // stage1_block0_relu1 [relu]
    relu(slot_2, 8192);
    // stage1_block0_conv2 [conv2d]
    conv2d_nhwc(slot_2, 4, 128, 16, stage1_block0_conv2_weight, 3, 3, 16, NULL, 1, 1, 1, slot_0);
    // stage1_block0_bn2 [batchnorm]
    batchnorm2d_nhwc(slot_0, 4, 128, 16, stage1_block0_bn2_gamma, stage1_block0_bn2_beta, stage1_block0_bn2_mean, stage1_block0_bn2_var, 1e-05f, slot_2);
    // add_2 [add]
    for (int i = 0; i < 8192; ++i) {
        slot_0[i] = slot_2[i] + slot_1[i];
    }
    // stage1_block0_relu2 [relu]
    relu(slot_0, 8192);
    // stage1_block1_conv1 [conv2d]
    conv2d_nhwc(slot_0, 4, 128, 16, stage1_block1_conv1_weight, 3, 3, 16, NULL, 1, 1, 1, slot_1);
    // stage1_block1_bn1 [batchnorm]
    batchnorm2d_nhwc(slot_1, 4, 128, 16, stage1_block1_bn1_gamma, stage1_block1_bn1_beta, stage1_block1_bn1_mean, stage1_block1_bn1_var, 1e-05f, slot_2);
    // stage1_block1_relu1 [relu]
    relu(slot_2, 8192);
    // stage1_block1_conv2 [conv2d]
    conv2d_nhwc(slot_2, 4, 128, 16, stage1_block1_conv2_weight, 3, 3, 16, NULL, 1, 1, 1, slot_1);
    // stage1_block1_bn2 [batchnorm]
    batchnorm2d_nhwc(slot_1, 4, 128, 16, stage1_block1_bn2_gamma, stage1_block1_bn2_beta, stage1_block1_bn2_mean, stage1_block1_bn2_var, 1e-05f, slot_2);
    // add_3 [add]
    for (int i = 0; i < 8192; ++i) {
        slot_1[i] = slot_2[i] + slot_0[i];
    }
    // stage1_block1_relu2 [relu]
    relu(slot_1, 8192);
    // stage2_block0_shortcut_conv [conv2d]
    conv2d_nhwc(slot_1, 4, 128, 16, stage2_block0_shortcut_conv_weight, 1, 1, 32, NULL, 2, 2, 0, slot_0);
    // stage2_block0_shortcut_bn [batchnorm]
    batchnorm2d_nhwc(slot_0, 2, 64, 32, stage2_block0_shortcut_bn_gamma, stage2_block0_shortcut_bn_beta, stage2_block0_shortcut_bn_mean, stage2_block0_shortcut_bn_var, 1e-05f, slot_2);
    // stage2_block0_conv1 [conv2d]
    conv2d_nhwc(slot_1, 4, 128, 16, stage2_block0_conv1_weight, 3, 3, 32, NULL, 2, 2, 1, slot_0);
    // stage2_block0_bn1 [batchnorm]
    batchnorm2d_nhwc(slot_0, 2, 64, 32, stage2_block0_bn1_gamma, stage2_block0_bn1_beta, stage2_block0_bn1_mean, stage2_block0_bn1_var, 1e-05f, slot_1);
    // stage2_block0_relu1 [relu]
    relu(slot_1, 4096);
    // stage2_block0_conv2 [conv2d]
    conv2d_nhwc(slot_1, 2, 64, 32, stage2_block0_conv2_weight, 3, 3, 32, NULL, 1, 1, 1, slot_0);
    // stage2_block0_bn2 [batchnorm]
    batchnorm2d_nhwc(slot_0, 2, 64, 32, stage2_block0_bn2_gamma, stage2_block0_bn2_beta, stage2_block0_bn2_mean, stage2_block0_bn2_var, 1e-05f, slot_1);
    // add_4 [add]
    for (int i = 0; i < 4096; ++i) {
        slot_0[i] = slot_1[i] + slot_2[i];
    }
    // stage2_block0_relu2 [relu]
    relu(slot_0, 4096);
    _t_end = micros();
    Serial.print("exit1: ");
    Serial.print((_t_end - _t_start) / 1000.0f, 2);
    Serial.println(" ms");
    // stage2_block1_conv1 [conv2d]
    conv2d_nhwc(slot_0, 2, 64, 32, stage2_block1_conv1_weight, 3, 3, 32, NULL, 1, 1, 1, slot_1);
    // stage2_block1_bn1 [batchnorm]
    batchnorm2d_nhwc(slot_1, 2, 64, 32, stage2_block1_bn1_gamma, stage2_block1_bn1_beta, stage2_block1_bn1_mean, stage2_block1_bn1_var, 1e-05f, slot_2);
    // stage2_block1_relu1 [relu]
    relu(slot_2, 4096);
    // stage2_block1_conv2 [conv2d]
    conv2d_nhwc(slot_2, 2, 64, 32, stage2_block1_conv2_weight, 3, 3, 32, NULL, 1, 1, 1, slot_1);
    // stage2_block1_bn2 [batchnorm]
    batchnorm2d_nhwc(slot_1, 2, 64, 32, stage2_block1_bn2_gamma, stage2_block1_bn2_beta, stage2_block1_bn2_mean, stage2_block1_bn2_var, 1e-05f, slot_2);
    // add_5 [add]
    for (int i = 0; i < 4096; ++i) {
        slot_1[i] = slot_2[i] + slot_0[i];
    }
    // stage2_block1_relu2 [relu]
    relu(slot_1, 4096);
    // stage3_block0_shortcut_conv [conv2d]
    conv2d_nhwc(slot_1, 2, 64, 32, stage3_block0_shortcut_conv_weight, 1, 1, 64, NULL, 2, 2, 0, slot_0);
    // stage3_block0_shortcut_bn [batchnorm]
    batchnorm2d_nhwc(slot_0, 1, 32, 64, stage3_block0_shortcut_bn_gamma, stage3_block0_shortcut_bn_beta, stage3_block0_shortcut_bn_mean, stage3_block0_shortcut_bn_var, 1e-05f, slot_2);
    // stage3_block0_conv1 [conv2d]
    conv2d_nhwc(slot_1, 2, 64, 32, stage3_block0_conv1_weight, 3, 3, 64, NULL, 2, 2, 1, slot_0);
    // stage3_block0_bn1 [batchnorm]
    batchnorm2d_nhwc(slot_0, 1, 32, 64, stage3_block0_bn1_gamma, stage3_block0_bn1_beta, stage3_block0_bn1_mean, stage3_block0_bn1_var, 1e-05f, slot_1);
    // stage3_block0_relu1 [relu]
    relu(slot_1, 2048);
    // stage3_block0_conv2 [conv2d]
    conv2d_nhwc(slot_1, 1, 32, 64, stage3_block0_conv2_weight, 3, 3, 64, NULL, 1, 1, 1, slot_0);
    // stage3_block0_bn2 [batchnorm]
    batchnorm2d_nhwc(slot_0, 1, 32, 64, stage3_block0_bn2_gamma, stage3_block0_bn2_beta, stage3_block0_bn2_mean, stage3_block0_bn2_var, 1e-05f, slot_1);
    // add_6 [add]
    for (int i = 0; i < 2048; ++i) {
        slot_0[i] = slot_1[i] + slot_2[i];
    }
    // stage3_block0_relu2 [relu]
    relu(slot_0, 2048);
    // stage3_block1_conv1 [conv2d]
    conv2d_nhwc(slot_0, 1, 32, 64, stage3_block1_conv1_weight, 3, 3, 64, NULL, 1, 1, 1, slot_1);
    // stage3_block1_bn1 [batchnorm]
    batchnorm2d_nhwc(slot_1, 1, 32, 64, stage3_block1_bn1_gamma, stage3_block1_bn1_beta, stage3_block1_bn1_mean, stage3_block1_bn1_var, 1e-05f, slot_2);
    // stage3_block1_relu1 [relu]
    relu(slot_2, 2048);
    // stage3_block1_conv2 [conv2d]
    conv2d_nhwc(slot_2, 1, 32, 64, stage3_block1_conv2_weight, 3, 3, 64, NULL, 1, 1, 1, slot_1);
    // stage3_block1_bn2 [batchnorm]
    batchnorm2d_nhwc(slot_1, 1, 32, 64, stage3_block1_bn2_gamma, stage3_block1_bn2_beta, stage3_block1_bn2_mean, stage3_block1_bn2_var, 1e-05f, slot_2);
    // add_7 [add]
    for (int i = 0; i < 2048; ++i) {
        slot_1[i] = slot_2[i] + slot_0[i];
    }
    // stage3_block1_relu2 [relu]
    relu(slot_1, 2048);
    // avgpool [adaptive_avg_pool]
    global_average_pool_2d(slot_1, 1, 32, 64, slot_0);
    // view [method_view]
    memcpy(slot_1, slot_0, 64 * sizeof(float));
    // fc1 [linear]
    dense(slot_1, 64, fc1_weight, fc1_bias, 64, slot_0);
    // fc1_relu [relu]
    relu(slot_0, 64);
    // fc2 [linear]
    dense(slot_0, 64, fc2_weight, fc2_bias, 10, slot_1);
    // final_softmax [softmax]
    memcpy(slot_0, slot_1, 10 * sizeof(float));
    softmax(slot_0, 10);
    _t_end = micros();
    Serial.print("Final Exit: ");
    Serial.print((_t_end - _t_start) / 1000.0f, 2);
    Serial.println(" ms");
    memcpy(output, slot_0, 10 * sizeof(float));
}
