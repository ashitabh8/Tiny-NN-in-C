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
    unsigned long _t_start = micros();
    {
        // stem_conv [conv2d]
        float buf_stem_conv[8192];
        conv2d_nhwc(input, 7, 256, 6, stem_conv_weight, 5, 5, 16, NULL, 2, 2, 1, buf_stem_conv);
        {
            // stem_bn [batchnorm]
            float buf_stem_bn[8192];
            batchnorm2d_nhwc(buf_stem_conv, 4, 128, 16, stem_bn_gamma, stem_bn_beta, stem_bn_mean, stem_bn_var, 1e-05f, buf_stem_bn);
            {
                // stem_relu [relu]
                float buf_stem_relu[8192];
                memcpy(buf_stem_relu, buf_stem_bn, 8192 * sizeof(float));
                relu(buf_stem_relu, 8192);
                {
                    // stage0_block0_conv1 [conv2d]
                    float buf_stage0_block0_conv1[8192];
                    conv2d_nhwc(buf_stem_relu, 4, 128, 16, stage0_block0_conv1_weight, 3, 3, 16, NULL, 1, 1, 1, buf_stage0_block0_conv1);
                    {
                        // stage0_block0_bn1 [batchnorm]
                        float buf_stage0_block0_bn1[8192];
                        batchnorm2d_nhwc(buf_stage0_block0_conv1, 4, 128, 16, stage0_block0_bn1_gamma, stage0_block0_bn1_beta, stage0_block0_bn1_mean, stage0_block0_bn1_var, 1e-05f, buf_stage0_block0_bn1);
                        {
                            // stage0_block0_relu1 [relu]
                            float buf_stage0_block0_relu1[8192];
                            memcpy(buf_stage0_block0_relu1, buf_stage0_block0_bn1, 8192 * sizeof(float));
                            relu(buf_stage0_block0_relu1, 8192);
                            {
                                // stage0_block0_conv2 [conv2d]
                                float buf_stage0_block0_conv2[8192];
                                conv2d_nhwc(buf_stage0_block0_relu1, 4, 128, 16, stage0_block0_conv2_weight, 3, 3, 16, NULL, 1, 1, 1, buf_stage0_block0_conv2);
                                {
                                    // stage0_block0_bn2 [batchnorm]
                                    float buf_stage0_block0_bn2[8192];
                                    batchnorm2d_nhwc(buf_stage0_block0_conv2, 4, 128, 16, stage0_block0_bn2_gamma, stage0_block0_bn2_beta, stage0_block0_bn2_mean, stage0_block0_bn2_var, 1e-05f, buf_stage0_block0_bn2);
                                    {
                                        // add [add]
                                        float buf_add[8192];
                                        for (int i = 0; i < 8192; ++i) {
                                            buf_add[i] = buf_stage0_block0_bn2[i] + buf_stem_relu[i];
                                        }
                                        {
                                            // stage0_block0_relu2 [relu]
                                            float buf_stage0_block0_relu2[8192];
                                            memcpy(buf_stage0_block0_relu2, buf_add, 8192 * sizeof(float));
                                            relu(buf_stage0_block0_relu2, 8192);
                                            {
                                                // stage0_block1_conv1 [conv2d]
                                                float buf_stage0_block1_conv1[8192];
                                                conv2d_nhwc(buf_stage0_block0_relu2, 4, 128, 16, stage0_block1_conv1_weight, 3, 3, 16, NULL, 1, 1, 1, buf_stage0_block1_conv1);
                                                {
                                                    // stage0_block1_bn1 [batchnorm]
                                                    float buf_stage0_block1_bn1[8192];
                                                    batchnorm2d_nhwc(buf_stage0_block1_conv1, 4, 128, 16, stage0_block1_bn1_gamma, stage0_block1_bn1_beta, stage0_block1_bn1_mean, stage0_block1_bn1_var, 1e-05f, buf_stage0_block1_bn1);
                                                    {
                                                        // stage0_block1_relu1 [relu]
                                                        float buf_stage0_block1_relu1[8192];
                                                        memcpy(buf_stage0_block1_relu1, buf_stage0_block1_bn1, 8192 * sizeof(float));
                                                        relu(buf_stage0_block1_relu1, 8192);
                                                        {
                                                            // stage0_block1_conv2 [conv2d]
                                                            float buf_stage0_block1_conv2[8192];
                                                            conv2d_nhwc(buf_stage0_block1_relu1, 4, 128, 16, stage0_block1_conv2_weight, 3, 3, 16, NULL, 1, 1, 1, buf_stage0_block1_conv2);
                                                            {
                                                                // stage0_block1_bn2 [batchnorm]
                                                                float buf_stage0_block1_bn2[8192];
                                                                batchnorm2d_nhwc(buf_stage0_block1_conv2, 4, 128, 16, stage0_block1_bn2_gamma, stage0_block1_bn2_beta, stage0_block1_bn2_mean, stage0_block1_bn2_var, 1e-05f, buf_stage0_block1_bn2);
                                                                {
                                                                    // add_1 [add]
                                                                    float buf_add_1[8192];
                                                                    for (int i = 0; i < 8192; ++i) {
                                                                        buf_add_1[i] = buf_stage0_block1_bn2[i] + buf_stage0_block0_relu2[i];
                                                                    }
                                                                    {
                                                                        // stage0_block1_relu2 [relu]
                                                                        float buf_stage0_block1_relu2[8192];
                                                                        memcpy(buf_stage0_block1_relu2, buf_add_1, 8192 * sizeof(float));
                                                                        relu(buf_stage0_block1_relu2, 8192);
                                                                        {
                                                                            // stage1_block0_conv1 [conv2d]
                                                                            float buf_stage1_block0_conv1[8192];
                                                                            conv2d_nhwc(buf_stage0_block1_relu2, 4, 128, 16, stage1_block0_conv1_weight, 3, 3, 16, NULL, 1, 1, 1, buf_stage1_block0_conv1);
                                                                            {
                                                                                // stage1_block0_bn1 [batchnorm]
                                                                                float buf_stage1_block0_bn1[8192];
                                                                                batchnorm2d_nhwc(buf_stage1_block0_conv1, 4, 128, 16, stage1_block0_bn1_gamma, stage1_block0_bn1_beta, stage1_block0_bn1_mean, stage1_block0_bn1_var, 1e-05f, buf_stage1_block0_bn1);
                                                                                {
                                                                                    // stage1_block0_relu1 [relu]
                                                                                    float buf_stage1_block0_relu1[8192];
                                                                                    memcpy(buf_stage1_block0_relu1, buf_stage1_block0_bn1, 8192 * sizeof(float));
                                                                                    relu(buf_stage1_block0_relu1, 8192);
                                                                                    {
                                                                                        // stage1_block0_conv2 [conv2d]
                                                                                        float buf_stage1_block0_conv2[8192];
                                                                                        conv2d_nhwc(buf_stage1_block0_relu1, 4, 128, 16, stage1_block0_conv2_weight, 3, 3, 16, NULL, 1, 1, 1, buf_stage1_block0_conv2);
                                                                                        {
                                                                                            // stage1_block0_bn2 [batchnorm]
                                                                                            float buf_stage1_block0_bn2[8192];
                                                                                            batchnorm2d_nhwc(buf_stage1_block0_conv2, 4, 128, 16, stage1_block0_bn2_gamma, stage1_block0_bn2_beta, stage1_block0_bn2_mean, stage1_block0_bn2_var, 1e-05f, buf_stage1_block0_bn2);
                                                                                            {
                                                                                                // add_2 [add]
                                                                                                float buf_add_2[8192];
                                                                                                for (int i = 0; i < 8192; ++i) {
                                                                                                    buf_add_2[i] = buf_stage1_block0_bn2[i] + buf_stage0_block1_relu2[i];
                                                                                                }
                                                                                                {
                                                                                                    // stage1_block0_relu2 [relu]
                                                                                                    float buf_stage1_block0_relu2[8192];
                                                                                                    memcpy(buf_stage1_block0_relu2, buf_add_2, 8192 * sizeof(float));
                                                                                                    relu(buf_stage1_block0_relu2, 8192);
                                                                                                    {
                                                                                                        // stage1_block1_conv1 [conv2d]
                                                                                                        float buf_stage1_block1_conv1[8192];
                                                                                                        conv2d_nhwc(buf_stage1_block0_relu2, 4, 128, 16, stage1_block1_conv1_weight, 3, 3, 16, NULL, 1, 1, 1, buf_stage1_block1_conv1);
                                                                                                        {
                                                                                                            // stage1_block1_bn1 [batchnorm]
                                                                                                            float buf_stage1_block1_bn1[8192];
                                                                                                            batchnorm2d_nhwc(buf_stage1_block1_conv1, 4, 128, 16, stage1_block1_bn1_gamma, stage1_block1_bn1_beta, stage1_block1_bn1_mean, stage1_block1_bn1_var, 1e-05f, buf_stage1_block1_bn1);
                                                                                                            {
                                                                                                                // stage1_block1_relu1 [relu]
                                                                                                                float buf_stage1_block1_relu1[8192];
                                                                                                                memcpy(buf_stage1_block1_relu1, buf_stage1_block1_bn1, 8192 * sizeof(float));
                                                                                                                relu(buf_stage1_block1_relu1, 8192);
                                                                                                                {
                                                                                                                    // stage1_block1_conv2 [conv2d]
                                                                                                                    float buf_stage1_block1_conv2[8192];
                                                                                                                    conv2d_nhwc(buf_stage1_block1_relu1, 4, 128, 16, stage1_block1_conv2_weight, 3, 3, 16, NULL, 1, 1, 1, buf_stage1_block1_conv2);
                                                                                                                    {
                                                                                                                        // stage1_block1_bn2 [batchnorm]
                                                                                                                        float buf_stage1_block1_bn2[8192];
                                                                                                                        batchnorm2d_nhwc(buf_stage1_block1_conv2, 4, 128, 16, stage1_block1_bn2_gamma, stage1_block1_bn2_beta, stage1_block1_bn2_mean, stage1_block1_bn2_var, 1e-05f, buf_stage1_block1_bn2);
                                                                                                                        {
                                                                                                                            // add_3 [add]
                                                                                                                            float buf_add_3[8192];
                                                                                                                            for (int i = 0; i < 8192; ++i) {
                                                                                                                                buf_add_3[i] = buf_stage1_block1_bn2[i] + buf_stage1_block0_relu2[i];
                                                                                                                            }
                                                                                                                            {
                                                                                                                                // stage1_block1_relu2 [relu]
                                                                                                                                float buf_stage1_block1_relu2[8192];
                                                                                                                                memcpy(buf_stage1_block1_relu2, buf_add_3, 8192 * sizeof(float));
                                                                                                                                relu(buf_stage1_block1_relu2, 8192);
                                                                                                                                {
                                                                                                                                    // stage2_block0_shortcut_conv [conv2d]
                                                                                                                                    float buf_stage2_block0_shortcut_conv[4096];
                                                                                                                                    conv2d_nhwc(buf_stage1_block1_relu2, 4, 128, 16, stage2_block0_shortcut_conv_weight, 1, 1, 32, NULL, 2, 2, 0, buf_stage2_block0_shortcut_conv);
                                                                                                                                    {
                                                                                                                                        // stage2_block0_shortcut_bn [batchnorm]
                                                                                                                                        float buf_stage2_block0_shortcut_bn[4096];
                                                                                                                                        batchnorm2d_nhwc(buf_stage2_block0_shortcut_conv, 2, 64, 32, stage2_block0_shortcut_bn_gamma, stage2_block0_shortcut_bn_beta, stage2_block0_shortcut_bn_mean, stage2_block0_shortcut_bn_var, 1e-05f, buf_stage2_block0_shortcut_bn);
                                                                                                                                        {
                                                                                                                                            // stage2_block0_conv1 [conv2d]
                                                                                                                                            float buf_stage2_block0_conv1[4096];
                                                                                                                                            conv2d_nhwc(buf_stage1_block1_relu2, 4, 128, 16, stage2_block0_conv1_weight, 3, 3, 32, NULL, 2, 2, 1, buf_stage2_block0_conv1);
                                                                                                                                            {
                                                                                                                                                // stage2_block0_bn1 [batchnorm]
                                                                                                                                                float buf_stage2_block0_bn1[4096];
                                                                                                                                                batchnorm2d_nhwc(buf_stage2_block0_conv1, 2, 64, 32, stage2_block0_bn1_gamma, stage2_block0_bn1_beta, stage2_block0_bn1_mean, stage2_block0_bn1_var, 1e-05f, buf_stage2_block0_bn1);
                                                                                                                                                {
                                                                                                                                                    // stage2_block0_relu1 [relu]
                                                                                                                                                    float buf_stage2_block0_relu1[4096];
                                                                                                                                                    memcpy(buf_stage2_block0_relu1, buf_stage2_block0_bn1, 4096 * sizeof(float));
                                                                                                                                                    relu(buf_stage2_block0_relu1, 4096);
                                                                                                                                                    {
                                                                                                                                                        // stage2_block0_conv2 [conv2d]
                                                                                                                                                        float buf_stage2_block0_conv2[4096];
                                                                                                                                                        conv2d_nhwc(buf_stage2_block0_relu1, 2, 64, 32, stage2_block0_conv2_weight, 3, 3, 32, NULL, 1, 1, 1, buf_stage2_block0_conv2);
                                                                                                                                                        {
                                                                                                                                                            // stage2_block0_bn2 [batchnorm]
                                                                                                                                                            float buf_stage2_block0_bn2[4096];
                                                                                                                                                            batchnorm2d_nhwc(buf_stage2_block0_conv2, 2, 64, 32, stage2_block0_bn2_gamma, stage2_block0_bn2_beta, stage2_block0_bn2_mean, stage2_block0_bn2_var, 1e-05f, buf_stage2_block0_bn2);
                                                                                                                                                            {
                                                                                                                                                                // add_4 [add]
                                                                                                                                                                float buf_add_4[4096];
                                                                                                                                                                for (int i = 0; i < 4096; ++i) {
                                                                                                                                                                    buf_add_4[i] = buf_stage2_block0_bn2[i] + buf_stage2_block0_shortcut_bn[i];
                                                                                                                                                                }
                                                                                                                                                                {
                                                                                                                                                                    // stage2_block0_relu2 [relu]
                                                                                                                                                                    float buf_stage2_block0_relu2[4096];
                                                                                                                                                                    memcpy(buf_stage2_block0_relu2, buf_add_4, 4096 * sizeof(float));
                                                                                                                                                                    relu(buf_stage2_block0_relu2, 4096);
                                                                                                                                                                    unsigned long _t_end = micros();
                                                                                                                                                                    Serial.print("exit1: ");
                                                                                                                                                                    Serial.print((_t_end - _t_start) / 1000.0f, 2);
                                                                                                                                                                    Serial.println(" ms");
                                                                                                                                                                    {
                                                                                                                                                                        // stage2_block1_conv1 [conv2d]
                                                                                                                                                                        float buf_stage2_block1_conv1[4096];
                                                                                                                                                                        conv2d_nhwc(buf_stage2_block0_relu2, 2, 64, 32, stage2_block1_conv1_weight, 3, 3, 32, NULL, 1, 1, 1, buf_stage2_block1_conv1);
                                                                                                                                                                        {
                                                                                                                                                                            // stage2_block1_bn1 [batchnorm]
                                                                                                                                                                            float buf_stage2_block1_bn1[4096];
                                                                                                                                                                            batchnorm2d_nhwc(buf_stage2_block1_conv1, 2, 64, 32, stage2_block1_bn1_gamma, stage2_block1_bn1_beta, stage2_block1_bn1_mean, stage2_block1_bn1_var, 1e-05f, buf_stage2_block1_bn1);
                                                                                                                                                                            {
                                                                                                                                                                                // stage2_block1_relu1 [relu]
                                                                                                                                                                                float buf_stage2_block1_relu1[4096];
                                                                                                                                                                                memcpy(buf_stage2_block1_relu1, buf_stage2_block1_bn1, 4096 * sizeof(float));
                                                                                                                                                                                relu(buf_stage2_block1_relu1, 4096);
                                                                                                                                                                                {
                                                                                                                                                                                    // stage2_block1_conv2 [conv2d]
                                                                                                                                                                                    float buf_stage2_block1_conv2[4096];
                                                                                                                                                                                    conv2d_nhwc(buf_stage2_block1_relu1, 2, 64, 32, stage2_block1_conv2_weight, 3, 3, 32, NULL, 1, 1, 1, buf_stage2_block1_conv2);
                                                                                                                                                                                    {
                                                                                                                                                                                        // stage2_block1_bn2 [batchnorm]
                                                                                                                                                                                        float buf_stage2_block1_bn2[4096];
                                                                                                                                                                                        batchnorm2d_nhwc(buf_stage2_block1_conv2, 2, 64, 32, stage2_block1_bn2_gamma, stage2_block1_bn2_beta, stage2_block1_bn2_mean, stage2_block1_bn2_var, 1e-05f, buf_stage2_block1_bn2);
                                                                                                                                                                                        {
                                                                                                                                                                                            // add_5 [add]
                                                                                                                                                                                            float buf_add_5[4096];
                                                                                                                                                                                            for (int i = 0; i < 4096; ++i) {
                                                                                                                                                                                                buf_add_5[i] = buf_stage2_block1_bn2[i] + buf_stage2_block0_relu2[i];
                                                                                                                                                                                            }
                                                                                                                                                                                            {
                                                                                                                                                                                                // stage2_block1_relu2 [relu]
                                                                                                                                                                                                float buf_stage2_block1_relu2[4096];
                                                                                                                                                                                                memcpy(buf_stage2_block1_relu2, buf_add_5, 4096 * sizeof(float));
                                                                                                                                                                                                relu(buf_stage2_block1_relu2, 4096);
                                                                                                                                                                                                {
                                                                                                                                                                                                    // stage3_block0_shortcut_conv [conv2d]
                                                                                                                                                                                                    float buf_stage3_block0_shortcut_conv[2048];
                                                                                                                                                                                                    conv2d_nhwc(buf_stage2_block1_relu2, 2, 64, 32, stage3_block0_shortcut_conv_weight, 1, 1, 64, NULL, 2, 2, 0, buf_stage3_block0_shortcut_conv);
                                                                                                                                                                                                    {
                                                                                                                                                                                                        // stage3_block0_shortcut_bn [batchnorm]
                                                                                                                                                                                                        float buf_stage3_block0_shortcut_bn[2048];
                                                                                                                                                                                                        batchnorm2d_nhwc(buf_stage3_block0_shortcut_conv, 1, 32, 64, stage3_block0_shortcut_bn_gamma, stage3_block0_shortcut_bn_beta, stage3_block0_shortcut_bn_mean, stage3_block0_shortcut_bn_var, 1e-05f, buf_stage3_block0_shortcut_bn);
                                                                                                                                                                                                        {
                                                                                                                                                                                                            // stage3_block0_conv1 [conv2d]
                                                                                                                                                                                                            float buf_stage3_block0_conv1[2048];
                                                                                                                                                                                                            conv2d_nhwc(buf_stage2_block1_relu2, 2, 64, 32, stage3_block0_conv1_weight, 3, 3, 64, NULL, 2, 2, 1, buf_stage3_block0_conv1);
                                                                                                                                                                                                            {
                                                                                                                                                                                                                // stage3_block0_bn1 [batchnorm]
                                                                                                                                                                                                                float buf_stage3_block0_bn1[2048];
                                                                                                                                                                                                                batchnorm2d_nhwc(buf_stage3_block0_conv1, 1, 32, 64, stage3_block0_bn1_gamma, stage3_block0_bn1_beta, stage3_block0_bn1_mean, stage3_block0_bn1_var, 1e-05f, buf_stage3_block0_bn1);
                                                                                                                                                                                                                {
                                                                                                                                                                                                                    // stage3_block0_relu1 [relu]
                                                                                                                                                                                                                    float buf_stage3_block0_relu1[2048];
                                                                                                                                                                                                                    memcpy(buf_stage3_block0_relu1, buf_stage3_block0_bn1, 2048 * sizeof(float));
                                                                                                                                                                                                                    relu(buf_stage3_block0_relu1, 2048);
                                                                                                                                                                                                                    {
                                                                                                                                                                                                                        // stage3_block0_conv2 [conv2d]
                                                                                                                                                                                                                        float buf_stage3_block0_conv2[2048];
                                                                                                                                                                                                                        conv2d_nhwc(buf_stage3_block0_relu1, 1, 32, 64, stage3_block0_conv2_weight, 3, 3, 64, NULL, 1, 1, 1, buf_stage3_block0_conv2);
                                                                                                                                                                                                                        {
                                                                                                                                                                                                                            // stage3_block0_bn2 [batchnorm]
                                                                                                                                                                                                                            float buf_stage3_block0_bn2[2048];
                                                                                                                                                                                                                            batchnorm2d_nhwc(buf_stage3_block0_conv2, 1, 32, 64, stage3_block0_bn2_gamma, stage3_block0_bn2_beta, stage3_block0_bn2_mean, stage3_block0_bn2_var, 1e-05f, buf_stage3_block0_bn2);
                                                                                                                                                                                                                            {
                                                                                                                                                                                                                                // add_6 [add]
                                                                                                                                                                                                                                float buf_add_6[2048];
                                                                                                                                                                                                                                for (int i = 0; i < 2048; ++i) {
                                                                                                                                                                                                                                    buf_add_6[i] = buf_stage3_block0_bn2[i] + buf_stage3_block0_shortcut_bn[i];
                                                                                                                                                                                                                                }
                                                                                                                                                                                                                                {
                                                                                                                                                                                                                                    // stage3_block0_relu2 [relu]
                                                                                                                                                                                                                                    float buf_stage3_block0_relu2[2048];
                                                                                                                                                                                                                                    memcpy(buf_stage3_block0_relu2, buf_add_6, 2048 * sizeof(float));
                                                                                                                                                                                                                                    relu(buf_stage3_block0_relu2, 2048);
                                                                                                                                                                                                                                    {
                                                                                                                                                                                                                                        // stage3_block1_conv1 [conv2d]
                                                                                                                                                                                                                                        float buf_stage3_block1_conv1[2048];
                                                                                                                                                                                                                                        conv2d_nhwc(buf_stage3_block0_relu2, 1, 32, 64, stage3_block1_conv1_weight, 3, 3, 64, NULL, 1, 1, 1, buf_stage3_block1_conv1);
                                                                                                                                                                                                                                        {
                                                                                                                                                                                                                                            // stage3_block1_bn1 [batchnorm]
                                                                                                                                                                                                                                            float buf_stage3_block1_bn1[2048];
                                                                                                                                                                                                                                            batchnorm2d_nhwc(buf_stage3_block1_conv1, 1, 32, 64, stage3_block1_bn1_gamma, stage3_block1_bn1_beta, stage3_block1_bn1_mean, stage3_block1_bn1_var, 1e-05f, buf_stage3_block1_bn1);
                                                                                                                                                                                                                                            {
                                                                                                                                                                                                                                                // stage3_block1_relu1 [relu]
                                                                                                                                                                                                                                                float buf_stage3_block1_relu1[2048];
                                                                                                                                                                                                                                                memcpy(buf_stage3_block1_relu1, buf_stage3_block1_bn1, 2048 * sizeof(float));
                                                                                                                                                                                                                                                relu(buf_stage3_block1_relu1, 2048);
                                                                                                                                                                                                                                                {
                                                                                                                                                                                                                                                    // stage3_block1_conv2 [conv2d]
                                                                                                                                                                                                                                                    float buf_stage3_block1_conv2[2048];
                                                                                                                                                                                                                                                    conv2d_nhwc(buf_stage3_block1_relu1, 1, 32, 64, stage3_block1_conv2_weight, 3, 3, 64, NULL, 1, 1, 1, buf_stage3_block1_conv2);
                                                                                                                                                                                                                                                    {
                                                                                                                                                                                                                                                        // stage3_block1_bn2 [batchnorm]
                                                                                                                                                                                                                                                        float buf_stage3_block1_bn2[2048];
                                                                                                                                                                                                                                                        batchnorm2d_nhwc(buf_stage3_block1_conv2, 1, 32, 64, stage3_block1_bn2_gamma, stage3_block1_bn2_beta, stage3_block1_bn2_mean, stage3_block1_bn2_var, 1e-05f, buf_stage3_block1_bn2);
                                                                                                                                                                                                                                                        {
                                                                                                                                                                                                                                                            // add_7 [add]
                                                                                                                                                                                                                                                            float buf_add_7[2048];
                                                                                                                                                                                                                                                            for (int i = 0; i < 2048; ++i) {
                                                                                                                                                                                                                                                                buf_add_7[i] = buf_stage3_block1_bn2[i] + buf_stage3_block0_relu2[i];
                                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                                            {
                                                                                                                                                                                                                                                                // stage3_block1_relu2 [relu]
                                                                                                                                                                                                                                                                float buf_stage3_block1_relu2[2048];
                                                                                                                                                                                                                                                                memcpy(buf_stage3_block1_relu2, buf_add_7, 2048 * sizeof(float));
                                                                                                                                                                                                                                                                relu(buf_stage3_block1_relu2, 2048);
                                                                                                                                                                                                                                                                {
                                                                                                                                                                                                                                                                    // avgpool [adaptive_avg_pool]
                                                                                                                                                                                                                                                                    float buf_avgpool[64];
                                                                                                                                                                                                                                                                    global_average_pool_2d(buf_stage3_block1_relu2, 1, 32, 64, buf_avgpool);
                                                                                                                                                                                                                                                                    {
                                                                                                                                                                                                                                                                        // view [method_view]
                                                                                                                                                                                                                                                                        float buf_view[64];
                                                                                                                                                                                                                                                                        memcpy(buf_view, buf_avgpool, 64 * sizeof(float));
                                                                                                                                                                                                                                                                        {
                                                                                                                                                                                                                                                                            // fc1 [linear]
                                                                                                                                                                                                                                                                            float buf_fc1[64];
                                                                                                                                                                                                                                                                            dense(buf_view, 64, fc1_weight, fc1_bias, 64, buf_fc1);
                                                                                                                                                                                                                                                                            {
                                                                                                                                                                                                                                                                                // fc1_relu [relu]
                                                                                                                                                                                                                                                                                float buf_fc1_relu[64];
                                                                                                                                                                                                                                                                                memcpy(buf_fc1_relu, buf_fc1, 64 * sizeof(float));
                                                                                                                                                                                                                                                                                relu(buf_fc1_relu, 64);
                                                                                                                                                                                                                                                                                {
                                                                                                                                                                                                                                                                                    // fc2 [linear]
                                                                                                                                                                                                                                                                                    float buf_fc2[10];
                                                                                                                                                                                                                                                                                    dense(buf_fc1_relu, 64, fc2_weight, fc2_bias, 10, buf_fc2);
                                                                                                                                                                                                                                                                                    {
                                                                                                                                                                                                                                                                                        // final_softmax [softmax]
                                                                                                                                                                                                                                                                                        float buf_final_softmax[10];
                                                                                                                                                                                                                                                                                        memcpy(buf_final_softmax, buf_fc2, 10 * sizeof(float));
                                                                                                                                                                                                                                                                                        softmax(buf_final_softmax, 10);
                                                                                                                                                                                                                                                                                        unsigned long _t_end = micros();
                                                                                                                                                                                                                                                                                        Serial.print("Final Exit: ");
                                                                                                                                                                                                                                                                                        Serial.print((_t_end - _t_start) / 1000.0f, 2);
                                                                                                                                                                                                                                                                                        Serial.println(" ms");
                                                                                                                                                                                                                                                                                        memcpy(output, buf_final_softmax, 10 * sizeof(float));
                                                                                                                                                                                                                                                                                    }
                                                                                                                                                                                                                                                                                }
                                                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                                                        }
                                                                                                                                                                                                                                                                    }
                                                                                                                                                                                                                                                                }
                                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                                        }
                                                                                                                                                                                                                                                    }
                                                                                                                                                                                                                                                }
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                        }
                                                                                                                                                                                                                                    }
                                                                                                                                                                                                                                }
                                                                                                                                                                                                                            }
                                                                                                                                                                                                                        }
                                                                                                                                                                                                                    }
                                                                                                                                                                                                                }
                                                                                                                                                                                                            }
                                                                                                                                                                                                        }
                                                                                                                                                                                                    }
                                                                                                                                                                                                }
                                                                                                                                                                                            }
                                                                                                                                                                                        }
                                                                                                                                                                                    }
                                                                                                                                                                                }
                                                                                                                                                                            }
                                                                                                                                                                        }
                                                                                                                                                                    }
                                                                                                                                                                }
                                                                                                                                                            }
                                                                                                                                                        }
                                                                                                                                                    }
                                                                                                                                                }
                                                                                                                                            }
                                                                                                                                        }
                                                                                                                                    }
                                                                                                                                }
                                                                                                                            }
                                                                                                                        }
                                                                                                                    }
                                                                                                                }
                                                                                                            }
                                                                                                        }
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
