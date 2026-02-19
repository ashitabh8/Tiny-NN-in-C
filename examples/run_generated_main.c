/*
 * Minimal main to run inference with generated model.
 * Compile from project root, e.g.:
 *   gcc -O2 -I <OUT_DIR> -o run_inference <OUT_DIR>/model.c examples/run_generated_main.c -lm
 * Then: ./run_inference
 *
 * For tiny_resnet float: OUT_DIR=tmp/tiny_resnet_for_embedded_device_float
 * Input: NHWC (1, 1, 200, 10) = 2000 floats. Output: 10 floats.
 */
#include <stdio.h>
#include <stdlib.h>
#include "model.h"

#define INPUT_SIZE  (1 * 1 * 200 * 10)
#define OUTPUT_SIZE 10

int main(void) {
    float* input = (float*)malloc(INPUT_SIZE * sizeof(float));
    float* output = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    if (!input || !output) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    /* Small deterministic input so output is reproducible */
    for (int i = 0; i < INPUT_SIZE; ++i)
        input[i] = 0.01f * (float)(i % 100);

    model_forward(input, output);

    printf("C inference output (first %d): ", OUTPUT_SIZE);
    for (int i = 0; i < OUTPUT_SIZE; ++i)
        printf("%.4f ", output[i]);
    printf("\n");

    free(input);
    free(output);
    return 0;
}
