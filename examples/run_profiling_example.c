/*
 * Runner for the profiling_example generated model.
 * Feeds a randomly generated NHWC input through model_forward() and prints
 * the output.  model.c already contains clock()-based timing printouts for
 * each profiled op, so simply running this binary shows per-layer latency.
 *
 * Input shape (PyTorch NCHW): (1, 2, 4, 4)
 * Input shape after NHWC permute: (1, 4, 4, 2) = 32 floats
 * Output size: 2 floats
 *
 * -------------------------------------------------------------------------
 * Compile (from project root):
 *
 *   gcc -O2 \
 *       -I tmp/profiling_example \
 *       -o run_profiling \
 *       tmp/profiling_example/model.c \
 *       examples/run_profiling_example.c \
 *       -lm
 *
 * Run:
 *
 *   ./run_profiling
 *
 * -------------------------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "model.h"

/* NHWC layout: N=1, H=4, W=4, C=2  =>  1 * 4 * 4 * 2 = 32 */
#define INPUT_SIZE  32
#define OUTPUT_SIZE 2

int main(void) {
    srand((unsigned int)time(NULL));

    float input[INPUT_SIZE];
    float output[OUTPUT_SIZE];

    /* Random floats in [-1, 1] to mimic normalised image data */
    for (int i = 0; i < INPUT_SIZE; ++i)
        input[i] = 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f;

    printf("Running model_forward on random input (%d floats, NHWC 1x4x4x2)...\n\n",
           INPUT_SIZE);

    model_forward(input, output);

    printf("\nOutput (%d values):", OUTPUT_SIZE);
    for (int i = 0; i < OUTPUT_SIZE; ++i)
        printf("  %.6f", output[i]);
    printf("\n");

    return 0;
}
