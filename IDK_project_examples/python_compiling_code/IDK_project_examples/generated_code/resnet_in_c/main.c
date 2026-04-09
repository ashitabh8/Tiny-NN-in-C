/*
 * Runner for resnet_in_c
 *
 * Model I/O:
 *   Input  (PyTorch NCHW): [1, 6, 7, 256]
 *   Input  (this runtime): float[H][W][C]  =  float[7][256][6]
 *   Output : [1, 5]  →  5 class probabilities (softmax)
 *
 * Build:
 *   gcc -O2 -o runner main.c model.c -lm
 *
 * Run:
 *   ./runner
 *
 * ============================================================================
 * INPUT FORMAT: float[H][W][C]  —  Height × Width × Channel
 * ============================================================================
 *
 * model_forward() expects data in HWC order: for every spatial position (h, w),
 * all C channel values sit next to each other in memory.
 *
 * In C, declaring  float input[H][W][C]  gives you exactly this layout for
 * free — the language stores multi-dimensional arrays in row-major order, so
 * the last index (C) varies fastest and is always contiguous (stride 1).
 * You can pass it directly to model_forward() with a simple cast:
 *
 *   model_forward((const float*)input, output);
 *
 * No copy, no conversion needed.
 *
 * If your data arrives from PyTorch / a pipeline in CHW order (channels first,
 * the PyTorch default), use chw_to_hwc() below to re-order it first.
 *
 * ============================================================================
 * WHY HWC AND NOT CHW?
 * ============================================================================
 *
 * PyTorch uses CHW (channels outermost).  This runtime uses HWC (channels
 * innermost) because it is faster on microcontrollers.
 *
 * The hot inner loop of conv2d reads all C channels for one pixel at a time:
 *
 *   const float* pixel = in + (h * W + w) * C;
 *   for (int c = 0; c < C; ++c)
 *       acc += pixel[c] * filter[c];   // stride-1: sequential memory access
 *
 * With HWC, pixel[0]…pixel[C-1] are contiguous — one straight read through
 * memory.  With CHW each channel would be H*W floats apart, forcing the CPU
 * to jump across the entire buffer for every multiply-accumulate — very slow
 * on tiny cores with little or no cache.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "model.h"

/* ---- Model dimensions ------------------------------------------------------ */
#define C  6    /* channels                                                     */
#define H  7    /* height                                                       */
#define W  256  /* width                                                        */

#define OUTPUT_SIZE 5

/* ---- Global buffers --------------------------------------------------------
 * Declared static so they live in BSS, not on the stack.
 * On microcontrollers the default stack is only a few KB; a 42 KB array
 * would overflow it instantly.
 * --------------------------------------------------------------------------- */
static float input[H][W][C];    /* 3D HWC input — fill this with your data     */
static float output[OUTPUT_SIZE];


/* ============================================================================
 * chw_to_hwc  —  convert a CHW buffer (PyTorch layout) to a HWC 3D array
 *
 * Use this when your data comes from PyTorch or any channels-first pipeline.
 * In Python, the equivalent is:  tensor.permute(1, 2, 0).numpy()
 *
 * src layout [C][H][W]: channel c, row h, column w  →  src[c][h][w]
 * dst layout [H][W][C]: same element lives at        →  dst[h][w][c]
 * ============================================================================ */
static void chw_to_hwc(const float src[C][H][W], float dst[H][W][C]) {
    for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
            for (int c = 0; c < C; ++c)
                dst[h][w][c] = src[c][h][w];
}


int main(void) {
    srand((unsigned int)time(NULL));

    /* ------------------------------------------------------------------
     * Fill the HWC input with random data.
     * In a real application, replace this with sensor readings or a call
     * to chw_to_hwc() if your data arrives in CHW (PyTorch) format.
     * ------------------------------------------------------------------ */
    for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
            for (int c = 0; c < C; ++c)
                input[h][w][c] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;

    printf("Input: random data, shape [H=%d][W=%d][C=%d]\n", H, W, C);
    printf("Running model_forward...\n\n");

    /* ------------------------------------------------------------------
     * Run inference.
     *
     * float[H][W][C] is already laid out in HWC order in memory (C
     * row-major = last index contiguous), so a plain cast is all that
     * is needed — no copy, no flattening step.
     * ------------------------------------------------------------------ */
    model_forward((const float*)input, output);

    /* ------------------------------------------------------------------
     * Print results.
     * ------------------------------------------------------------------ */
    printf("Output scores (softmax over %d classes):\n", OUTPUT_SIZE);
    for (int i = 0; i < OUTPUT_SIZE; ++i)
        printf("  class %d: %.6f\n", i, output[i]);

    int best = 0;
    for (int i = 1; i < OUTPUT_SIZE; ++i)
        if (output[i] > output[best]) best = i;

    printf("\nPredicted class: %d\n", best);
    return 0;
}
