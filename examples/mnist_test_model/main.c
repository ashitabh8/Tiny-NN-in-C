/*
 * Run the generated MNIST C model on embedded test_inputs.h samples.
 * Writes c_outputs.txt (one line per sample: 10 space-separated logits).
 *
 * Build (from this directory):
 *   gcc -O2 -I../../tmp/mnist_cnn -o test_runner main.c ../../tmp/mnist_cnn/model.c -lm
 */

#include <stdio.h>
#include <stdlib.h>

#include "model.h"
#include "test_inputs.h"

int main(void) {
    FILE *f = fopen("c_outputs.txt", "w");
    if (!f) {
        perror("c_outputs.txt");
        return 1;
    }

    float out[10];
    for (int i = 0; i < NUM_SAMPLES; ++i) {
        model_forward(test_samples[i], out);
        for (int j = 0; j < 10; ++j) {
            fprintf(f, "%.9e", (double)out[j]);
            if (j < 9) {
                fputc(' ', f);
            }
        }
        fputc('\n', f);
    }

    fclose(f);
    return 0;
}
