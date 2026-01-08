// Test harness template for comparing PyTorch and C outputs
// This will be used to run the generated C model

#ifndef TEST_HARNESS_H_
#define TEST_HARNESS_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Will be replaced with actual model
#include "model.h"
#include "weights.h"

// Read binary float array from file
int read_binary_floats(const char* filename, float* buffer, int size) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open input file: %s\n", filename);
        return -1;
    }
    
    size_t read = fread(buffer, sizeof(float), size, f);
    fclose(f);
    
    if (read != size) {
        fprintf(stderr, "Expected %d floats, read %zu\n", size, read);
        return -1;
    }
    
    return 0;
}

// Write binary float array to file
int write_binary_floats(const char* filename, const float* buffer, int size) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Failed to open output file: %s\n", filename);
        return -1;
    }
    
    size_t written = fwrite(buffer, sizeof(float), size, f);
    fclose(f);
    
    if (written != size) {
        fprintf(stderr, "Expected to write %d floats, wrote %zu\n", size, written);
        return -1;
    }
    
    return 0;
}

#endif // TEST_HARNESS_H_

