// Auto-generated Arduino sketch: resnet_1_sketch
// DO NOT EDIT

// Input shape  (NCHW): [1, 6, 7, 256]
// Output shape: [1, 10]

#include "model.h"

// Global buffers — avoids stack overflow for large activations
static float input_buf[10752];
static float output_buf[10];
static bool _inference_done = false;

void setup() {
    Serial.begin(115200);
    while (!Serial) {}
    // TODO: fill input_buf with real sensor data before inference
    for (int i = 0; i < 10752; ++i)
        input_buf[i] = 0.0f;
}

void loop() {
    if (_inference_done) return;
    _inference_done = true;

    model_forward(input_buf, output_buf);
}
