/*
 * Sample Arduino runner for the generated resnet_1 model.
 *
 * Copy this file and all files from:
 *   IDK_project_examples/generated_code/resnet_1_arduino/
 * into a single Arduino sketch folder named "resnet_1_sketch" (folder name
 * must match the .ino filename for the Arduino IDE / arduino-cli).
 *
 * Board: Arduino Giga R1  (FQBN: arduino:mbed_giga:giga)
 *
 * Compile check (no upload):
 *   arduino-cli compile --fqbn arduino:mbed_giga:giga resnet_1_sketch/
 *
 * Model I/O:
 *   Input  (NCHW): [1, 6, 7, 256]  ->  NHWC flat: 10752 floats
 *   Output        : [1, 10]         ->  10 class scores
 */

#include "model.h"

// ---------------------------------------------------------------------------
// Buffer sizes
// ---------------------------------------------------------------------------
#define INPUT_SIZE  10752   // 1 * 7 * 256 * 6  (NHWC)
#define OUTPUT_SIZE 10

// Global arrays — keeps them off the stack (avoids stack overflow)
static float input_buf[INPUT_SIZE];
static float output_buf[OUTPUT_SIZE];
static bool  _done = false;

// ---------------------------------------------------------------------------
// setup
// ---------------------------------------------------------------------------
void setup() {
    Serial.begin(115200);
    while (!Serial) {}   // wait for USB serial on Giga R1

    // Seed RNG from a floating ADC pin (unconnected = noise)
    randomSeed(analogRead(A0));

    // Fill input with random floats in [-1.0, 1.0]
    // Replace this block with real sensor data in your application.
    for (int i = 0; i < INPUT_SIZE; ++i) {
        // random(-1000, 1001) gives integers in [-1000, 1000]
        input_buf[i] = (float)random(-1000, 1001) / 1000.0f;
    }

    Serial.println("Input buffer filled with random data.");
    Serial.println("Running model_forward...");
    Serial.println();
}

// ---------------------------------------------------------------------------
// loop — runs inference once, then halts
// ---------------------------------------------------------------------------
void loop() {
    if (_done) return;
    _done = true;

    // model_forward prints profiling checkpoints (Serial.print) internally
    model_forward(input_buf, output_buf);

    // Print output class scores
    Serial.println();
    Serial.println("Output scores:");
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        Serial.print("  class ");
        Serial.print(i);
        Serial.print(": ");
        Serial.println(output_buf[i], 6);
    }

    // Find argmax
    int best = 0;
    for (int i = 1; i < OUTPUT_SIZE; ++i) {
        if (output_buf[i] > output_buf[best]) best = i;
    }
    Serial.print("Predicted class: ");
    Serial.println(best);
}
