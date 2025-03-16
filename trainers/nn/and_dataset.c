#include <stdlib.h>
#include "and_dataset.h"
#include "../../models/nn/matrix.h"

// Define input and target patterns for the AND gate (4 examples)
static const float input_patterns[4][2] = {
    {0, 0},   // Output: 0
    {0, 1},   // Output: 0
    {1, 0},   // Output: 0
    {1, 1}    // Output: 1
};

static const float target_patterns[4][2] = {
    {1, 0},   // Output: 0 (one-hot encoded)
    {1, 0},   // Output: 0
    {1, 0},   // Output: 0
    {0, 1}    // Output: 1
};

/**
 * Creates an AND dataset with input and target patterns.
 *
 * This function initializes the inputs and targets for a simple AND gate dataset.
 * Each example consists of a 1x2 input matrix and a 1x2 target matrix (one-hot encoded).
 *
 * @param inputs Pointer to store the array of input matrices.
 * @param targets Pointer to store the array of target matrices.
 * @param dataset_size Pointer to store the number of examples in the dataset.
 */
void create_and_dataset(Matrix ***inputs, Matrix ***targets, int *dataset_size) {
    *dataset_size = 4;  // Number of examples

    // Allocate memory for inputs and targets
    *inputs = (Matrix**)malloc(sizeof(Matrix*) * (*dataset_size));
    *targets = (Matrix**)malloc(sizeof(Matrix*) * (*dataset_size));

    // Initialize each example
    for (int i = 0; i < *dataset_size; ++i) {
        // Create input matrix (1 row, 2 columns)
        (*inputs)[i] = create_matrix(1, 2);
        (*inputs)[i]->data[0][0] = input_patterns[i][0];
        (*inputs)[i]->data[0][1] = input_patterns[i][1];

        // Create target matrix (1 row, 2 columns for one-hot encoding)
        (*targets)[i] = create_matrix(1, 2);
        (*targets)[i]->data[0][0] = target_patterns[i][0];
        (*targets)[i]->data[0][1] = target_patterns[i][1];
    }
}