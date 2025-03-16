// create a simple dataset of the XOR problem

#include <stdio.h>
#include <stdlib.h>
#include "xor_dataset.h"
#include "../../models/nn/matrix.h"

// Function to create the XOR dataset
void create_xor_dataset(Matrix ***inputs, Matrix ***targets, int *dataset_size) {
    *dataset_size = 4; // There are 4 possible input combinations
    *inputs = (Matrix**)malloc(sizeof(Matrix*) * (*dataset_size));
    *targets = (Matrix**)malloc(sizeof(Matrix*) * (*dataset_size));

    // Define the input and target data
    float input_data[4][2] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    float target_data[4][2] = {
        {1, 0}, // Output 0 represented as [1, 0]
        {0, 1}, // Output 1 represented as [0, 1]
        {0, 1}, // Output 1 represented as [0, 1]
        {1, 0}  // Output 0 represented as [1, 0]
    };

    for (int i = 0; i < *dataset_size; i++) {
        // Create input matrices
        (*inputs)[i] = create_matrix(1, 2); // 1 row, 2 columns
        (*inputs)[i]->data[0][0] = input_data[i][0];
        (*inputs)[i]->data[0][1] = input_data[i][1];

        // Create target matrices
        (*targets)[i] = create_matrix(1, 2); // 1 row, 2 columns (for one-hot encoding)
        (*targets)[i]->data[0][0] = target_data[i][0];
        (*targets)[i]->data[0][1] = target_data[i][1];
    }
}
