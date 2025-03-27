#ifndef OR_DATASET_H
#define OR_DATASET_H

#include "../../models/nn/matrix.h"

/**
 * Creates an OR dataset with input and target patterns.
 *
 * This function initializes the inputs and targets for a simple OR gate dataset.
 * Each example consists of a 1x2 input matrix and a 1x2 target matrix (one-hot encoded).
 *
 * @param inputs Pointer to store the array of input matrices.
 * @param targets Pointer to store the array of target matrices.
 * @param dataset_size Pointer to store the number of examples in the dataset.
 */
void create_or_dataset(Matrix ***inputs, Matrix ***targets, int *dataset_size);

#endif // OR_DATASET_H