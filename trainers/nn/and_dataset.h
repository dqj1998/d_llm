#ifndef AND_DATASET_H
#define AND_DATASET_H

#include "../../models/nn/neural_network.h"
#include "../../models/nn/matrix.h"

/**
 * Creates an AND dataset with input and target patterns.
 *
 * @param inputs Pointer to store the input matrices for each example.
 * @param targets Pointer to store the target (label) matrices for each example.
 * @param dataset_size Pointer to store the number of examples in the dataset.
 */
void create_and_dataset(Matrix ***inputs, Matrix ***targets, int *dataset_size);

#endif  // AND_DATASET_H