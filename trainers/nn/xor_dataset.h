#ifndef DATASET_H
#define DATASET_H

#include "../../models/nn/neural_network.h"
#include "../../models/nn/matrix.h"

// Function declarations
void create_xor_dataset(Matrix ***inputs, Matrix ***targets, int *dataset_size);

#endif
