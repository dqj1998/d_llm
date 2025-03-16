#ifndef FORWARD_H
#define FORWARD_H

#include "neural_network.h"
#include "matrix.h"

// Function declarations
Matrix* forward_pass(NeuralNetwork *nn, Matrix *input);

#endif
