#ifndef BACKPROP_H
#define BACKPROP_H

#include "neural_network.h"
#include "matrix.h"

// Function declarations
void backpropagation(NeuralNetwork *nn, Matrix *input, Matrix *target, float learning_rate);

#endif
