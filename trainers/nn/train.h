#ifndef TRAIN_H
#define TRAIN_H

#include "../../models/nn/neural_network.h"
#include "../../models/nn/matrix.h"

// Function declarations
void train(NeuralNetwork *nn, Matrix **inputs, Matrix **targets, int dataset_size, int epochs, float learning_rate);

#endif
