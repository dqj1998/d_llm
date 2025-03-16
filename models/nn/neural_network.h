#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "matrix.h"

typedef struct {
    Matrix *weights_input_hidden;
    Matrix *weights_hidden_output;
    Matrix *bias_hidden;
    Matrix *bias_output;
} NeuralNetwork;

// Function declarations
NeuralNetwork* initialize_network(int input_size, int hidden_size, int output_size);
void free_network(NeuralNetwork *nn);

#endif
