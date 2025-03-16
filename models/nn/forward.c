#include "forward.h"
#include "matrix_ops.h"
#include "activations.h"

Matrix* forward_pass(NeuralNetwork *nn, Matrix *input) {
    // Input to Hidden Layer
    Matrix *hidden_input = matrix_multiply(input, nn->weights_input_hidden);
    Matrix *hidden_layer = matrix_add(hidden_input, nn->bias_hidden);
    relu(hidden_layer);

    free_matrix(hidden_input);

    // Hidden to Output Layer
    Matrix *output_input = matrix_multiply(hidden_layer, nn->weights_hidden_output);
    Matrix *output_layer = matrix_add(output_input, nn->bias_output);
    softmax(output_layer);

    free_matrix(output_input);
    free_matrix(hidden_layer);

    return output_layer;
}
