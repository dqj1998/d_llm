#include "backprop.h"
#include "forward.h"
#include "matrix_ops.h"
#include "activations.h"

// Derivative of ReLU
void relu_derivative(Matrix *m, Matrix *grad) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            grad->data[i][j] = (m->data[i][j] > 0) ? 1.0 : 0.0;
        }
    }
}

void backpropagation(NeuralNetwork *nn, Matrix *input, Matrix *target, float learning_rate) {
    // Forward Pass
    // (Same as before, but we need to keep intermediate results for gradients)
    Matrix *hidden_input = matrix_multiply(input, nn->weights_input_hidden);
    Matrix *hidden_layer = matrix_add(hidden_input, nn->bias_hidden);
    relu(hidden_layer);

    Matrix *output_input = matrix_multiply(hidden_layer, nn->weights_hidden_output);
    Matrix *output_layer = matrix_add(output_input, nn->bias_output);
    softmax(output_layer);

    // Compute Loss Derivative (Cross-Entropy Loss)
    // For simplicity, we'll assume one-hot encoded targets
    Matrix *output_error = create_matrix(output_layer->rows, output_layer->cols);
    for (int i = 0; i < output_layer->rows; i++) {
        for (int j = 0; j < output_layer->cols; j++) {
            output_error->data[i][j] = output_layer->data[i][j] - target->data[i][j];
        }
    }

    // Backpropagate to Hidden Layer
    Matrix *weights_hidden_output_T = matrix_transpose(nn->weights_hidden_output);
    Matrix *hidden_error = matrix_multiply(output_error, weights_hidden_output_T);

    // Derivative of ReLU
    Matrix *relu_grad = create_matrix(hidden_layer->rows, hidden_layer->cols);
    relu_derivative(hidden_layer, relu_grad);

    for (int i = 0; i < hidden_error->rows; i++) {
        for (int j = 0; j < hidden_error->cols; j++) {
            hidden_error->data[i][j] *= relu_grad->data[i][j];
        }
    }

    // Update Weights and Biases
    // Hidden to Output
    Matrix *hidden_layer_T = matrix_transpose(hidden_layer);
    Matrix *delta_weights_hidden_output = matrix_multiply(hidden_layer_T, output_error);

    // Input to Hidden
    Matrix *input_T = matrix_transpose(input);
    Matrix *delta_weights_input_hidden = matrix_multiply(input_T, hidden_error);

    // Update weights with gradients
    for (int i = 0; i < nn->weights_hidden_output->rows; i++) {
        for (int j = 0; j < nn->weights_hidden_output->cols; j++) {
            nn->weights_hidden_output->data[i][j] -= learning_rate * delta_weights_hidden_output->data[i][j];
        }
    }

    for (int i = 0; i < nn->weights_input_hidden->rows; i++) {
        for (int j = 0; j < nn->weights_input_hidden->cols; j++) {
            nn->weights_input_hidden->data[i][j] -= learning_rate * delta_weights_input_hidden->data[i][j];
        }
    }

    // Update biases
    for (int i = 0; i < nn->bias_output->rows; i++) {
        for (int j = 0; j < nn->bias_output->cols; j++) {
            nn->bias_output->data[i][j] -= learning_rate * output_error->data[i][j];
        }
    }

    for (int i = 0; i < nn->bias_hidden->rows; i++) {
        for (int j = 0; j < nn->bias_hidden->cols; j++) {
            nn->bias_hidden->data[i][j] -= learning_rate * hidden_error->data[i][j];
        }
    }

    // Free allocated memory
    free_matrix(hidden_input);
    free_matrix(hidden_layer);
    free_matrix(output_input);
    free_matrix(output_layer);
    free_matrix(output_error);
    free_matrix(hidden_error);
    free_matrix(weights_hidden_output_T);
    free_matrix(hidden_layer_T);
    free_matrix(delta_weights_hidden_output);
    free_matrix(input_T);
    free_matrix(delta_weights_input_hidden);
    free_matrix(relu_grad);
}
