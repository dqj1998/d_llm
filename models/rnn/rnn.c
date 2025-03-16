#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "rnn.h"
#include "matrix_ops.h"
#include "activations.h"

// Initialize the RNN with random weights and zero biases
RNN* initialize_rnn(int input_size, int hidden_size, int output_size) {
    RNN *rnn = (RNN*)malloc(sizeof(RNN));
    rnn->input_size = input_size;
    rnn->hidden_size = hidden_size;
    rnn->output_size = output_size;

    rnn->W_xh = create_matrix(input_size, hidden_size);
    rnn->W_hh = create_matrix(hidden_size, hidden_size);
    rnn->W_hy = create_matrix(hidden_size, output_size);

    rnn->b_h = create_matrix(1, hidden_size);
    rnn->b_y = create_matrix(1, output_size);

    // Initialize weights with small random values
    // For simplicity, we'll initialize weights to small random values between -0.1 and 0.1
    srand((unsigned int)time(NULL));
    for (int i = 0; i < input_size; i++) {
        for (int j = 0; j < hidden_size; j++) {
            rnn->W_xh->data[i][j] = ((float)rand() / RAND_MAX) * 0.2 - 0.1;
        }
    }
    for (int i = 0; i < hidden_size; i++) {
        for (int j = 0; j < hidden_size; j++) {
            rnn->W_hh->data[i][j] = ((float)rand() / RAND_MAX) * 0.2 - 0.1;
        }
    }
    for (int i = 0; i < hidden_size; i++) {
        for (int j = 0; j < output_size; j++) {
            rnn->W_hy->data[i][j] = ((float)rand() / RAND_MAX) * 0.2 - 0.1;
        }
    }

    // Initialize biases and previous hidden state to zero
    for (int i = 0; i < hidden_size; i++) {
        rnn->b_h->data[0][i] = 0.0;
    }
    for (int i = 0; i < output_size; i++) {
        rnn->b_y->data[0][i] = 0.0;
    }

    rnn->h_prev = create_matrix(1, hidden_size);
    for (int i = 0; i < hidden_size; i++) {
        rnn->h_prev->data[0][i] = 0.0;
    }

    return rnn;
}

// Free the RNN
void free_rnn(RNN *rnn) {
    free_matrix(rnn->W_xh);
    free_matrix(rnn->W_hh);
    free_matrix(rnn->W_hy);
    free_matrix(rnn->b_h);
    free_matrix(rnn->b_y);
    free_matrix(rnn->h_prev);
    free(rnn);
}

// Forward propagation through the RNN
Matrix* rnn_forward(RNN *rnn, Matrix **inputs, int sequence_length) {
    // Store hidden states and outputs for each time step
    Matrix **hiddens = (Matrix**)malloc(sizeof(Matrix*) * sequence_length);
    Matrix **outputs = (Matrix**)malloc(sizeof(Matrix*) * sequence_length);

    Matrix *h_t = rnn->h_prev;

    for (int t = 0; t < sequence_length; t++) {
        // Compute hidden state
        Matrix *W_xh_x_t = matrix_multiply(inputs[t], rnn->W_xh);
        Matrix *W_hh_h_t_prev = matrix_multiply(h_t, rnn->W_hh);
        Matrix *h_raw = matrix_add(matrix_add(W_xh_x_t, W_hh_h_t_prev), rnn->b_h);
        tanh_activation(h_raw);

        // Save hidden state
        h_t = h_raw;
        hiddens[t] = h_t;

        // Compute output
        Matrix *W_hy_h_t = matrix_multiply(h_t, rnn->W_hy);
        Matrix *y_t = matrix_add(W_hy_h_t, rnn->b_y);
        softmax(y_t);
        outputs[t] = y_t;

        // Free intermediate matrices
        free_matrix(W_xh_x_t);
        free_matrix(W_hh_h_t_prev);
        free_matrix(W_hy_h_t);
        // Do not free h_raw as it's stored in h_t and hiddens[t]
    }

    // Update previous hidden state
    for (int i = 0; i < rnn->hidden_size; i++) {
        rnn->h_prev->data[0][i] = h_t->data[0][i];
    }

    // For simplicity, return the outputs of the last time step
    Matrix *final_output = create_matrix(1, rnn->output_size);
    for (int i = 0; i < rnn->output_size; i++) {
        final_output->data[0][i] = outputs[sequence_length - 1]->data[0][i];
    }

    // Free stored hidden states and outputs except for the last one
    for (int t = 0; t < sequence_length - 1; t++) {
        free_matrix(hiddens[t]);
        free_matrix(outputs[t]);
    }
    free(hiddens);
    free(outputs);

    return final_output;
}

// Backward propagation through the RNN
void rnn_backward(RNN *rnn, Matrix **inputs, Matrix **targets, int sequence_length, float learning_rate) {
    // Initialize gradients
    Matrix *dW_xh = create_matrix(rnn->input_size, rnn->hidden_size);
    Matrix *dW_hh = create_matrix(rnn->hidden_size, rnn->hidden_size);
    Matrix *dW_hy = create_matrix(rnn->hidden_size, rnn->output_size);
    Matrix *db_h = create_matrix(1, rnn->hidden_size);
    Matrix *db_y = create_matrix(1, rnn->output_size);

    // Initialize hidden states and outputs
    Matrix **hiddens = (Matrix**)malloc(sizeof(Matrix*) * (sequence_length + 1));
    Matrix **outputs = (Matrix**)malloc(sizeof(Matrix*) * sequence_length);

    // Forward pass to store all hidden states and outputs
    hiddens[0] = create_matrix(1, rnn->hidden_size);
    for (int i = 0; i < rnn->hidden_size; i++) {
        hiddens[0]->data[0][i] = rnn->h_prev->data[0][i];
    }

    for (int t = 0; t < sequence_length; t++) {
        // Compute hidden state
        Matrix *W_xh_x_t = matrix_multiply(inputs[t], rnn->W_xh);
        Matrix *W_hh_h_t_prev = matrix_multiply(hiddens[t], rnn->W_hh);
        Matrix *h_raw = matrix_add(matrix_add(W_xh_x_t, W_hh_h_t_prev), rnn->b_h);
        tanh_activation(h_raw);

        hiddens[t + 1] = h_raw; // Store hidden state

        // Compute output
        Matrix *W_hy_h_t = matrix_multiply(h_raw, rnn->W_hy);
        Matrix *y_t = matrix_add(W_hy_h_t, rnn->b_y);
        softmax(y_t);
        outputs[t] = y_t;

        // Free intermediate matrices
        free_matrix(W_xh_x_t);
        free_matrix(W_hh_h_t_prev);
        free_matrix(W_hy_h_t);
    }

    // Initialize error terms
    Matrix *dh_next = create_matrix(1, rnn->hidden_size);
    for (int i = 0; i < rnn->hidden_size; i++) {
        dh_next->data[0][i] = 0.0;
    }

    // Backward pass
    for (int t = sequence_length - 1; t >= 0; t--) {
        // Output error
        Matrix *dy = create_matrix(1, rnn->output_size);
        for (int i = 0; i < rnn->output_size; i++) {
            dy->data[0][i] = outputs[t]->data[0][i] - targets[t]->data[0][i];
        }

        // Gradients for W_hy and b_y
        Matrix *h_t_T = matrix_transpose(hiddens[t + 1]);
        Matrix *dW_hy_t = matrix_multiply(h_t_T, dy);
        for (int i = 0; i < rnn->hidden_size; i++) {
            for (int j = 0; j < rnn->output_size; j++) {
                dW_hy->data[i][j] += dW_hy_t->data[i][j];
            }
        }
        for (int i = 0; i < rnn->output_size; i++) {
            db_y->data[0][i] += dy->data[0][i];
        }

        // Backpropagate into h
        Matrix *W_hy_T = matrix_transpose(rnn->W_hy);
        Matrix *dh = matrix_multiply(dy, W_hy_T);
        for (int i = 0; i < rnn->hidden_size; i++) {
            dh->data[0][i] += dh_next->data[0][i];
        }

        // Backprop through tanh nonlinearity
        for (int i = 0; i < rnn->hidden_size; i++) {
            dh->data[0][i] *= (1 - hiddens[t + 1]->data[0][i] * hiddens[t + 1]->data[0][i]);
        }

        // Gradients for W_xh and W_hh and b_h
        Matrix *x_t_T = matrix_transpose(inputs[t]);
        Matrix *dW_xh_t = matrix_multiply(x_t_T, dh);
        for (int i = 0; i < rnn->input_size; i++) {
            for (int j = 0; j < rnn->hidden_size; j++) {
                dW_xh->data[i][j] += dW_xh_t->data[i][j];
            }
        }

        Matrix *h_t_prev_T = matrix_transpose(hiddens[t]);
        Matrix *dW_hh_t = matrix_multiply(h_t_prev_T, dh);
        for (int i = 0; i < rnn->hidden_size; i++) {
            for (int j = 0; j < rnn->hidden_size; j++) {
                dW_hh->data[i][j] += dW_hh_t->data[i][j];
            }
        }

        for (int i = 0; i < rnn->hidden_size; i++) {
            db_h->data[0][i] += dh->data[0][i];
        }

        // Update dh_next
        Matrix *W_hh_T = matrix_transpose(rnn->W_hh);
        free_matrix(dh_next);
        dh_next = matrix_multiply(dh, W_hh_T);

        // Free intermediate matrices
        free_matrix(dy);
        free_matrix(h_t_T);
        free_matrix(dW_hy_t);
        free_matrix(W_hy_T);
        free_matrix(dh);
        free_matrix(x_t_T);
        free_matrix(dW_xh_t);
        free_matrix(h_t_prev_T);
        free_matrix(dW_hh_t);
        free_matrix(W_hh_T);
    }

    // Update weights and biases
    // W_xh
    for (int i = 0; i < rnn->input_size; i++) {
        for (int j = 0; j < rnn->hidden_size; j++) {
            rnn->W_xh->data[i][j] -= learning_rate * dW_xh->data[i][j];
        }
    }
    // W_hh
    for (int i = 0; i < rnn->hidden_size; i++) {
        for (int j = 0; j < rnn->hidden_size; j++) {
            rnn->W_hh->data[i][j] -= learning_rate * dW_hh->data[i][j];
        }
    }
    // W_hy
    for (int i = 0; i < rnn->hidden_size; i++) {
        for (int j = 0; j < rnn->output_size; j++) {
            rnn->W_hy->data[i][j] -= learning_rate * dW_hy->data[i][j];
        }
    }
    // b_h
    for (int i = 0; i < rnn->hidden_size; i++) {
        rnn->b_h->data[0][i] -= learning_rate * db_h->data[0][i];
    }
    // b_y
    for (int i = 0; i < rnn->output_size; i++) {
        rnn->b_y->data[0][i] -= learning_rate * db_y->data[0][i];
    }

    // Free allocated memory
    free_matrix(dW_xh);
    free_matrix(dW_hh);
    free_matrix(dW_hy);
    free_matrix(db_h);
    free_matrix(db_y);
    free_matrix(dh_next);

    // Free stored hidden states and outputs
    for (int t = 0; t <= sequence_length; t++) {
        free_matrix(hiddens[t]);
    }
    for (int t = 0; t < sequence_length; t++) {
        free_matrix(outputs[t]);
    }
    free(hiddens);
    free(outputs);
}
