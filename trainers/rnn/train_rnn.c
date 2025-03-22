#include <stdio.h>
#include <math.h>
#include "train_rnn.h"

void train_rnn(RNN *rnn, Matrix ***inputs, Matrix ***targets, int num_sequences, int sequence_length, int epochs, float learning_rate) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0;
        for (int i = 0; i < num_sequences; i++) {
            // Forward pass
            Matrix *output = rnn_forward(rnn, inputs[i], sequence_length);

            // Compute loss (cross-entropy)
            // For simplicity, we only compute the loss for the last output
            float loss = 0.0;
            for (int j = 0; j < output->cols; j++) {
                float predicted = output->data[0][j] + 1e-9; // Prevent log(0)
                loss += -targets[i][sequence_length - 1]->data[0][j] * log(predicted);
            }
            total_loss += loss;

            // Backward pass
            rnn_backward(rnn, inputs[i], targets[i], sequence_length, learning_rate);

            // Free output matrix
            free_matrix(output);
        }
        float average_loss = total_loss / num_sequences;
        printf("Epoch %d completed. Average Loss: %f\n", epoch + 1, average_loss);
    }
}
