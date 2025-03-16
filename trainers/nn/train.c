#include <stdio.h>
#include "train.h"
#include "../../models/nn/backprop.h"
#include "../../models/nn/forward.h"
#include "loss.h"

void train(NeuralNetwork *nn, Matrix **inputs, Matrix **targets, int dataset_size, int epochs, float learning_rate) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0;
        for (int i = 0; i < dataset_size; i++) {
            // Forward pass
            Matrix *output = forward_pass(nn, inputs[i]);

            // Compute loss
            float loss = compute_loss(output, targets[i]);
            total_loss += loss;

            // Backward pass
            backpropagation(nn, inputs[i], targets[i], learning_rate);

            // Free output matrix
            free_matrix(output);
        }
        float average_loss = total_loss / dataset_size;
        if ((epoch + 1) % 100 == 0) {
            printf("Epoch %d/%d, Average Loss: %f\n", epoch + 1, epochs, average_loss);
        }
    }
}
