// gcc -o bin/nn_main main_nn.c trainers/nn/xor_dataset.c trainers/nn/train.c models/nn/backprop.c models/nn/forward.c models/nn/neural_network.c models/nn/activations.c models/nn/matrix.c models/nn/matrix_ops.c trainers/nn/loss.c -lm

#include <stdio.h>
#include <stdlib.h>
#include "models/nn/neural_network.h"
#include "models/nn/forward.h"
#include "models/nn/matrix.h"
#include "trainers/nn/train.h"
#include "trainers/nn/xor_dataset.h"

int main() {
    // Create the dataset
    Matrix **inputs;
    Matrix **targets;
    int dataset_size;
    create_xor_dataset(&inputs, &targets, &dataset_size);

    // Define network architecture
    int input_size = 2;
    int hidden_size = 4; // Can be adjusted
    int output_size = 2; // Since we have two classes (0 and 1), one-hot encoded

    // Initialize neural network
    NeuralNetwork *nn = initialize_network(input_size, hidden_size, output_size);

    // Training parameters
    int epochs = 10000;
    float learning_rate = 0.01;

    // Train the neural network
    train(nn, inputs, targets, dataset_size, epochs, learning_rate);

    // Testing the trained neural network
    printf("\nTesting the trained neural network:\n");
    for (int i = 0; i < dataset_size; i++) {
        Matrix *output = forward_pass(nn, inputs[i]);
        printf("Input: [%.0f, %.0f], Predicted Output: [%.3f, %.3f], Target: [%.0f, %.0f]\n",
            inputs[i]->data[0][0], inputs[i]->data[0][1],
            output->data[0][0], output->data[0][1],
            targets[i]->data[0][0], targets[i]->data[0][1]);
        free_matrix(output);
    }

    // Free allocated memory
    for (int i = 0; i < dataset_size; i++) {
        free_matrix(inputs[i]);
        free_matrix(targets[i]);
    }
    free(inputs);
    free(targets);
    free_network(nn);

    return 0;
}
