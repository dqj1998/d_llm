#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "models/nn/neural_network.h"
#include "models/nn/forward.h"
#include "models/nn/matrix.h"
#include "trainers/nn/train.h"
#include "trainers/nn/xor_dataset.h"
#include "trainers/nn/and_dataset.h"
#include "trainers/nn/or_dataset.h"

// Enum to represent dataset types
typedef enum {
    DATASET_TYPE_AND,
    DATASET_TYPE_XOR,
    DATASET_TYPE_OR
} DatasetType;

void train_network(DatasetType dataset_type, NeuralNetwork *nn, int epochs, float learning_rate) {
    // Create the dataset based on type
    Matrix **inputs;
    Matrix **targets;
    int dataset_size;
    
    if (dataset_type == DATASET_TYPE_AND) {
        create_and_dataset(&inputs, &targets, &dataset_size);
    } else if (dataset_type == DATASET_TYPE_XOR) {
        create_xor_dataset(&inputs, &targets, &dataset_size);
    } else if (dataset_type == DATASET_TYPE_OR) {
        create_or_dataset(&inputs, &targets, &dataset_size);
    }

    // Train the neural network
    train(nn, inputs, targets, dataset_size, epochs, learning_rate);

    free(inputs);
    free(targets);
}

void infer_network(DatasetType dataset_type, NeuralNetwork *network) {
    // Create the dataset based on type for inference
    Matrix **inputs;
    Matrix **targets;
    int dataset_size;
    
    if (dataset_type == DATASET_TYPE_AND) {
        create_and_dataset(&inputs, &targets, &dataset_size);
    } else if (dataset_type == DATASET_TYPE_XOR) {
        create_xor_dataset(&inputs, &targets, &dataset_size);
    } else if (dataset_type == DATASET_TYPE_OR) {
        create_or_dataset(&inputs, &targets, &dataset_size);
    }

    // Show dataset_type in the output
    if (dataset_type == DATASET_TYPE_AND) {
        printf("\nTesting the AND neural network:\n");
    } else if (dataset_type == DATASET_TYPE_XOR) {
        printf("\nTesting the XOR neural network:\n");
    } else if (dataset_type == DATASET_TYPE_OR) {
        printf("\nTesting the OR neural network:\n");
    }

    for (int i = 0; i < dataset_size; i++) {
        // Assuming we have a loaded model here
        Matrix *output = forward_pass(network, inputs[i]);
        printf("Input: [%.0f, %.0f], Predicted Output: [%.3f, %.3f], Target: [%.0f, %.0f]\n",
            inputs[i]->data[0][0], inputs[i]->data[0][1],
            output->data[0][0], output->data[0][1],
            targets[i]->data[0][0], targets[i]->data[0][1]);

        free_matrix(output);
    }

    // Free allocated memory after inference
    free(inputs);
    free(targets);
}

int main(int argc, char *argv[]) {
    // Parse command-line arguments for dataset type
    DatasetType dataset_type = DATASET_TYPE_AND;
    if (argc > 1) {
        if (!strcmp(argv[1], "xor")) {
            dataset_type = DATASET_TYPE_XOR;
        } else if (!strcmp(argv[1], "or")) {
            dataset_type = DATASET_TYPE_OR; // Assuming OR dataset uses AND dataset type
        }
    }

    // Training parameters
    int epochs = 10000;
    float learning_rate = 0.01;

    // Define network architecture
    int input_size = 2;
    int hidden_size = 8; //8;//4; // Can be adjusted
    int output_size = 2;

    // Initialize neural network
    NeuralNetwork *nn = initialize_network(input_size, hidden_size, output_size);

    // Train the network first
    train_network(dataset_type, nn, epochs, learning_rate);  // Use all examples for training

    // Then perform inference
    infer_network(dataset_type, nn);
    
    // Free allocated memory after training
    free_network(nn);

    return 0;
}