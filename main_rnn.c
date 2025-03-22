#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "models/rnn/rnn.h"
#include "trainers/rnn/train_rnn.h"
#include "trainers/nn/and_dataset.h"
#include "trainers/nn/xor_dataset.h"

// Enum to represent dataset types
typedef enum {
    DATASET_TYPE_AND,
    DATASET_TYPE_XOR
} DatasetType;

void train_rnn_network(DatasetType dataset_type, RNN *rnn, int sequence_length, int epochs, float learning_rate) {
    // Create the dataset based on type
    Matrix **inputs;
    Matrix **targets;
    int dataset_size;

    if (dataset_type == DATASET_TYPE_AND) {
        create_and_dataset(&inputs, &targets, &dataset_size);
    } else if (dataset_type == DATASET_TYPE_XOR) {
        create_xor_dataset(&inputs, &targets, &dataset_size);
    }

    // Create sequence arrays for training
    Matrix ***sequence_inputs = (Matrix ***)malloc(sizeof(Matrix **) * dataset_size);
    Matrix ***sequence_targets = (Matrix ***)malloc(sizeof(Matrix **) * dataset_size);

    for (int i = 0; i < dataset_size; i++) {
        sequence_inputs[i] = (Matrix **)malloc(sizeof(Matrix *) * sequence_length);
        sequence_targets[i] = (Matrix **)malloc(sizeof(Matrix *) * sequence_length);
        
        // For AND/XOR, we use the same input and target for all time steps
        for (int t = 0; t < sequence_length; t++) {
            sequence_inputs[i][t] = inputs[i];
            sequence_targets[i][t] = targets[i];
        }
    }

    // Train the RNN
    train_rnn(rnn, sequence_inputs, sequence_targets, dataset_size, sequence_length, epochs, learning_rate);

    // Free allocated memory
    for (int i = 0; i < dataset_size; i++) {
        free(sequence_inputs[i]);
        free(sequence_targets[i]);
    }
    free(sequence_inputs);
    free(sequence_targets);

    for (int i = 0; i < dataset_size; i++) {
        free_matrix(inputs[i]);
        free_matrix(targets[i]);
    }
    free(inputs);
    free(targets);
}

void infer_rnn_network(DatasetType dataset_type, RNN *rnn) {
    // Create the dataset based on type for inference
    Matrix **inputs;
    Matrix **targets;
    int dataset_size;

    if (dataset_type == DATASET_TYPE_AND) {
        create_and_dataset(&inputs, &targets, &dataset_size);
    } else if (dataset_type == DATASET_TYPE_XOR) {
        create_xor_dataset(&inputs, &targets, &dataset_size);
    }

    // Show dataset_type in the output
    if (dataset_type == DATASET_TYPE_AND) {
        printf("\nTesting the AND RNN:\n");
    } else if (dataset_type == DATASET_TYPE_XOR) {
        printf("\nTesting the XOR RNN:\n");
    }

    for (int i = 0; i < dataset_size; i++) {
        // Create a single-element array for the input sequence
        Matrix **sequence = (Matrix **)malloc(sizeof(Matrix *));
        sequence[0] = inputs[i];
        
        // Perform forward pass
        Matrix *output = rnn_forward(rnn, sequence, 1); // Sequence length is 1 for AND/XOR
        printf("Input: [%.0f, %.0f], Predicted Output: [%.3f, %.3f], Target: [%.0f, %.0f]\n",
            inputs[i]->data[0][0], inputs[i]->data[0][1],
            output->data[0][0], output->data[0][1],
            targets[i]->data[0][0], targets[i]->data[0][1]);

        free(sequence);
        free_matrix(output);
    }

    // Free allocated memory after inference
    for (int i = 0; i < dataset_size; i++) {
        free_matrix(inputs[i]);
        free_matrix(targets[i]);
    }
    free(inputs);
    free(targets);
}

int main(int argc, char *argv[]) {
    // Parse command-line arguments for dataset type
    DatasetType dataset_type = DATASET_TYPE_AND;
    if (argc > 1) {
        if (!strcmp(argv[1], "xor")) {
            dataset_type = DATASET_TYPE_XOR;
        }
    }

    // Training parameters
    int sequence_length = 3; 
    int epochs = 10000;     
    float learning_rate = 0.01;

    // Define RNN architecture
    int input_size = 2;
    int hidden_size = 8;
    int output_size = 2;

    // Initialize RNN
    RNN *rnn = initialize_rnn(input_size, hidden_size, output_size);

    // Train the RNN
    train_rnn_network(dataset_type, rnn, sequence_length, epochs, learning_rate);

    // Perform inference
    infer_rnn_network(dataset_type, rnn);

    // Free allocated memory
    free_rnn(rnn);

    return 0;
}