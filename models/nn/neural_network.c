#include <stdlib.h>
#include <math.h>
#include "neural_network.h"
#include "matrix_ops.h"

// 使用 Xavier 初始化方法初始化权重
void xavier_initialization(Matrix *matrix) {
    int rows = matrix->rows;
    int cols = matrix->cols;
    float scale = sqrt(2.0 / (rows + cols));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix->data[i][j] = ((float)rand() / RAND_MAX) * 2 * scale - scale;
        }
    }
}

// 使用 Xavier 初始化方法初始化神经网络
NeuralNetwork* initialize_network(int input_size, int hidden_size, int output_size) {
    NeuralNetwork *nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));

    nn->weights_input_hidden = create_matrix(input_size, hidden_size);
    nn->weights_hidden_output = create_matrix(hidden_size, output_size);
    nn->bias_hidden = create_matrix(1, hidden_size);
    nn->bias_output = create_matrix(1, output_size);

    // 使用 Xavier 初始化方法初始化权重
    xavier_initialization(nn->weights_input_hidden);
    xavier_initialization(nn->weights_hidden_output);

    // 将偏置初始化为零
    for (int i = 0; i < hidden_size; i++) {
        nn->bias_hidden->data[0][i] = 0.0;
    }
    for (int i = 0; i < output_size; i++) {
        nn->bias_output->data[0][i] = 0.0;
    }

    return nn;
}

void free_network(NeuralNetwork *nn) {
    free_matrix(nn->weights_input_hidden);
    free_matrix(nn->weights_hidden_output);
    free_matrix(nn->bias_hidden);
    free_matrix(nn->bias_output);
    free(nn);
}
