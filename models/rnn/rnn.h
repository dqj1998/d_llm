#ifndef RNN_H
#define RNN_H

#include "matrix.h"

// RNN structure definition
typedef struct {
    Matrix *W_xh; // Input to hidden weights
    Matrix *W_hh; // Hidden to hidden weights
    Matrix *W_hy; // Hidden to output weights
    Matrix *b_h;  // Hidden bias
    Matrix *b_y;  // Output bias
    Matrix *h_prev; // Previous hidden state
    int input_size;
    int hidden_size;
    int output_size;
} RNN;

// Function declarations
RNN* initialize_rnn(int input_size, int hidden_size, int output_size);
void free_rnn(RNN *rnn);
Matrix* rnn_forward(RNN *rnn, Matrix **inputs, int sequence_length);
void rnn_backward(RNN *rnn, Matrix **inputs, Matrix **targets, int sequence_length, float learning_rate);

#endif
