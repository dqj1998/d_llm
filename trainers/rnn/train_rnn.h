#ifndef TRAIN_RNN_H
#define TRAIN_RNN_H

#include "../../models/rnn/rnn.h"
#include "../../models/nn/matrix.h"

// Function declarations
void train_rnn(RNN *rnn, Matrix ***inputs, Matrix ***targets, int num_sequences, int sequence_length, int epochs, float learning_rate);

#endif
