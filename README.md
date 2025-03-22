=== Build LLM from zero. ===

20241102
Started coding
The order of coding: matrix, matrix_ops, activations, neural_network, forward, backprop, train

For future: dealing with "For simplicity, we're not computing total loss here"

Compile NN:
gcc -g -o bin/main_nn main_nn.c trainers/nn/xor_dataset.c trainers/nn/and_dataset.c trainers/nn/train.c \
models/nn/backprop.c models/nn/forward.c models/nn/neural_network.c \
models/nn/activations.c models/nn/matrix.c models/nn/matrix_ops.c trainers/nn/loss.c -lm

Compile RNN:
gcc -g -o bin/main_rnn main_rnn.c \
trainers/nn/xor_dataset.c trainers/nn/and_dataset.c \
trainers/rnn/train_rnn.c \
models/nn/matrix.c  models/nn/matrix_ops.c models/nn/activations.c \
models/rnn/rnn.c -lm

20241103
the order of coding for rnn extantion: rnn, adding tanh_activation into activations, train_rnn

project/
├── main_nn.c                // Main file for neural network
├── main_rnn.c               // Main file for recurrent neural network
├── bin/                     // Compiled binaries directory
├── models/                  // Neural network model implementations
│   ├── nn/                 // Feed-forward neural network related
│   │   ├── matrix.h
│   │   ├── matrix.c
│   │   ├── matrix_ops.h
│   │   ├── matrix_ops.c
│   │   ├── activations.h
│   │   ├── activations.c
│   │   ├── neural_network.h
│   │   ├── neural_network.c
│   │   ├── forward.h
│   │   ├── forward.c
│   │   ├── backprop.h
│   │   └── backprop.c
│   └── rnn/                // Recurrent neural network related
│       ├── rnn.h
│       └── rnn.c
├── trainers/               // Training implementations
│   ├── nn/                // Neural network training
│   │   ├── train.h
│   │   ├── train.c
│   │   ├── loss.c
│   │   ├── xor_dataset.c
│   │   └── and_dataset.c
│   └── rnn/               // RNN training
│       ├── train_rnn.h
│       └── train_rnn.c
└── clients/               // Client applications


20250316 XOR got right by changing init to xavier_initialization in neural_network

20250322 Trained RNN, and got right results after changing sequence_length from 1 to 3
