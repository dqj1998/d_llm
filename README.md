=== Build LLM from zero. ===

20241102
Started coding
The order of coding: matrix, matrix_ops, activations, neural_network, forward, backprop, train


For future: dealing with ”For simplicity, we're not computing total loss here“

20241103
the order of coding for rnn extantion: rnn, adding tanh_activation into activations, train_rnn

'''
project/
├── matrix.h
├── matrix.c
├── matrix_ops.h
├── matrix_ops.c
├── activations.h
├── activations.c
├── neural_network.h      // 前馈神经网络相关
├── neural_network.c
├── forward.h
├── forward.c
├── backprop.h
├── backprop.c
├── train.h
├── train.c
├── rnn.h                 // 循环神经网络相关
├── rnn.c
├── train_rnn.h
├── train_rnn.c
├── main.c                // 可以有多个 main 文件，针对不同的模型
'''


20250316 XOR got right by changing init to xavier_initialization in neural_network

