#include <math.h>
#include "activations.h"

// Apply ReLU activation function
void relu(Matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            if (m->data[i][j] < 0) {
                m->data[i][j] = 0;
            }
        }
    }
}

// Apply Softmax activation function
void softmax(Matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        float sum = 0.0;
        for (int j = 0; j < m->cols; j++) {
            m->data[i][j] = exp(m->data[i][j]);
            sum += m->data[i][j];
        }
        for (int j = 0; j < m->cols; j++) {
            m->data[i][j] /= sum;
        }
    }
}

//  Apply tanh activation function
void tanh_activation(Matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m->data[i][j] = tanh(m->data[i][j]);
        }        
    }     
}
