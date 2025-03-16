#include <stdio.h>
#include <stdlib.h>
#include "matrix_ops.h"

// Multiply two matrices
Matrix* matrix_multiply(Matrix *a, Matrix *b) {
    if (a->cols != b->rows) {
        fprintf(stderr, "Error: Incompatible dimensions for matrix multiplication\n");
        exit(EXIT_FAILURE);
    }
    Matrix *result = create_matrix(a->rows, b->cols);

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            float sum = 0.0;
            for (int k = 0; k < a->cols; k++) {
                sum += a->data[i][k] * b->data[k][j];
            }
            result->data[i][j] = sum;
        }
    }
    return result;
}

// Add two matrices
Matrix* matrix_add(Matrix *a, Matrix *b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "Error: Incompatible dimensions for matrix addition\n");
        exit(EXIT_FAILURE);
    }
    Matrix *result = create_matrix(a->rows, a->cols);

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            result->data[i][j] = a->data[i][j] + b->data[i][j];
        }
    }
    return result;
}

// Transpose a matrix
Matrix* matrix_transpose(Matrix *m) {
    Matrix *result = create_matrix(m->cols, m->rows);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            result->data[j][i] = m->data[i][j];
        }
    }
    return result;
}
