#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

// Create a matrix with specified dimensions
Matrix* create_matrix(int rows, int cols) {
    Matrix *m = (Matrix*)malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
    m->data = (float**)malloc(rows * sizeof(float*));

    for (int i = 0; i < rows; i++) {
        m->data[i] = (float*)calloc(cols, sizeof(float));
    }
    return m;
}

// Free the allocated memory for a matrix
void free_matrix(Matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        free(m->data[i]);
    }
    free(m->data);
    free(m);
}

// Print the matrix (for debugging)
void print_matrix(Matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%f ", m->data[i][j]);
        }
        printf("\n");
    }
}

