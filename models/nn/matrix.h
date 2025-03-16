#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    int rows;
    int cols;
    float **data;
} Matrix;

// Function declarations
Matrix* create_matrix(int rows, int cols);
void free_matrix(Matrix *m);
void print_matrix(Matrix *m);

#endif
