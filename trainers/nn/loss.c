
#include <math.h>
#include "loss.h"

float compute_loss(Matrix *output, Matrix *target) {
    float loss = 0.0;
    for (int i = 0; i < output->rows; i++) {
        for (int j = 0; j < output->cols; j++) {
            // Add a small epsilon to prevent log(0)
            float predicted = output->data[i][j] + 1e-9;
            loss += -target->data[i][j] * log(predicted);
        }
    }
    return loss;
}
