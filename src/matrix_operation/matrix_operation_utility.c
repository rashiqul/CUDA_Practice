/*
 * matrix_operation_utility.c
 *
 * Implements getData(), which loads the n×n host matrices A, B, and C
 * from plain-text .raw files in the data/ subdirectory.
 *
 * File format: first line contains "rows cols" (both equal to n),
 * followed by n rows of n space-separated float values, row-major order.
 * Each file must contain exactly n rows of n floats after the header.
 *
 * Generate the data files with:  python3 data/generate_data.py
 */

#include "matrix_operation_utility.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* --------------------------------------------------------------------------
 * Internal helper
 * -------------------------------------------------------------------------- */

static void read_dat(const char *path, float *M, int n) {
    int total = n * n;
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "[getData] Cannot open '%s'.\n", path);
        exit(EXIT_FAILURE);
    }
    int rows, cols;
    if (fscanf(fp, "%d %d", &rows, &cols) != 2 || rows != n || cols != n) {
        fprintf(stderr, "[getData] '%s': expected header '%d %d'.\n", path, n, n);
        fclose(fp);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < total; i++) {
        if (fscanf(fp, "%f", &M[i]) != 1) {
            fprintf(stderr, "[getData] '%s': parse error at element %d/%d.\n", path, i, total);
            fclose(fp);
            exit(EXIT_FAILURE);
        }
    }
    fclose(fp);
}

/* --------------------------------------------------------------------------
 * Public API
 * -------------------------------------------------------------------------- */

void getData(float *A, float *B, float *C, int n) {
    read_dat("src/matrix_operation/data/A.raw", A, n);
    read_dat("src/matrix_operation/data/B.raw", B, n);
    read_dat("src/matrix_operation/data/C.raw", C, n);
}

void verifyResult(float *output, int n) {
    float *expected = (float *)malloc((size_t)n * n * sizeof(float));
    if (!expected) {
        fprintf(stderr, "[verifyResult] malloc failed.\n");
        return;
    }

    read_dat("src/matrix_operation/data/result.raw", expected, n);

    float max_err = 0.0f;
    int   max_idx = 0;
    int   total   = n * n;

    for (int i = 0; i < total; i++) {
        float err = fabsf(output[i] - expected[i]);
        if (err > max_err) {
            max_err = err;
            max_idx = i;
        }
    }

    if (max_err < 1e-3f) {
        printf("[verifyResult] PASS — max absolute error: %.6f\n", max_err);
    } else {
        int row = max_idx / n;
        int col = max_idx % n;
        printf("[verifyResult] FAIL — max absolute error: %.6f "
               "at output[%d][%d] (got %.6f, expected %.6f)\n",
               max_err, row, col, output[max_idx], expected[max_idx]);
    }

    free(expected);
}
