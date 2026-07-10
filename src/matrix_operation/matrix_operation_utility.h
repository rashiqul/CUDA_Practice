#ifndef MATRIX_OPERATION_UTILITY_H
#define MATRIX_OPERATION_UTILITY_H

#ifdef __cplusplus
extern "C" {
#endif

/* Fills n×n matrices A, B, and C with float data for the MAC computation.
 * A, B, C must already be allocated (n*n floats each).
 * Reads from CSV data files if present in the data/ subdirectory;
 * falls back to random float values otherwise.
 */
void getData(float *A, float *B, float *C, int n);

/* Compares output (n×n, already copied back to host) against result.raw.
 * Prints PASS/FAIL and the maximum absolute error.
 */
void verifyResult(float *output, int n);

#ifdef __cplusplus
}
#endif

#endif /* MATRIX_OPERATION_UTILITY_H */
