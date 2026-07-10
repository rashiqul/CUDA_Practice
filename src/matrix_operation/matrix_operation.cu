#include <wb.h>
#include "matrix_operation_utility.h"

/* --------------------------------------------------------------------------
 * Problem constraints
 *   - n×n matrices, n divisible by 64
 *   - Block dimensions: (16, 32, 1)  →  BLOCK_X = 16, BLOCK_Y = 32
 *   - Each thread computes exactly 2 adjacent elements in the output row
 * -------------------------------------------------------------------------- */
#define N       512   /* N×N matrices; must be divisible by 64 */
#define BLOCK_X 16
#define BLOCK_Y 32

/* --------------------------------------------------------------------------
 * Kernel Function
 * -------------------------------------------------------------------------- */
__global__ void MAC(float *A, float *B, float *C, float *output, int n)
{
    __shared__ float tileM[BLOCK_Y][BLOCK_X];          
    __shared__ float tileN[BLOCK_Y][BLOCK_X * 2]; // Each thread computes 2 adjacent elements in the output row

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x * 2 + tx;

    float sum0 = 0.0f;  // Sum0 for the first output element
    float sum1 = 0.0f;  // Sum1 for the second output element

    int sharedDim = n/BLOCK_X;  // Number of tiles in the shared dimension (divisible by BLOCK_X)

    for (int m = 0; m < sharedDim; m++)
    {
        // Bound limits of A
        if ((row < n) && (m * BLOCK_X + tx < n))
        {
            tileM[ty][tx] = A[row * n + m * BLOCK_X + tx];
        }
        else
        {
            tileM[ty][tx] = 0.0f;
        }
        // Bound limits of B for first column
        if ((m * BLOCK_X + ty < n) && (col < n))
        {
            tileN[ty][tx] = B[(m * BLOCK_X + ty) * n + col];
        }
        else
        {
            tileN[ty][tx] = 0.0f;
        }
        // Bound limits of B for second column
        if ((m * BLOCK_X + ty < n) && (col + BLOCK_X < n))
        {
            tileN[ty][tx + BLOCK_X] = B[(m * BLOCK_X + ty) * n + col + BLOCK_X];
        }
        else
        {
            tileN[ty][tx + BLOCK_X] = 0.0f;
        }

        // Synchronize to make sure the tiles are loaded
        __syncthreads();

        // Iterate over the tiles to compute the result of A * B
        for (int k = 0; k < BLOCK_X; k++)
        {
            sum0 += tileM[ty][k] * tileN[k][tx];
            sum1 += tileM[ty][k] * tileN[k][tx + BLOCK_X];
        }

        // Synchronize to make sure the results are computed before loading new tiles
        __syncthreads();
    }

    if (row < n && col < n)
    {
        output[row * n + col] = sum0 + C[row * n + col];
    }
    if (row < n && col + BLOCK_X < n)
    {
        output[row * n + col + BLOCK_X] = sum1 + C[row * n + col + BLOCK_X];
    }   
}

/* --------------------------------------------------------------------------
 * Host Function
 * -------------------------------------------------------------------------- */
int main(int argc, char **argv) {
    wbArg_t args = wbArg_read(argc, argv);

    /* ----- Host allocations (matches the exact structure from the spec) -- */
    float *A, *B, *C, *output;
    A      = (float *)malloc(N * N * sizeof(float));
    B      = (float *)malloc(N * N * sizeof(float));
    C      = (float *)malloc(N * N * sizeof(float));
    output = (float *)malloc(N * N * sizeof(float));
    getData(A, B, C, N);  /* fills in data for matrixes A, B, and C */

    wbLog(TRACE, "Matrices A, B, C loaded — size ", N, "×", N);

    /* ----- Device pointers ------------------------------------------------ */
    float *d_A, *d_B, *d_C, *d_output;

    cudaMalloc((void **)&d_A, N * N * sizeof(float));
    cudaMalloc((void **)&d_B, N * N * sizeof(float));
    cudaMalloc((void **)&d_C, N * N * sizeof(float));
    cudaMalloc((void **)&d_output, N * N * sizeof(float));

    /* ----- Copy data from host to device ---------------------------------- */
    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, N * N * sizeof(float), cudaMemcpyHostToDevice);

    /* ----- Kernel launch -------------------------------------------------- */
    // gridDim.x uses BLOCK_X*2 because each block covers 2 output columns per thread
    dim3 blockDim(BLOCK_X, BLOCK_Y, 1);
    dim3 gridDim(ceil((float)N / (BLOCK_X * 2)), ceil((float)N / BLOCK_Y), 1);

    MAC<<<gridDim, blockDim>>>(d_A, d_B, d_C, d_output, N);

    cudaDeviceSynchronize();

    /* ----- Copy data from device to host ---------------------------------- */
    cudaMemcpy(output, d_output, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    verifyResult(output, N);


    /* ----- Free device memory -------------------------------------------- */
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_output);

    /* ----- Host cleanup --------------------------------------------------- */
    free(A);
    free(B);
    free(C);
    free(output);

    (void)args;   /* suppress unused-variable warning until args is used */
    return 0;
}
