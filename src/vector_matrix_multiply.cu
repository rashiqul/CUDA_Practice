/*
 * This example uses tiling for m x n dimension for matrix multiplication.
 */

#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define TILE_WIDTH 2u    /* Width of the square tile, 2x2 */
#define EPSILON    1e-5  /* Error threshold for verification */

__global__ void MatrixMultiply(float *A, float *B, float *C, int M, int N, int K)
{
    /* Calculate the row element */
    int row = blockIdx.y *TILE_WIDTH + threadIdx.y;
    /* Calculate the column element */
    int col = blockIdx.x *TILE_WIDTH + threadIdx.x;

    if ((row < M) && (col < N))
    {
        float Pvalue = 0.0f;
        for (int i = 0; i < K; i++)
        {
            /* Dot product for x column */
            Pvalue += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = Pvalue;
    }
}

void cpuMatrixMultiply(float *A, float *B, float *C, int M, int K, int N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
            {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}


void verifyResult(float *C_gpu, float *C_cpu, int M, int N)
{
    int errors = 0;
    for (int i = 0; i < M * N; i++)
    {
        if (fabs(C_gpu[i] - C_cpu[i]) > EPSILON)
        {
            printf("Mismatch at index %d: GPU = %f, CPU = %f\n", i, C_gpu[i], C_cpu[i]);
            errors++;
        }
    }
    if (errors == 0)
    {
        printf("\n✅ GPU result matches CPU result!\n");
    }
    else
    {
        printf("\n❌ Verification failed! %d mismatches found.\n", errors);
    }
}

void initializeMatrix(float* matrix, int size)
{
    for (int i = 0; i < size; i++) 
    {
        matrix[i] = static_cast<float>(rand() % 10);
    }
}

void printMatrix(float *matrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main(void)
{
    /* Seed random number generator */
    srand(time(NULL));
    /* Random rows for A and C (5 to 14) */
    int M = rand() % 10 + 5;
    /* Random common dimension (5 to 14) */  
    int K = rand() % 10 + 5;
    /* Random columns for B and C (5 to 14) */  
    int N = rand() % 10 + 5;  

    printf("Randomized Matrix Dimensions: M = %d, K = %d, N = %d\n", M, K, N);

    /* Host arrays */
    float *A_h = (float *)malloc(M * K * sizeof(float));
    float *B_h = (float *)malloc(K * N * sizeof(float));
    float *C_h = (float *)malloc(M * N * sizeof(float));
    float *C_r = (float *)malloc(M * N * sizeof(float));

    /* Device arrays */
    float *A_d, *B_d, *C_d;

    /* Allocate host memory */
    initializeMatrix(A_h, M * K);
    initializeMatrix(B_h, K * N);

    /* Compute CPU reference result */
    cpuMatrixMultiply(A_h, B_h, C_r, M, N, K);

    /* Device arrays */
    cudaMalloc((void **)&A_d, M * K * sizeof(float));
    cudaMalloc((void **)&B_d, K * N * sizeof(float));
    cudaMalloc((void **)&C_d, M * N * sizeof(float));

    /* Copy data to device */
    cudaMemcpy(A_d, A_h, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, K * N * sizeof(float), cudaMemcpyHostToDevice);

    /* Kernel launch parameters */
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(ceil(1.0 * (M / TILE_WIDTH)), ceil(1.0 * (N / TILE_WIDTH)));
    MatrixMultiply<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, M, N, K);

    /* Copy data back to host */
    cudaMemcpy(C_h, C_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    /* Print the result */
    printf("\nGPU Computed Matrix C:\n");
    printMatrix(C_h, M, N);

    /* Verify GPU result */
    verifyResult(C_h, C_r, M, N);

    /* Free memory */
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    free(A_h);
    free(B_h);
    free(C_h);
    free(C_r);

    return 0;
}

