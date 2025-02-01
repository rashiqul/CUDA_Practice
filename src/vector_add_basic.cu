#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

/* Tolerance for floating-point comparisons */
#define TOLERANCE 1e-5

__global__ void vectorAddKernel(float* A_d, float* B_d, float* C_d, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        C_d[i] = A_d[i] + B_d[i];
    }
}

void vectorAdd(float *A, float *B, float *C, int n)
{
    float *A_d, *B_d, *C_d;
    int size = n * sizeof(float);

    /* Transfer A and B to device memory */
    cudaMalloc((void **)&A_d, size);
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&B_d, size);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    /* Allocate device memory for C_d */
    cudaMalloc((void **)&C_d, size);

    /* Kernel invocation code */
    vectorAddKernel<<<ceil(n/256.0), 256.0>>>(A_d, B_d, C_d, n);

    /* Transfer C from device to host */
    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    /* Free device memory foe A_d, B_d and C_d */
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

void verifyResult(float *A, float *B, float *C, int n)
{
    float expected = 0.0;

    for (int i = 0; i < n; i++)
    {
        expected = A[i] + B[i];
        if (fabs(C[i] - expected) > TOLERANCE)
        {
             printf("Error at index %d: GPU result %f, expected %f\n", i, C[i], expected);
            return;
        }
    }
    printf("Unit test passed! GPU result matches CPU calculation.\n");
}

int main(void)
{
    /* Size of large vectors, 1 million elements as an example */
    int n = 1000000; 
    
    /* Allocates memory, creates large vectors A and B */
    float *A = (float *)malloc(n * sizeof(float));
    float *B = (float *)malloc(n * sizeof(float));
    float *C = (float *)malloc(n * sizeof(float));

    /* Initialize vectors A and B with some values */
    for (int i = 0; i < n; i++)
    {
        A[i] = (float)i;        // Example: A[i] = i
        B[i] = (float)(i * 2);  // Example: B[i] = 2 * i
    }

    /* Call vector addition function */
    vectorAdd(A, B, C, n);

    /* Print first 10 elements of C for verification */
    for (int i = 0; i < 10; i++)
    {
        printf("C[%d] = %f\n", i, C[i]);
    }

    verifyResult(A, B, C, n);

    /* Free host memory */
    free(A);
    free(B);
    free(C);

    return 0;
}