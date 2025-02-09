/*
 * This example uses tiling with two 7x7 grids for the matrix multiplication.
 */

#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define NUM_OF_ELEMENTS 196u  /* Number of elements in the array  */
#define TILE_WIDTH        7u  /* Width of the square tile, 7x7    */
#define EPSILON         1e-5  /* Error threshold for verification */ 

__global__ void MatrixMultiply(float *A, float *B, float *C)
{
    /* Calculate the row index of the matrices  */
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    /* Calculate the column index of the matrices */
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < TILE_WIDTH) && (col < TILE_WIDTH))
    {
        float Pvalue = 0.0f;
        /* Each thread computes one element of the block sub-matrix */
        for (int i = 0; i < TILE_WIDTH; i++)
        {
            Pvalue += A[row * TILE_WIDTH + i] * B[i * TILE_WIDTH + col];
        }
        C[row * TILE_WIDTH + col] = Pvalue;
    }
}

void cpuMatrixMultiply(float *A, float *B, float *C)
{
    for (int i = 0; i < TILE_WIDTH; i++)
    {
        for (int j = 0; j < TILE_WIDTH; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < TILE_WIDTH; k++)
            {
                sum += A[i * TILE_WIDTH + k] * B[k * TILE_WIDTH + j];
            }
            C[i * TILE_WIDTH + j] = sum;
        }
    }
}


void verifyResult(float *C_gpu, float *C_cpu)
{
    int errors = 0;
    for (int i = 0; i < NUM_OF_ELEMENTS; i++)
    {
        if (fabs(C_gpu[i] - C_cpu[i]) > EPSILON)
        {
            printf("Mismatch at index %d: GPU = %f, CPU = %f\n", i, C_gpu[i], C_cpu[i]);
            errors++;
        }
    }
    if (errors == 0)
        printf("\n✅ GPU result matches CPU result!\n");
    else
        printf("\n❌ Verification failed! %d mismatches found.\n", errors);
}

void initializeMatrix(float* matrix)
{
    for (int i = 0; i < (NUM_OF_ELEMENTS); i++) 
    {
        matrix[i] = static_cast<float>(rand() % 10);
    }
}

void printMatrix(float *matrix)
{
    for (int i = 0; i < TILE_WIDTH; i++)
    {
        for (int j = 0; j < TILE_WIDTH; j++)
        {
            printf("%f ", matrix[i * TILE_WIDTH + j]);
        }
        printf("\n");
    }
}

int main(void)
{
    /* Host Arrays */
    float *A_h, *B_h, *C_h, *C_r;

    /* Device Arrays */
    float *A_d, *B_d, *C_d;

    /* Allocate host memory */
    A_h = (float*)malloc(NUM_OF_ELEMENTS * sizeof(float));
    B_h = (float*)malloc(NUM_OF_ELEMENTS * sizeof(float));
    C_h = (float*)malloc(NUM_OF_ELEMENTS * sizeof(float));
    C_r = (float*)malloc(NUM_OF_ELEMENTS * sizeof(float));

    /* Check if memory allocation was successful */
    if (A_h == NULL || B_h == NULL || C_h == NULL || C_r == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }

    /* Initialize the matrixes */
    initializeMatrix(A_h);
    initializeMatrix(B_h);

    /* Allocate device memory */
    cudaMalloc((void**)&A_d, NUM_OF_ELEMENTS * sizeof(float));
    cudaMalloc((void**)&B_d, NUM_OF_ELEMENTS * sizeof(float));
    cudaMalloc((void**)&C_d, NUM_OF_ELEMENTS * sizeof(float));

    /* Check if memory allocation was successful */
    if (A_d == NULL || B_d == NULL || C_d == NULL) {
        fprintf(stderr, "Failed to allocate device memory\n");
        return -1;
    }

    /* Copy data from the host memory to the device */
    cudaMemcpy(A_d, A_h, NUM_OF_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, NUM_OF_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);

    /* Compute CPU reference result */
    cpuMatrixMultiply(A_h, B_h, C_r);

    /* Define the thread block and grid size */
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    /* One block needed for 14 * 14 matrix */
    dim3 dimGrid(ceil(1.0 * (TILE_WIDTH / TILE_WIDTH)), ceil(1.0 * (TILE_WIDTH / TILE_WIDTH)));

    /* Launch the kernel */
    MatrixMultiply<<<dimGrid, dimBlock>>>(A_d, B_d, C_d);

    /* Copy the result from the device to the host */
    cudaMemcpy(C_h, C_d, NUM_OF_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost);

    /* Print the result */
    printf("\nGPU Computed Matrix C:\n");
    printMatrix(C_h);

    /* Verify the result */
    verifyResult(C_h, C_r);

    /* Free the device memory */
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    /* Free the host memory */
    free(A_h);
    free(B_h);
    free(C_h);

    return 0;
}
