#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

#define NUM_OF_ELEMENTS 192u // Number of elements in the array
#define K               6u   // Number of adjacent elements to sum
#define BLOCK_SIZE      32u  // Number of threads per block

void verifyResult(int *A, int *B, int *C, int *R)
{
    int num_groups = (NUM_OF_ELEMENTS + K - 1) / K; // Correct ceil(NUM_OF_ELEMENTS / K)
    
    /* Each output R[i] corresponds to a group of K elements */
    for (int i = 0; i < num_groups; i++) 
    {  
        int sum = 0;  // Reset sum for each group
        for (int j = 0; j < K; j++) 
        {
            int current_idx = (i * K) + j;  // Access K adjacent elements
            if (current_idx < NUM_OF_ELEMENTS) 
            {  
                /* Bounds check */
                sum += A[current_idx] + B[current_idx];
            }
        }
        /* Store the sum at R[i] */
        R[i] = sum;  
    }

    /* Verify the result */
    for (int i = 0; i < num_groups; i++) 
    {
        if (C[i] != R[i]) 
        {
            printf("❌ Mismatch at index %d: Expected %d, Got %d\n", i, R[i], C[i]);
            return;
        }
    }
    printf("✅ Test Passed! GPU and CPU results match.\n");
}

/* CUDA Kernel */
__global__ void vectorAdditionAdjacent(int *A, int *B, int *R, int N)
{
    int idx_out = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_in = idx_out * K;

    int sum = 0;
    for (int i = 0; i < K; i++)
    {
        int element_index = idx_in + i;
        if (element_index < N)  // Bounds check
        {
            sum += A[element_index] + B[element_index];
        }
    }

    /* Ensure valid output index */
    if (idx_out < (N + K - 1) / K)  
    {
        /* Store result sequentially in R */
        R[idx_out] = sum;  
    }
}

int main(void)
{
    /* Allocate host memory */
    int input_size  = NUM_OF_ELEMENTS * sizeof(int);
    /* Ensure correct memory allocation for R */
    int output_size = ((NUM_OF_ELEMENTS + K - 1) / K) * sizeof(int); 

    int *h_A, *h_B, *h_R, *h_C;
    int *d_A, *d_B, *d_R;

    h_A = (int *)malloc(input_size);
    h_B = (int *)malloc(input_size);
    h_R = (int *)malloc(output_size);
    h_C = (int *)malloc(output_size);

    /* Initialize host arrays */
    for (int i = 0; i < NUM_OF_ELEMENTS; i++)
    {
        h_A[i] = 1;
        h_B[i] = 2;
    }

    /* Allocate device memory */
    cudaMalloc((void **)&d_A, input_size);
    cudaMalloc((void **)&d_B, input_size);
    cudaMalloc((void **)&d_R, output_size);

    /* Copy data to device memory */
    cudaMemcpy(d_A, h_A, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, input_size, cudaMemcpyHostToDevice);

    /* Compute correct grid size */
    int num_groups = (NUM_OF_ELEMENTS + K - 1) / K;
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((num_groups + BLOCK_SIZE - 1) / BLOCK_SIZE);

    /* Launch kernel */
    vectorAdditionAdjacent<<<dimGrid, dimBlock>>>(d_A, d_B, d_R, NUM_OF_ELEMENTS);
    /* Ensure execution is complete before copying back */
    cudaDeviceSynchronize(); 

    /* Copy result back to host */
    cudaMemcpy(h_R, d_R, output_size, cudaMemcpyDeviceToHost);

    /* Print results */
    printf("Results (first 10 elements):\n");
    for (int i = 0; i < 10; i++) 
    {
        printf("R[%d] = %d\n", i, h_R[i]);
    }

    /* Verify results */
    verifyResult(h_A, h_B, h_R, h_C);

    /* Free memory */
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_R);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_R);

    return 0;
}
