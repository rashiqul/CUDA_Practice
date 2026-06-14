#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define BLOCK_SIZE     1
#define NUM_OF_THREADS 1024

// A CUDA kernel that adds two vectors element-wise and stores the result in a third vector
__global__ void add_two_vectors(const int *d_A, const int *d_B, int *d_C)
{
    // Thread index to identify which element of the vectors to operate on
    int idx = threadIdx.x;

    // Perform the addition of the two vectors element-wise and store the result in the output vector
    d_C[idx] = d_A[idx] + d_B[idx];
}

// Function to allocate memory for the host vectors and populate them with sample data
static void allocate_and_populate_vector_cpu(int *&h_A, int *&h_B)
{
    // Allocate memory for the host vectors
    h_A = (int*)malloc(NUM_OF_THREADS * sizeof(int));
    h_B = (int*)malloc(NUM_OF_THREADS * sizeof(int));

    // Populate the host vectors with sample data
    for (int i = 0; i < NUM_OF_THREADS; i++)
    {
        h_A[i] = i;                  // Vector A will contain values from 0 to NUM_OF_THREADS-1
        h_B[i] = NUM_OF_THREADS - i; // Vector B will contain values from NUM_OF_THREADS to 1
    }
}

// Function to allocate memory for the device vectors and copy the data from the host vectors to the device vectors
static void allocate_and_populate_vector_gpu(const int *h_A, const int *h_B, int *&d_A, int *&d_B, int *&d_C)
{
    // Allocate memory for the device vectors (GPU)
    cudaMalloc((void**)&d_A, NUM_OF_THREADS * sizeof(int));
    cudaMalloc((void**)&d_B, NUM_OF_THREADS * sizeof(int));
    cudaMalloc((void**)&d_C, NUM_OF_THREADS * sizeof(int));

    // Copy the data from the host vectors to the device vectors
    cudaMemcpy(d_A, h_A, NUM_OF_THREADS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, NUM_OF_THREADS * sizeof(int), cudaMemcpyHostToDevice);
}

// Function to compare the results from the GPU with the expected results and print the output
static void compare_and_print_results(const int *h_A, const int *h_B, const int *h_C)
{
    // Compare the results from the GPU with the expected results and print the output
    for (int i = 0; i < NUM_OF_THREADS; i++)
    {
        int expected = h_A[i] + h_B[i]; // Calculate the expected result for
        if (h_C[i] != expected)
        {
            printf("Error at index %d: Expected %d, but got %d\n", i, expected, h_C[i]);
            return;
        }
    }
    printf("All results are correct!\n");
}

int main()
{
    printf("Adding two vectors on the GPU!\n");

    // Host vectors (CPU)
    int *h_A, *h_B, *h_C;
    // Device vectors (GPU)
    int *d_A, *d_B, *d_C; 

    // Allocate memory and populate the input vectors on the host (CPU)
    allocate_and_populate_vector_cpu(h_A, h_B);
    h_C = (int*)malloc(NUM_OF_THREADS * sizeof(int));

    // Allocate memory and populate the input vectors on the device (GPU)
    allocate_and_populate_vector_gpu(h_A, h_B, d_A, d_B, d_C);

    // Launch the kernel to add the two vectors on the GPU
    add_two_vectors<<<BLOCK_SIZE, NUM_OF_THREADS>>>(d_A, d_B, d_C);

    // Wait for the GPU to finish before exiting
    cudaDeviceSynchronize();

    // Copy the result from the device vector back to the host vector
    cudaMemcpy(h_C, d_C, NUM_OF_THREADS * sizeof(int), cudaMemcpyDeviceToHost);


    // Compare the results and print the output
    compare_and_print_results(h_A, h_B, h_C);

    // Free the allocated memory on the device (GPU)
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free the allocated memory on the host (CPU)
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}