#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define BLOCK_SIZE     2
#define NUM_OF_THREADS 32

// A simple CUDA kernel that prints the block and thread IDs to demonstrate parallel execution
__global__ void print_block_thread_info()
{
    // Print the block and thread IDs to demonstrate parallel execution
    printf("The block ID is (%d, %d, %d). The thread ID is (%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
}

int main()
{
    printf("Hello, World from CPU!\n");
    
    // Launch the kernel with the defined block size and number of threads
    // Kernel launch syntax: kernel<<<numBlocks, threadsPerBlock>>>(arguments);
    print_block_thread_info<<<BLOCK_SIZE, NUM_OF_THREADS>>>();

    // Wait for the GPU to finish before exiting
    cudaDeviceSynchronize();

    return 0;
}