#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define  BLOCK_SIZE     1
#define  NUM_OF_THREADS 128
#define  WARP_SIZE      32

// A simple CUDA kernel that prints hello world from the GPU
__global__ void print_hello_world()
{
    printf("Hello, World from GPU!\n");
}

// A simple CUDA kernel that prints the block and thread IDs to demonstrate parallel execution
__global__ void print_block_thread_info()
{
    // Print the block and thread IDs to demonstrate parallel execution
    printf("The block ID is (%d, %d, %d). The thread ID is (%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
}

// A simple CUDA kernel that prints the warp information to demonstrate warp-level execution
__global__ void print_warp_info()
{
    // A warp is a group of 32 threads that execute the same instruction simultaneously.
    int warp_id = threadIdx.x / WARP_SIZE;
    
    // Print the warp information to demonstrate warp-level execution
    printf("Thread ID: %d, Block ID: %d, Warp ID: %d\n", threadIdx.x, blockIdx.x, warp_id);
}

int main()
{
    printf("Hello, World from CPU!\n");
    
    // Launch the kernel with the defined block size and number of threads
    // Kernel launch syntax: kernel<<<numBlocks, threadsPerBlock>>>(arguments);

    // Print hello world from the GPU using the kernel, using 1 block and 1 thread since we only want to print once
    print_hello_world<<<1, 1>>>();
    
    // Wait for the GPU to finish before proceeding
    cudaDeviceSynchronize();

    // Print new line for better readability
    printf("\n");

    // Print the block and thread information using the kernel
    print_block_thread_info<<<BLOCK_SIZE, NUM_OF_THREADS>>>();

    // Wait for the GPU to finish before proceeding
    cudaDeviceSynchronize();

    // Print new line for better readability
    printf("\n");
    
    // Print the warp information using the kernel
    print_warp_info<<<BLOCK_SIZE, NUM_OF_THREADS>>>();

    // Wait for the GPU to finish before exiting
    cudaDeviceSynchronize();


    return 0;
}