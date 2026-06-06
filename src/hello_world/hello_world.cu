#include <stdio.h>

// Each thread prints its block and thread index.
// __global__ marks this as a kernel — runs on the GPU, called from the CPU.
__global__ void hello_world() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    // Launch configuration: <<<blocks, threads_per_block>>>
    //   - 2 blocks
    //   - 4 threads per block
    //   - 8 total threads (2 x 4)
    //
    // Each thread gets a unique (blockIdx.x, threadIdx.x) pair:
    //   block 0: threads 0-3
    //   block 1: threads 0-3
    hello_world<<<2, 4>>>();

    // Wait for all GPU threads to finish before the program exits.
    cudaDeviceSynchronize();
    return 0;
}
