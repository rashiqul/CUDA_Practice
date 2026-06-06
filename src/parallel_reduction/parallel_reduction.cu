#include <stdio.h>
#include <stdlib.h>

// Problem: sum all N elements of array input[] into a single value.
//
// Pattern: each block reduces its chunk into one partial sum (written to
// output[blockIdx.x]). The host then sums the partial results.
//
// Key tools you will need:
//   __shared__ float sdata[BLOCK_SIZE];   -- fast on-chip shared memory
//   __syncthreads();                      -- barrier: wait for all threads in block
//   threadIdx.x, blockIdx.x, blockDim.x  -- your coordinates
//
// Naive reduction tree (one approach):
//   stride = 1
//   while stride < blockDim.x:
//       if threadIdx.x % (2*stride) == 0:
//           sdata[threadIdx.x] += sdata[threadIdx.x + stride]
//       __syncthreads()
//       stride *= 2
//   thread 0 holds the block's sum

#define BLOCK_SIZE 256
#define N (1 << 20)  // 1M elements

// TODO: implement the reduction kernel
__global__ void reduce(const float *input, float *output, int n) {

}

int main() {
    // --- host setup ---------------------------------------------------------
    float *h_input = (float *)malloc(N * sizeof(float));
    float h_result = 0.0f;

    // fill with 1.0f so the expected sum == N
    for (int i = 0; i < N; i++) h_input[i] = 1.0f;

    // --- device setup -------------------------------------------------------
    float *d_input, *d_output;
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMalloc(&d_input,  N    * sizeof(float));
    cudaMalloc(&d_output, grid * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // --- TODO: launch kernel ------------------------------------------------


    // --- copy partial sums back and finish on CPU ---------------------------
    float *h_output = (float *)malloc(grid * sizeof(float));
    cudaMemcpy(h_output, d_output, grid * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < grid; i++) h_result += h_output[i];

    printf("Sum: %.0f  (expected: %d)\n", h_result, N);

    // --- cleanup ------------------------------------------------------------
    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output);
    return 0;
}
