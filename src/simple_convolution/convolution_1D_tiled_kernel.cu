#include <wb.h>
#include "convolution_verify.h"

#define MASK_WIDTH (5)
#define TILE_WIDTH (16)
#define BLOCK_SIZE (TILE_WIDTH + MASK_WIDTH - 1)

__constant__ float mask_1D_tiled[MASK_WIDTH];

__global__ void convolution_1D_tiled_kernel(float *input, float *output, int width)
{
    __shared__ float tile[BLOCK_SIZE]; // Shared memory tile for input elements

    int tx = threadIdx.x;
    
    int index_o = blockIdx.x * TILE_WIDTH + tx;
    int index_i = index_o - MASK_WIDTH / 2;

    float pvalue = 0.0f;

    // Load input elements into shared memory
    if (index_i >= 0 && index_i < width) 
    {
        tile[tx] = input[index_i];
    }
    else
    {
        tile[tx] = 0.0f; // Ghost cell for out-of-bound elements
    }

    __syncthreads(); // Ensure all threads have loaded their input elements into shared memory

    // Perform convolution using the shared memory tile
    if (tx < TILE_WIDTH)
    {
        for (int j = 0; j < MASK_WIDTH; ++j)
        {
            pvalue += tile[tx + j] * mask_1D_tiled[j];
        }
        if (index_o < width)
        {
            output[index_o] = pvalue;
        }
    }

}

void setup_for_1D_tiled_conv(void)
{
    int inputLength;
    
    float *hostInput;
    float *hostOutput;
    float *hostMask;
    float *deviceInput;
    float *deviceOutput;

    // Load input matrix
    hostInput = (float *)wbImport(DATA_DIRECTORY_1D "/input.dat", &inputLength);
    wbLog(TRACE, "1D input width: ", inputLength);

    // Load the mask matrix
    hostMask = (float *)wbImport(DATA_DIRECTORY_1D "/kernel.dat", NULL);
    wbLog(TRACE, "1D mask width: ", MASK_WIDTH);

    // Allocate memory for CPU
    // CPU Output
    hostOutput = (float *)malloc(inputLength * sizeof(float));

    // Allocate memory for GPU
    // GPU Inputs
    cudaMalloc((void**)&deviceInput, inputLength * sizeof(float));
    // GPU Output
    cudaMalloc((void**)&deviceOutput, inputLength * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(float), cudaMemcpyHostToDevice);
    
    // Copy the mask to constant memory on the device
    cudaMemcpyToSymbol(mask_1D_tiled, hostMask, MASK_WIDTH * sizeof(float));

    // Define block and grid dimensions
    dim3 blockDim(BLOCK_SIZE, 1, 1);
    dim3 gridDim(ceil(inputLength/(1.0 * TILE_WIDTH)), 1, 1);

    // Launch the kernel
    convolution_1D_tiled_kernel<<<gridDim, blockDim>>>(deviceInput, deviceOutput, inputLength);

    // Synchronize to ensure kernel execution is complete
    cudaDeviceSynchronize();

    // Copy result from device to host
    cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify result against expected output
    verify_1D_tiled_conv(hostOutput, inputLength);

    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    free(hostInput);
    free(hostOutput);
    free(hostMask);
}