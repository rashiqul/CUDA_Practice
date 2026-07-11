#include <wb.h>
#include "convolution_verify.h"

#define BLOCK_SIZE 16
#define MASK_WIDTH 5
#define TILE_SIZE (BLOCK_SIZE + MASK_WIDTH - 1)

// Defined in convolution_2D_basic_kernel.cu
extern __constant__ float M[MASK_WIDTH * MASK_WIDTH];

__global__ void convolution_2D_tiled_kernel(float *N, float *P, int mask_width, int num_rows, int num_cols)
{
    __shared__ float tile_N[TILE_SIZE][TILE_SIZE];

    // Each thread loads exactly one element of the tile (including halo)
    int in_row = blockIdx.y * BLOCK_SIZE - (mask_width / 2) + threadIdx.y;
    int in_col = blockIdx.x * BLOCK_SIZE - (mask_width / 2) + threadIdx.x;

    if (in_row >= 0 && in_row < num_rows && in_col >= 0 && in_col < num_cols)
        tile_N[threadIdx.y][threadIdx.x] = N[in_row * num_cols + in_col];
    else
        tile_N[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    // Only the inner BLOCK_SIZE x BLOCK_SIZE threads compute output
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (threadIdx.y < BLOCK_SIZE && threadIdx.x < BLOCK_SIZE)
    {
        float p_value = 0.0f;

        for (int i = 0; i < mask_width; i++)
        {
            for (int j = 0; j < mask_width; j++)
            {
                p_value += tile_N[threadIdx.y + i][threadIdx.x + j] * M[i * mask_width + j];
            }
        }

        if (row < num_rows && col < num_cols)
        {
            P[row * num_cols + col] = p_value;
        }
    }
}

void setup_for_2D_conv_tiled(void)
{
    float *hostInput;
    float *hostMask;
    float *hostOutput;

    float *deviceInput;
    float *deviceOutput;

    int inputLength;
    int maskWidth;

    int numRows;
    int numCols;

    float *rawInput;

    // Import raw buffers
    rawInput = (float *)wbImport(DATA_DIRECTORY_2D "/input.dat", &inputLength);
    hostMask  = (float *)wbImport(DATA_DIRECTORY_2D "/kernel.dat", &maskWidth);

    // Extract dimensions from the two header floats
    numRows = (int)rawInput[0];
    numCols = (int)rawInput[1];
    wbLog(TRACE, "2D tiled input size: ", numRows, "x", numCols);

    hostInput = rawInput + 2;

    // Allocate memory for CPU
    hostOutput = (float *)malloc(numRows * numCols * sizeof(float));

    // Allocate memory for GPU
    cudaMalloc((void**)&deviceInput, numRows * numCols * sizeof(float));
    cudaMalloc((void**)&deviceOutput, numRows * numCols * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(deviceInput, hostInput, numRows * numCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(M, hostMask, maskWidth * sizeof(float), 0, cudaMemcpyHostToDevice);

    // TILE_SIZE x TILE_SIZE threads: each thread loads exactly one tile element
    dim3 blockDim(TILE_SIZE, TILE_SIZE, 1);
    dim3 gridDim(ceil((float)numCols / BLOCK_SIZE), ceil((float)numRows / BLOCK_SIZE), 1);

    // Launch the kernel
    convolution_2D_tiled_kernel<<<gridDim, blockDim>>>(deviceInput, deviceOutput, MASK_WIDTH, numRows, numCols);

    cudaDeviceSynchronize();

    // Copy result from device to host
    cudaMemcpy(hostOutput, deviceOutput, numRows * numCols * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify result against expected output
    verify_2D_tiled_conv(hostOutput, numRows, numCols);

    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    free(rawInput);
    free(hostMask);
    free(hostOutput);
}
