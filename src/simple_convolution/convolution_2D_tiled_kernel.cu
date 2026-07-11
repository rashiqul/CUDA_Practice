#include <wb.h>
#include "convolution_verify.h"

#define MASK_WIDTH (5)
#define TILE_WIDTH (12)
#define BLOCK_SIZE (TILE_WIDTH + MASK_WIDTH - 1)

__constant__ float mask_2D_tiled[MASK_WIDTH][MASK_WIDTH];

__global__ void convolution_2D_tiled_kernel(float *N, float *P, int num_rows, int num_cols)
{
    // Strategy: Each thread loads exactly one element of the tile (parallelizing tile loading)
    
    __shared__ float tile_N[BLOCK_SIZE][BLOCK_SIZE];

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    
    int row_out = blockIdx.y * TILE_WIDTH + ty;
    int col_out = blockIdx.x * TILE_WIDTH + tx;

    int row_in = row_out - (MASK_WIDTH / 2);
    int col_in = col_out - (MASK_WIDTH / 2);

    float pvalue = 0.0f;

    if (row_in >= 0 && row_in < num_rows && col_in >= 0 && col_in < num_cols)
    {
        tile_N[ty][tx] = N[row_in * num_cols + col_in];
    }
    else
    {
        tile_N[ty][tx] = 0.0f;
    }

    __syncthreads();

    if (ty < TILE_WIDTH && tx < TILE_WIDTH)
    {
        for (int i = 0; i < MASK_WIDTH; i++) // row
        {
            for (int j = 0; j < MASK_WIDTH; j++) // column
            {
                pvalue += tile_N[ty + i][tx + j] * mask_2D_tiled[i][j];
            }
        }

        if (row_out < num_rows && col_out < num_cols)
        {
            P[row_out * num_cols + col_out] = pvalue;
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

    int numRows;
    int numCols;

    float *rawInput;

    // Import raw buffers
    rawInput = (float *)wbImport(DATA_DIRECTORY_2D "/input.dat", NULL);
    hostMask  = (float *)wbImport(DATA_DIRECTORY_2D "/kernel.dat", NULL);

    // Extract dimensions from the two header floats
    numRows = (int)rawInput[0];
    numCols = (int)rawInput[1];
    wbLog(TRACE, "2D tiled input size: ", numRows, "x", numCols);
    wbLog(TRACE, "2D tiled mask width: ", MASK_WIDTH);

    hostInput = rawInput + 2;

    // Allocate memory for CPU
    hostOutput = (float *)malloc(numRows * numCols * sizeof(float));

    // Allocate memory for GPU
    cudaMalloc((void**)&deviceInput, numRows * numCols * sizeof(float));
    cudaMalloc((void**)&deviceOutput, numRows * numCols * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(deviceInput, hostInput, numRows * numCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask_2D_tiled, hostMask, MASK_WIDTH * MASK_WIDTH * sizeof(float), 0, cudaMemcpyHostToDevice);

    // BLOCK_SIZE x BLOCK_SIZE threads: each thread loads exactly one tile element
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridDim(ceil((float)numCols / TILE_WIDTH), ceil((float)numRows / TILE_WIDTH), 1);

    // Launch the kernel
    convolution_2D_tiled_kernel<<<gridDim, blockDim>>>(deviceInput, deviceOutput, numRows, numCols);

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
