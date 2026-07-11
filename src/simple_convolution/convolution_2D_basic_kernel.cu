#include <wb.h>
#include "convolution_verify.h"

#define BLOCK_SIZE 16
#define MASK_WIDTH 5

__constant__ float mask_2D_basic[MASK_WIDTH * MASK_WIDTH];

__global__ void convolution_2D_basic_kernel(float *N, float *P, int mask_width, int num_rows, int num_cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float p_value = 0.0f;

    if (row < num_rows && col < num_cols)
    {
        int start_row = row - (mask_width / 2);
        int start_col = col - (mask_width / 2);

        for(int i = 0; i < mask_width; i++)
        {
            for(int j = 0; j < mask_width; j++)
            {
                if ((start_row + i >= 0) && (start_row + i < num_rows) && (start_col + j >= 0) && (start_col + j < num_cols))
                {
                    p_value += N[(start_row + i) * num_cols + (start_col + j)] * mask_2D_basic[i * mask_width + j];
                }
            }
        }
        P[row * num_cols + col] = p_value;
    }
}

void setup_for_2D_basic_conv(void)
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
    wbLog(TRACE, "2D input size: ", numRows, "x", numCols);
    wbLog(TRACE, "2D mask width: ", MASK_WIDTH);

    // hostInput[0] must point to the first element of the input matrix
    hostInput = rawInput + 2;

    // Allocate memory for CPU
    hostOutput = (float *)malloc(numRows * numCols * sizeof(float));

    // Allocate memory for GPU
    cudaMalloc((void**)&deviceInput, numRows * numCols * sizeof(float));
    cudaMalloc((void**)&deviceOutput, numRows * numCols * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(deviceInput, hostInput, numRows * numCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask_2D_basic, hostMask, MASK_WIDTH * MASK_WIDTH * sizeof(float), 0, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridDim(ceil((float)numCols / BLOCK_SIZE), ceil((float)numRows / BLOCK_SIZE), 1);

    // Launch the kernel
    convolution_2D_basic_kernel<<<gridDim, blockDim>>>(deviceInput, deviceOutput, MASK_WIDTH, numRows, numCols);

    cudaDeviceSynchronize();

    // Copy result from device to host
    cudaMemcpy(hostOutput, deviceOutput, numRows * numCols * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify result against expected output
    verify_2D_basic_conv(hostOutput, numRows, numCols);

    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    free(rawInput);
    free(hostMask);
    free(hostOutput);
}
