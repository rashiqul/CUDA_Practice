#include <wb.h>
#include "convolution_verify.h"

#define BLOCK_SIZE 8
#define MASK_WIDTH 3

__constant__ float mask_3D[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void convolution_3D_basic_kernel(const float *input, float *output, int mask_width, int depths, int rows, int cols)
{
    int depth = blockIdx.z * blockDim.z + threadIdx.z;
    int row   = blockIdx.y * blockDim.y + threadIdx.y;
    int col   = blockIdx.x * blockDim.x + threadIdx.x;

    float p_value = 0.0f;

    if (row < rows && col < cols && depth < depths)
    {
        int start_depth = depth - mask_width / 2;
        int start_row   = row   - mask_width / 2;
        int start_col   = col   - mask_width / 2;

        for (int k = 0; k < mask_width; k++) // Iterating over mask depth (z)
        {
            for (int i = 0; i < mask_width; i++) // Iterating over mask rows (y)
            {
                for (int j = 0; j < mask_width; j++) // Iterating over mask columns (x)
                {
                    if (start_depth + k >= 0 && start_depth + k < depths &&
                        start_row   + i >= 0 && start_row   + i < rows  &&
                        start_col   + j >= 0 && start_col   + j < cols)
                    {
                        // Indexing pattern for 3D array: input[depth * rows * cols + row * cols + col]
                        p_value += input[(start_depth + k) * rows * cols + (start_row + i) * cols + (start_col + j)] * 
                                   mask_3D[k][i][j];
                    }
                }
            }
        }
        output[depth * rows * cols + row * cols + col] = p_value;
    }
}

void setup_for_3D_basic_conv(void)
{
    float *hostInput;
    float *hostMask;
    float *hostOutput;

    float *deviceInput;
    float *deviceOutput;

    int totalElements;

    int numDepths;
    int numRows;
    int numCols;

    float *rawInput;

    // Import raw buffers
    rawInput = (float *)wbImport(DATA_DIRECTORY_3D "/input.dat", NULL);
    hostMask  = (float *)wbImport(DATA_DIRECTORY_3D "/kernel.dat", NULL);

    // Extract dimensions from the three header floats: depth, rows, cols
    numDepths = (int)rawInput[0];
    numRows   = (int)rawInput[1];
    numCols   = (int)rawInput[2];
    wbLog(TRACE, "3D input size: ", numDepths, "x", numRows, "x", numCols);
    wbLog(TRACE, "3D mask width: ", MASK_WIDTH);

    // hostInput points past the three header floats
    hostInput = rawInput + 3;

    totalElements = numDepths * numRows * numCols;

    // Allocate memory for CPU
    hostOutput = (float *)malloc(totalElements * sizeof(float));

    // Allocate memory for GPU
    cudaMalloc((void**)&deviceInput,  totalElements * sizeof(float));
    cudaMalloc((void**)&deviceOutput, totalElements * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(deviceInput, hostInput, totalElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask_3D, hostMask, MASK_WIDTH * MASK_WIDTH * MASK_WIDTH * sizeof(float), 0, cudaMemcpyHostToDevice);

    // Define block and grid dimensions (8^3 = 512 threads, within the 1024-thread limit)
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(ceil((float)numCols   / BLOCK_SIZE),
                 ceil((float)numRows   / BLOCK_SIZE),
                 ceil((float)numDepths / BLOCK_SIZE));

    // Launch the kernel
    convolution_3D_basic_kernel<<<gridDim, blockDim>>>(deviceInput, deviceOutput, MASK_WIDTH, numDepths, numRows, numCols);

    cudaDeviceSynchronize();

    // Copy result from device to host
    cudaMemcpy(hostOutput, deviceOutput, totalElements * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify result against expected output
    verify_3D_basic_conv(hostOutput, numDepths, numRows, numCols);

    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    free(rawInput);
    free(hostMask);
    free(hostOutput);
}
