#include <wb.h>
#include "convolution_verify.h"

#define MASK_WIDTH (3)
#define TILE_WIDTH (6)
#define BLOCK_SIZE (TILE_WIDTH + MASK_WIDTH - 1)

__constant__ float mask_3D_tiled[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void convolution_3D_tiled_kernel(const float *input, float *output, int depths, int rows, int cols)
{
    __shared__ float tile_N[BLOCK_SIZE][BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    
    int depth_out = blockIdx.z * TILE_WIDTH + tz;
    int row_out   = blockIdx.y * TILE_WIDTH + ty;
    int col_out   = blockIdx.x * TILE_WIDTH + tx;

    int row_in   = row_out   - MASK_WIDTH / 2;
    int col_in   = col_out   - MASK_WIDTH / 2;
    int depth_in = depth_out - MASK_WIDTH / 2;

    float pvalue = 0.0f;

    // Load data into shared memory tile
    if (depth_in >= 0 && depth_in < depths &&
        row_in   >= 0 && row_in   < rows &&
        col_in   >= 0 && col_in   < cols)
    {
        tile_N[tz][ty][tx] = input[depth_in * rows * cols + row_in * cols + col_in];
    }
    else
    {
        tile_N[tz][ty][tx] = 0.0f;
    }

    __syncthreads();

    // Perform convolution
    if (tz < TILE_WIDTH && ty < TILE_WIDTH && tx < TILE_WIDTH)
    {
        for (int k = 0; k < MASK_WIDTH; ++k)
        {
            for (int i = 0; i < MASK_WIDTH; ++i)
            {
                for (int j = 0; j < MASK_WIDTH; ++j)
                {
                    pvalue += tile_N[tz + k][ty + i][tx + j] * mask_3D_tiled[k][i][j];
                }
            }
        }
        if (depth_out < depths && row_out < rows && col_out < cols)
        {
            output[depth_out * rows * cols + row_out * cols + col_out] = pvalue;
        }
    }
}

void setup_for_3D_tiled_conv(void)
{
    float *hostInput;
    float *hostMask;
    float *hostOutput;

    float *deviceInput;
    float *deviceOutput;

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

    // Allocate memory for CPU
    hostOutput = (float *)malloc(numDepths * numRows * numCols * sizeof(float));

    // Allocate memory for GPU
    cudaMalloc((void**)&deviceInput,  numDepths * numRows * numCols * sizeof(float));
    cudaMalloc((void**)&deviceOutput, numDepths * numRows * numCols * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(deviceInput, hostInput, numDepths * numRows * numCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask_3D_tiled, hostMask, MASK_WIDTH * MASK_WIDTH * MASK_WIDTH * sizeof(float), 0, cudaMemcpyHostToDevice);

    // Define block and grid dimensions (8^3 = 512 threads, within the 1024-thread limit)
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(ceil((float)numCols   / TILE_WIDTH),
                 ceil((float)numRows   / TILE_WIDTH),
                 ceil((float)numDepths / TILE_WIDTH));

    // Launch the kernel
    convolution_3D_tiled_kernel<<<gridDim, blockDim>>>(deviceInput, deviceOutput, numDepths, numRows, numCols);

    cudaDeviceSynchronize();

    // Copy result from device to host
    cudaMemcpy(hostOutput, deviceOutput, numDepths * numRows * numCols * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify result against expected output
    verify_3D_tiled_conv(hostOutput, numDepths, numRows, numCols);

    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    free(rawInput);
    free(hostMask);
    free(hostOutput);
}
