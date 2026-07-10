#include <wb.h>
#include "convolution_verify.h"

#define BLOCK_SIZE 16
#define MASK_WIDTH 5
#define TILE_SIZE (BLOCK_SIZE + MASK_WIDTH - 1)


__constant__ float M[MASK_WIDTH * MASK_WIDTH];

__global__ void convolution_1D_basic_kernel(float *N, float *M, float *P, int mask_width, int width)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float p_value = 0.0f;
    int start_point = i - (mask_width / 2);

    for (int j = 0; j < mask_width; j++)
    {
        if (start_point + j >= 0 && start_point + j < width)
        {
            p_value += N[start_point + j] * M[j];
        }
    }

    if (i < width)
    {
        P[i] = p_value;
    }
}

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
                    p_value += N[(start_row + i) * num_cols + (start_col + j)] * M[i * mask_width + j];
                }
            }
        }
        P[row * num_cols + col] = p_value;
    }
}

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

static void setup_for_1D_conv(void)
{
    int inputLength;
    int maskWidth;
    float *hostInput;
    float *hostOutput;
    float *hostMask;
    float *deviceInput;
    float *deviceOutput;
    float *deviceMask;
    
    // Load input matrix
    hostInput = (float *)wbImport(DATA_DIRECTORY_1D "/input.dat", &inputLength);
    wbLog(TRACE, "1D input width: ", inputLength);

    // Load the mask matrix
    hostMask = (float *)wbImport(DATA_DIRECTORY_1D "/kernel.dat", &maskWidth);
    wbLog(TRACE, "1D mask width: ", maskWidth);

    // Allocate memory for CPU
    hostOutput = (float *)malloc(inputLength * sizeof(float));

    // Allocate memory for GPU
    cudaMalloc((void**)&deviceInput, inputLength * sizeof(float));
    cudaMalloc((void**)&deviceOutput, inputLength * sizeof(float));
    cudaMalloc((void**)&deviceMask, maskWidth * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMask, hostMask, maskWidth * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(BLOCK_SIZE, 1, 1);
    dim3 gridDim(ceil((float)inputLength / BLOCK_SIZE), 1, 1);

    // Launch the kernel
    convolution_1D_basic_kernel<<<gridDim, blockDim>>>(deviceInput, deviceMask, deviceOutput, maskWidth, inputLength);

    cudaDeviceSynchronize();

    // Copy result from device to host
    cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify result against expected output
    verify_1D_conv(hostOutput, inputLength);

    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    cudaFree(deviceMask);

    free(hostInput);
    free(hostOutput);
    free(hostMask);
}

static void setup_for_2D_conv(void)
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
    wbLog(TRACE, "2D input size: ", numRows, "x", numCols);

    // hostInput[0] must point to the first element of the input matrix
    hostInput = rawInput + 2;

    // Allocate memory for CPU
    hostOutput = (float *)malloc(numRows * numCols * sizeof(float));

    // Allocate memory for GPU
    cudaMalloc((void**)&deviceInput, numRows * numCols * sizeof(float));
    cudaMalloc((void**)&deviceOutput, numRows * numCols * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(deviceInput, hostInput, numRows * numCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(M, hostMask, maskWidth * sizeof(float), 0, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridDim(ceil((float)numCols / BLOCK_SIZE), ceil((float)numRows / BLOCK_SIZE), 1);

    // Launch the kernel
    convolution_2D_basic_kernel<<<gridDim, blockDim>>>(deviceInput, deviceOutput, MASK_WIDTH, numRows, numCols);

    cudaDeviceSynchronize();

    // Copy result from device to host
    cudaMemcpy(hostOutput, deviceOutput, numRows * numCols * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify result against expected output
    verify_2D_conv(hostOutput, numRows, numCols);

    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    free(rawInput);
    free(hostMask);
    free(hostOutput);
}

static void setup_for_2D_conv_tiled(void)
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
    verify_2D_conv(hostOutput, numRows, numCols);

    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    free(rawInput);
    free(hostMask);
    free(hostOutput);
}

int main(int argc, char *argv[])
{
    // Call setup function for 1D basic convolution
    setup_for_1D_conv();

    // Call setup function for 2D basic convolution
    setup_for_2D_conv(); 

    // Call setup function for 2D tiled convolution
    setup_for_2D_conv_tiled();


    return 0;
}