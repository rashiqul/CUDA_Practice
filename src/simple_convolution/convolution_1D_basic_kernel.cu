#include <wb.h>
#include "convolution_verify.h"

#define BLOCK_SIZE 16
#define MASK_WIDTH 5

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

void setup_for_1D_basic_conv(void)
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
    verify_1D_basic_conv(hostOutput, inputLength);

    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    cudaFree(deviceMask);

    free(hostInput);
    free(hostOutput);
    free(hostMask);
}
