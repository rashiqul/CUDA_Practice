#include <wb.h>

#define DATA_DIRECTORY "extern/ECE408_SP25/lab2/data/9"

// CUDA kernel for simple matrix multiplication
__global__ void matrix_multiply(float *deviceA, float *deviceB, float *deviceC, int numARows, int numAColumns, int numBRows, int numBColumns)
{
    float sum = 0.0f;
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < numARows && col < numBColumns)
    {
        for (int i = 0; i < numAColumns; i++)
        {
            float a = deviceA[row * numAColumns + i];
            float b = deviceB[i * numBColumns + col];
            sum += a * b;
        }
        deviceC[row * numBColumns + col] = sum;
    }
}


int main(int argc, char **argv) 
{
    float *hostA; // The A matrix
    float *hostB; // The B matrix
    float *hostC; // The output C matrix

    int numARows;    // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows;    // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B

    // Import data from the input files
    hostA = (float *)wbImport(DATA_DIRECTORY "/input0.raw", &numARows,
                            &numAColumns);
    hostB = (float *)wbImport(DATA_DIRECTORY "/input1.raw", &numBRows,
                            &numBColumns);

    // Print the dimensions of the matrices A and B
    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
    
    if (numAColumns != numBRows) {
        wbLog(ERROR, "Matrix multiplication is not possible: A's columns (", numAColumns, ") must equal B's rows (", numBRows, ").");
        return -1;
    }

    // Allocate memory for the output C matrix
    hostC = (float*)malloc(numARows * numBColumns * sizeof(float));

    // Define device pointers for matrices A, B, and C
    float *deviceA, *deviceB, *deviceC;

    // Allocate memory on the GPU for matrices A, B, and C
    cudaMalloc((void**)&deviceA, numARows * numAColumns * sizeof(float));
    cudaMalloc((void**)&deviceB, numBRows * numBColumns * sizeof(float));
    cudaMalloc((void**)&deviceC, numARows * numBColumns * sizeof(float));

    // Copy data from host (CPU) to device (GPU)
    cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);

    // Launch configuration for the kernel
    dim3 dimBlock(32, 32, 1); 
    dim3 dimGrid(((ceil(float(numBColumns) / 32))), ((ceil(float(numARows) / 32))), 1);

    matrix_multiply<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);

    cudaDeviceSynchronize();

    // Copy data from device (GPU) to host (CPU)
    cudaMemcpy(hostC, deviceC, numARows * numBColumns * sizeof(float), cudaMemcpyDeviceToHost);


    // Load expected output for side-by-side comparison
    int numExpRows, numExpColumns;
    float *hostExpected = (float *)wbImport(DATA_DIRECTORY "/output.raw",
                                            &numExpRows, &numExpColumns);

    int total = numARows * numBColumns;
    printf("\n%-6s  %-12s  %-12s\n", "Index", "GPU Result", "Expected");
    printf("------  ------------  ------------\n");
    for (int i = 0; i < total; i++) {
        if (i == 10 && total > 20)
            printf("  ...   (showing first 10 and last 10)\n");
        if (i >= 10 && i < total - 10)
            continue;
        printf("[%4d]  %12.6f  %12.6f\n", i, hostC[i], hostExpected[i]);
    }
    printf("\n");

    free(hostExpected);

    // Validate the output by comparing it to the expected result
    wbBool correct = wbSolution((char *)DATA_DIRECTORY "/output.raw", nullptr,
                                (char *)"matrix", hostC, numARows, numBColumns);
    
    if (correct)
    {
        wbLog(TRACE, "The output is correct.");
    }
    else
    {
        wbLog(TRACE, "The output is incorrect.");
    }
    
    // Free device memory
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    // Free host memory
    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}

