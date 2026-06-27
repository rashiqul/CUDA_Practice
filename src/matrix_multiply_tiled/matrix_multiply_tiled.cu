#include <wb.h>

#define DATA_DIRECTORY "extern/ECE408_SP25/lab3/data/0"

#define TILE_WIDTH 16u

// Function prototypes
static void generate_random_matrix(float *matrix, int N);
static bool verify_result(float *A, float *B, float *C, int N);

// CUDA kernel for tiled matrix multiplication
__global__ void matrix_multiply(float *deviceA, float *deviceB, float *deviceC, int width)
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];
   
    int bx = blockIdx.x;  // Block index in the x-dimension
    int by = blockIdx.y;  // Block index in the y-dimension
    
    int tx = threadIdx.x; // Thread index in the x-dimension
    int ty = threadIdx.y; // Thread index in the y-dimension

    // Get the row and col indexes of the element to work on
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx; 

    float value = 0.0f;

    // Loop over the tiled sections of the input matrices
    for (int m = 0; m < ((width- 1) / TILE_WIDTH + 1); m++)
    {
        // Load the deviceA tiles into shared memory
        if (row < width && m * TILE_WIDTH + tx < width)
        {
            tileA[ty][tx] = deviceA[row * width + m * TILE_WIDTH + tx];
        }
        else
        {
            tileA[ty][tx] = 0.0f;
        }

        // Load the deviceB tiles into shared memory
        if (m * TILE_WIDTH + ty < width && col < width)
        {
            tileB[ty][tx] = deviceB[(m * TILE_WIDTH + ty) * width + col];
        }
        else
        {
            tileB[ty][tx] = 0.0f;
        }

        // Synchronize to make sure the tiles are loaded
        __syncthreads(); 

        // Iterate over the tiles to compute the value for the current element
        for (int k = 0; k < TILE_WIDTH ; ++k)
        {
            value += tileA[ty][k] * tileB[k][tx];
        }

        __syncthreads();
    }

    if (row < width && col < width)
    {
        deviceC[row * width + col] = value;
    }
}

// Generate square matrices of size N x N with random values
static void generate_random_matrix(float *matrix, int N)
{
    for (int i = 0; i < N * N; ++i)
    {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Verify the result of matrix multiplication by comparing with CPU computation.
// Uses a relative tolerance to handle accumulated floating-point rounding in large dot products.
// Writes a full comparison report (mismatches + summary) to the given output file.
static bool verify_result(float *A, float *B, float *C, int N)
{
    const float REL_TOL = 1e-3f; // relative tolerance
    int mismatch_count = 0;

    // Build the output path next to the source file using the compile-time __FILE__ macro
    char outpath[512];
    strncpy(outpath, __FILE__, sizeof(outpath) - 1);
    outpath[sizeof(outpath) - 1] = '\0';
    char *sep = strrchr(outpath, '/');
    if (sep)
        strcpy(sep + 1, "matrix_multiply_tiled_output.txt");
    else
        strncpy(outpath, "matrix_multiply_tiled_output.txt", sizeof(outpath));

    FILE *fp = fopen(outpath, "w");
    if (!fp)
    {
        fprintf(stderr, "Warning: could not open output file for writing.\n");
    }
    else
    {
        fprintf(fp, "Matrix multiplication comparison report\n");
        fprintf(fp, "Matrix size: %d x %d, relative tolerance: %g\n\n", N, N, (double)REL_TOL);
    }

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
            {
                sum += A[i * N + k] * B[k * N + j];
            }
            float abs_err  = fabsf(C[i * N + j] - sum);
            float rel_err  = abs_err / (fabsf(sum) + 1e-6f);
            if (rel_err > REL_TOL)
            {
                mismatch_count++;
                printf("Mismatch at C[%d][%d]: expected %f, got %f (rel_err=%e)\n",
                       i, j, sum, C[i * N + j], (double)rel_err);
                if (fp)
                    fprintf(fp, "Mismatch at C[%d][%d]: expected %f, got %f (rel_err=%e)\n",
                            i, j, sum, C[i * N + j], (double)rel_err);
            }
        }
    }

    bool passed = (mismatch_count == 0);
    if (fp)
    {
        fprintf(fp, "\n--- Summary ---\n");
        fprintf(fp, "Total elements checked : %d\n", N * N);
        fprintf(fp, "Mismatches found       : %d\n", mismatch_count);
        fprintf(fp, "Result                 : %s\n", passed ? "PASS" : "FAIL");
        fclose(fp);
    }
    return passed;
}

int main()
{
    // Size of the square matrices
    const int N = 1024;

    // Define and allocate host (CPU) memory for matrices A, B, and C
    float *hostA = (float *)malloc(N * N * sizeof(float));
    float *hostB = (float *)malloc(N * N * sizeof(float));
    float *hostC = (float *)malloc(N * N * sizeof(float));

    // Initialize matrices A and B with random values
    generate_random_matrix(hostA, N);
    generate_random_matrix(hostB, N);

    // Define device (GPU) pointers for matrices A, B, and C
    float *deviceA, *deviceB, *deviceC;

    // Allocate device memory for matrices A, B, and C
    cudaMalloc((void**)&deviceA, N * N * sizeof(float));
    cudaMalloc((void**)&deviceB, N * N * sizeof(float));
    cudaMalloc((void**)&deviceC, N * N * sizeof(float));

    // Copy matrices A and B from host (CPU) to device (GPU)
    cudaMemcpy(deviceA, hostA, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions for kernel launch
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(ceil((float)N/TILE_WIDTH), ceil((float)N/TILE_WIDTH), 1);

    // Launch the matrix multiplication kernel
    matrix_multiply<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, N);

    // Wait for the GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Copy the result matrix C from device (GPU) to host (CPU)
    cudaMemcpy(hostC, deviceC, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify the result by comparing with CPU computation
    if (verify_result(hostA, hostB, hostC, N))
    {
        printf("Matrix multiplication result is correct.\n");
    }
    else
    {
        printf("Matrix multiplication result is incorrect.\n");
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