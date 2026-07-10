#include <wb.h>

#define TILE_WIDTH 16U
#define DATA_DIRECTORY "extern/ECE408_SP25/lab2/data/9"

__global__ void plus_sign_sum(float* in, float* out, int width)
{
    int bx = blockIdx.x; int tx = threadIdx.x;
    int by = blockIdx.y; int ty = threadIdx.y;

    __shared__ float M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float N[TILE_WIDTH][TILE_WIDTH];

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    float row_sum = 0.0f;
    float col_sum = 0.0f;

    // For each tile
    for(int i = 0; i < width / TILE_WIDTH; i++)
    {
        // Bound limits of M
        if ((row < width) && ((i * TILE_WIDTH + tx) < width))
        {
            // Load one horizental strip for row sum
            M[ty][tx] = in[row * width + (i * TILE_WIDTH + tx)];
        }
        else
        {
            M[ty][tx] = 0.0f;
        }

        // Bound limits of N
        if (((i * TILE_WIDTH + ty) < width) && (col < width))
        {
            // Load one vertical strip for col sum
            N[ty][tx] = in[(i * TILE_WIDTH + ty) * width + col];
        }
        else
        {
            N[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute the row and col sum for this tile
        for (int j = 0; j < TILE_WIDTH; j++)
        {
            row_sum += M[ty][j];
            col_sum += N[j][tx];
        }

        __syncthreads();
    }

    // Write output once after all tiles have been accumulated
    if (row < width && col < width)
    {
        out[row * width + col] = ((row_sum + col_sum) - in[row * width + col]);
    }
}

static void plus_sign_sum_host(float* h_in, float* h_out, int N)
{
    float *in;
    float *out;

    // Allocate GPU memory 
    cudaMalloc((void**)&in, N * N * sizeof(float));
    cudaMalloc((void**)&out, N * N * sizeof(float));

    // Copy memory to the GPU
    cudaMemcpy(in, h_in, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define block dimensions (threads per block in x, y and z)
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    // Define grid dimensions (blocks per grid in x, y and z)
    dim3 gridDim(ceil((float)N/TILE_WIDTH), (ceil((float)N/TILE_WIDTH)), 1);

    // Call the GPU kernel
    plus_sign_sum<<<gridDim, blockDim>>>(in, out, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_out, out, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(in);
    cudaFree(out);
}

static bool verify_result(float *h_in, float *gpu_out, int width)
{
    const float tolerance = 1e-3f;
    int mismatch_count = 0;

    for (int row = 0; row < width; row++)
    {
        for (int col = 0; col < width; col++)
        {
            float row_sum = 0.0f;
            float col_sum = 0.0f;

            for (int k = 0; k < width; k++)
            {
                row_sum += h_in[row * width + k];   // entire row
                col_sum += h_in[k * width + col];   // entire column
            }

            // element at (row,col) is counted in both row_sum and col_sum; subtract once
            float expected = row_sum + col_sum - h_in[row * width + col];
            float actual   = gpu_out[row * width + col];

            if (fabsf(expected - actual) > tolerance)
            {
                if (mismatch_count < 5)
                    wbLog(ERROR, "Mismatch at [", row, "][", col, "]: CPU=", expected, " GPU=", actual);
                mismatch_count++;
            }
        }
    }

    if (mismatch_count == 0)
        wbLog(TRACE, "Verification PASSED.");
    else
        wbLog(ERROR, "Verification FAILED: ", mismatch_count, " / ", width * width, " elements differ.");

    return mismatch_count == 0;
}

int main(void)
{
    float *input;
    float *output;

    int int_row;
    int int_col;

    // Import data
    input = (float *)wbImport(DATA_DIRECTORY "/input0.raw", &int_row, &int_col);
    // Print the dimensions of A
    wbLog(TRACE, "The dimensions of A are ", int_row, " x ", int_col);

    // width is the first dimension (number of rows)
    int width = int_row; // int_col will work as well

    // Allocate memory for output
    output = (float*)malloc(width * width * sizeof(float));

    // Call host code
    plus_sign_sum_host(input, output, width);

    // Verify GPU result against CPU reference
    verify_result(input, output, width);

    free(input);
    free(output);

    return 0;
}