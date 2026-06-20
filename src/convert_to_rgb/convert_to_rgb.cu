#include <wb.h>

static void pixel_setup(void);
static void pixel_print(unsigned char *imageData, int width, int height, const char *filename);

#define IMAGE_WIDTH  (1388)
#define IMAGE_HEIGHT (1133)

// Host side buffer to hold the input grayscale data 
unsigned char *hostImageData = (unsigned char *)malloc(IMAGE_WIDTH  * IMAGE_HEIGHT * 
                                                       sizeof(unsigned char));

// A setup function to read the pixel values from the input file and store them in a host buffer
static void pixel_setup(void)
{
    /* Open the input file for reading */
    FILE *f = fopen("/home/rashiqul/Workspace/CUDA_Practice/src/convert_to_rgb/input/input_pixels.txt", 
                    "r");

    /* Check for null pointer */
    if (f == NULL) 
    {
        /* Prints the actual error message to help with debugging */
        perror("fopen failed to read the input file");  
        return;
    }

    /* Parse the pixels from the input file */
    for(int row = 0; row < IMAGE_HEIGHT; row++)
    {
        for(int col = 0; col < IMAGE_WIDTH; col++)
        {
            int value;
            fscanf(f, "%d", &value);
            hostImageData[row * IMAGE_WIDTH + col] = (unsigned char)value;
        }
    }
    fclose(f);
}

// A function to print the pixel values from the host buffer to an output file
static void pixel_print(unsigned char *imageData, int width, int height, const char *filename)
{
    /* Open the output file for writing */
    FILE *f = fopen(filename, "w");
    if (f == NULL)
    {
        perror("fopen failed to write the output file");
        return;
    }

    for(int row = 0; row < height; row++)
    {
        for(int col = 0; col < width; col++)
        {
            int index = row * width + col;
            fprintf(f, "(%d, %d, %d) ",  imageData[3 * index + 0], 
                                         imageData[3 * index + 1], 
                                         imageData[3 * index + 2]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

// Device helper: maps a grayscale intensity (0-255) to an RGB triplet using the "hot" colormap.
// black(0) -> red -> yellow -> white(255)
// Divided into 3 equal segments of ~85 steps each.
__device__ void hotColormap(unsigned char I, unsigned char *R, unsigned char *G, unsigned char *B)
{
    if (I < 85)
    {
        *R = (unsigned char)(I * 3);
        *G = 0;
        *B = 0;
    }
    else if (I < 170)
    {
        *R = 255;
        *G = (unsigned char)((I - 85) * 3);
        *B = 0;
    }
    else
    {
        *R = 255;
        *G = 255;
        *B = (unsigned char)((I - 170) * 3);
    }
}

// CUDA kernel to convert grayscale image to RGB format
__global__ void convert2RGBkernel(unsigned char *inputImageData, 
                                  unsigned char *outputRGBImageData,
                                  int width,
                                  int height)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col < width && row < height)
    {
        /* Using the row major layout to get the cordinate per thread operation */
        int index = row * width + col;

        /* Copy the grayscale value to the RGB channels */
        outputRGBImageData[3 * index + 0] = inputImageData[index];
        outputRGBImageData[3 * index + 1] = inputImageData[index];
        outputRGBImageData[3 * index + 2] = inputImageData[index];
    }
}

// CUDA kernel to apply a false-color ("hot") colormap to a grayscale image
__global__ void convert2FalseColorKernel(unsigned char *inputImageData,
                                         unsigned char *outputRGBImageData,
                                         int width,
                                         int height)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col < width && row < height)
    {
        int index = row * width + col;
        unsigned char R, G, B;

        /* Map the grayscale intensity to a false color using the hot colormap */
        hotColormap(inputImageData[index], &R, &G, &B);

        outputRGBImageData[3 * index + 0] = R;
        outputRGBImageData[3 * index + 1] = G;
        outputRGBImageData[3 * index + 2] = B;
    }
}


int main(void)
{
    // Call the setup function to obtain the pixel values
    pixel_setup();

    // Host side buffer to hold the output RGB data
    unsigned char *hostRGBImageData = (unsigned char *)malloc(3 * IMAGE_WIDTH  * IMAGE_HEIGHT * 
                                                              sizeof(unsigned char));

    // Define some device side buffer pointers
    unsigned char *deviceImageData;
    unsigned char *deviceRGBImageData;

    // Allocate memory on the device
    cudaMalloc((void**)&deviceImageData, IMAGE_WIDTH  * IMAGE_HEIGHT * sizeof(unsigned char));
    cudaMalloc((void**)&deviceRGBImageData, 3 * IMAGE_WIDTH  * IMAGE_HEIGHT * sizeof(unsigned char));

    // Copy the pixel values from the host buffer to the device buffer
    cudaMemcpy(deviceImageData, 
               hostImageData, 
               IMAGE_WIDTH  * IMAGE_HEIGHT * sizeof(unsigned char), 
               cudaMemcpyHostToDevice);
    
    // Define the block dimensions
    dim3 blockDim(32, 32);

    // Define the grid dimensions
    // ceil(IMAGE_WIDTH  / 32) = 44
    // ceil(IMAGE_HEIGHT / 32) = 36
    dim3 gridDim(44, 36, 1);

    // Host/device buffers for false-color output
    unsigned char *hostFalseColorData = (unsigned char *)malloc(3 * IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(unsigned char));
    unsigned char *deviceFalseColorData;
    cudaMalloc((void**)&deviceFalseColorData, 3 * IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(unsigned char));

    // Launch the grayscale-to-RGB kernel
    convert2RGBkernel<<<gridDim, blockDim>>>(deviceImageData, deviceRGBImageData, IMAGE_WIDTH, IMAGE_HEIGHT);
    cudaDeviceSynchronize();

    // Launch the false-color kernel (reuses the same input buffer)
    convert2FalseColorKernel<<<gridDim, blockDim>>>(deviceImageData, deviceFalseColorData, IMAGE_WIDTH, IMAGE_HEIGHT);
    cudaDeviceSynchronize();

    // Copy both results back to host
    cudaMemcpy(hostRGBImageData,
               deviceRGBImageData,
               3 * IMAGE_WIDTH  * IMAGE_HEIGHT * sizeof(unsigned char), 
               cudaMemcpyDeviceToHost);

    cudaMemcpy(hostFalseColorData,
               deviceFalseColorData,
               3 * IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(unsigned char),
               cudaMemcpyDeviceToHost);

    // Write both output files
    pixel_print(hostRGBImageData,    IMAGE_WIDTH, IMAGE_HEIGHT,
                "/home/rashiqul/Workspace/CUDA_Practice/src/convert_to_rgb/output/output_pixels.txt");
    pixel_print(hostFalseColorData,  IMAGE_WIDTH, IMAGE_HEIGHT,
                "/home/rashiqul/Workspace/CUDA_Practice/src/convert_to_rgb/output/output_false_color.txt");

    // Free the device memory
    cudaFree(deviceImageData);
    cudaFree(deviceRGBImageData);
    cudaFree(deviceFalseColorData);

    // Free the host memory
    free(hostImageData);
    free(hostRGBImageData);
    free(hostFalseColorData);

    return 0;
}
