#include <wb.h>

#define  BLOCK_SIZE     1
#define  NUM_OF_THREADS 128
#define  WARP_SIZE      32

// A simple CUDA kernel that prints hello world from the GPU
__global__ void print_hello_world()
{
    printf("Hello, World from GPU!\n");
}

// A simple CUDA kernel that prints the block and thread IDs to demonstrate parallel execution
__global__ void print_block_thread_info()
{
    // Print the block and thread IDs to demonstrate parallel execution
    printf("The block ID is (%d, %d, %d). The thread ID is (%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
}

// A simple CUDA kernel that prints the warp information to demonstrate warp-level execution
__global__ void print_warp_info()
{
    // A warp is a group of 32 threads that execute the same instruction simultaneously.
    int warp_id = threadIdx.x / WARP_SIZE;
    
    // Print the warp information to demonstrate warp-level execution
    printf("Thread ID: %d, Block ID: %d, Warp ID: %d\n", threadIdx.x, blockIdx.x, warp_id);
}

// Function to print the properties of the CUDA devices available on the system
void print_device_properties()
{
    int deviceCount = 0;

    // Get the number of CUDA devices available
    cudaGetDeviceCount(&deviceCount);
    
    // Loop through each device and print its properties
    for (int device = 0; device < deviceCount; ++device) 
    {
        // Get the properties of the current device
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        if (device == 0) 
        {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999) 
            {
                wbLog(TRACE, "No CUDA GPU has been detected");
                return;
            } 
            else if (deviceCount == 1) 
            {
                //@@ WbLog is a provided logging API (similar to Log4J).
                //@@ The logging function wbLog takes a level which is either
                //@@ OFF, FATAL, ERROR, WARN, INFO, DEBUG, or TRACE and a
                //@@ message to be printed.
                wbLog(TRACE, "There is 1 device supporting CUDA");
            } 
            else 
            {
                wbLog(TRACE, "There are ", deviceCount,
                    " devices supporting CUDA");
            }
        }

        // Print the properties of the current device
        wbLog(TRACE, "Device ", device, " name: ", deviceProp.name);
        wbLog(TRACE, " Computational Capabilities: ", deviceProp.major, ".",
            deviceProp.minor);
        wbLog(TRACE, " Maximum global memory size: ",
            deviceProp.totalGlobalMem);
        wbLog(TRACE, " Maximum constant memory size: ",
            deviceProp.totalConstMem);
        wbLog(TRACE, " Maximum shared memory size per block: ",
            deviceProp.sharedMemPerBlock);
        wbLog(TRACE, " Maximum block dimensions: ",
            deviceProp.maxThreadsDim[0], " x ", deviceProp.maxThreadsDim[1],
            " x ", deviceProp.maxThreadsDim[2]);
        wbLog(TRACE, " Maximum grid dimensions: ", deviceProp.maxGridSize[0],
            " x ", deviceProp.maxGridSize[1], " x ",
            deviceProp.maxGridSize[2]);
        wbLog(TRACE, " Warp size: ", deviceProp.warpSize);
    }
}

int main()
{
    printf("Hello, World from CPU!\n");
    
    // Get the number of CUDA devices available and print it to the console
    print_device_properties();
    
    // Launch the kernel with the defined block size and number of threads
    // Kernel launch syntax: kernel<<<numBlocks, threadsPerBlock>>>(arguments);

    // Print hello world from the GPU using the kernel, using 1 block and 1 thread since we only want to print once
    print_hello_world<<<1, 1>>>();
    
    // Wait for the GPU to finish before proceeding
    cudaDeviceSynchronize();

    // Print new line for better readability
    printf("\n");

    // Print the block and thread information using the kernel
    print_block_thread_info<<<BLOCK_SIZE, NUM_OF_THREADS>>>();

    // Wait for the GPU to finish before proceeding
    cudaDeviceSynchronize();

    // Print new line for better readability
    printf("\n");
    
    // Print the warp information using the kernel
    print_warp_info<<<BLOCK_SIZE, NUM_OF_THREADS>>>();

    // Wait for the GPU to finish before exiting
    cudaDeviceSynchronize();


    return 0;
}