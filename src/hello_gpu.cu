#include <cuda_runtime.h>
#include <iostream>

/* Kernel function to print "Hello, World!" from the device (GPU) */
__global__ void helloFromGPU(void)
{
    printf("Hello, World! From thread %d in block %d!\n", 
           threadIdx.x, blockIdx.x);
}

int main(void)
{
    /* Local defines */
    cudaDeviceProp deviceProp;
    int deviceCount = 0;
    
    /* Print a message from the host */
    std::cout << "Hello, World! From the CPU!" << std::endl;

    /* Launch the kernel eith 1 block and 10 threads*/
    helloFromGPU<<<1, 10>>>();
    /* Ensure all threads finish before moving on */
    cudaDeviceSynchronize();

    /* Get device count */
    cudaGetDeviceCount(&deviceCount);

    /* Retrieve and display device properties */
    cudaGetDeviceProperties(&deviceProp, deviceCount);

    std::cout << "Device " << deviceCount << " name: " << deviceProp.name << std::endl;
    std::cout << "Computational Capabilities: " << deviceProp.major << "." 
              << deviceProp.minor << std::endl;
    std::cout << "Maximum global memory size: " << deviceProp.totalGlobalMem 
              << " bytes" << std::endl;
    std::cout << "Maximum constant memory size: " << deviceProp.totalConstMem 
              << " bytes" << std::endl;
    std::cout << "Maximum shared memory size per block: " << deviceProp.sharedMemPerBlock 
              << " bytes" << std::endl;
    std::cout << "Maximum block dimensions: " << deviceProp.maxThreadsDim[0] << " x " 
              << deviceProp.maxThreadsDim[1] << " x " << deviceProp.maxThreadsDim[2] << std::endl;
    std::cout << "Maximum grid dimensions: " << deviceProp.maxGridSize[0] << " x "
              << deviceProp.maxGridSize[1] << " x " << deviceProp.maxGridSize[2] << std::endl;
    std::cout << "Warp size: " << deviceProp.warpSize << std::endl;
    std::cout << "Number of SMs: " << deviceProp.multiProcessorCount << std::endl;

    return 0;
}