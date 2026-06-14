# Add Two Vectors

Intro CUDA program that adds two integer vectors element-wise on the GPU:

- allocate and populate input vectors on the CPU
- copy data to the GPU, run a parallel kernel, copy result back
- verify results against expected CPU output

## Concepts

- `__global__` kernel declaration
- Launch configuration `<<<blocks, threads_per_block>>>`
- `threadIdx.x` to map each thread to a unique vector element
- `cudaMalloc` / `cudaFree` for device memory management
- `cudaMemcpy` with `cudaMemcpyHostToDevice` and `cudaMemcpyDeviceToHost`
- `cudaDeviceSynchronize()` to wait for GPU work before reading results

## Launch Configuration

```cpp
add_two_vectors<<<BLOCK_SIZE, NUM_OF_THREADS>>>(d_A, d_B, d_C);
```

- `BLOCK_SIZE = 1`
- `NUM_OF_THREADS = 1024`
- Each thread handles exactly one element: `d_C[threadIdx.x] = d_A[threadIdx.x] + d_B[threadIdx.x]`

Input data layout:

- `h_A[i] = i` — values `0, 1, 2, ..., 1023`
- `h_B[i] = 1024 - i` — values `1024, 1023, ..., 1`
- `h_C[i] = h_A[i] + h_B[i] = 1024` for all `i`

## Program Flow

1. CPU allocates and populates `h_A` and `h_B`
2. GPU memory allocated for `d_A`, `d_B`, `d_C`
3. `h_A` and `h_B` copied to device (`cudaMemcpyHostToDevice`)
4. Kernel `add_two_vectors` launched with 1 block × 1024 threads
5. `cudaDeviceSynchronize()` waits for kernel completion
6. `d_C` copied back to `h_C` (`cudaMemcpyDeviceToHost`)
7. Results verified against expected values and output printed

## Sample Output

```
Adding two vectors on the GPU!
All results are correct!
```

> If any element mismatches the expected sum, an error message indicating the index, expected value, and actual value is printed instead.

## Build & Run

```bash
make build add_two_vectors.cu
```
