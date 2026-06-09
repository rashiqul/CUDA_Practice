# Hello World

First CUDA kernel. Demonstrates the basic anatomy of a GPU program and how blocks/threads are indexed in 3D.

## Concepts

- `__global__` kernel declaration
- Launch configuration `<<<blocks, threads_per_block>>>`
- `blockIdx` and `threadIdx` as 3D coordinates: `(x, y, z)`
- `cudaDeviceSynchronize()` — wait for GPU before CPU exits

## Launch Configuration

```
print_block_thread_info<<<2, 32>>>()
```

- 2 blocks × 32 threads = **64 total threads**
- For this 1D launch, `blockIdx.y = blockIdx.z = 0` and `threadIdx.y = threadIdx.z = 0`
- Each thread prints its block ID and thread ID as `(x, y, z)`

## Expected Output

```
Hello, World from CPU!
The block ID is (0, 0, 0). The thread ID is (0, 0, 0)
The block ID is (0, 0, 0). The thread ID is (1, 0, 0)
...
The block ID is (0, 0, 0). The thread ID is (31, 0, 0)
The block ID is (1, 0, 0). The thread ID is (0, 0, 0)
...
The block ID is (1, 0, 0). The thread ID is (31, 0, 0)
```

> Output order may vary — GPU scheduling does not guarantee a fixed execution order.

## Build & Run

```bash
make build hello_world.cu
```
