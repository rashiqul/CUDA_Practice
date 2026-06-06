# Hello World

First CUDA kernel. Demonstrates the basic anatomy of a GPU program.

## Concepts

- `__global__` kernel declaration
- Launch configuration `<<<blocks, threads_per_block>>>`
- `blockIdx.x` and `threadIdx.x` thread coordinates
- `cudaDeviceSynchronize()` — wait for GPU before CPU exits

## Launch Configuration

```
hello_world<<<2, 4>>>()
```

- 2 blocks × 4 threads = **8 total threads**
- Each thread gets a unique `(blockIdx.x, threadIdx.x)` pair

## Expected Output

```
Hello from block 0, thread 0
Hello from block 0, thread 1
Hello from block 0, thread 2
Hello from block 0, thread 3
Hello from block 1, thread 0
Hello from block 1, thread 1
Hello from block 1, thread 2
Hello from block 1, thread 3
```

> Order may vary — the GPU scheduler does not guarantee block execution order.

## Build & Run

```bash
make build hello_world.cu
```
