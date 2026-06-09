# Hello World

Intro CUDA program with three simple kernels:

- print one line from GPU
- print block/thread indices
- print warp ID per thread

## Concepts

- `__global__` kernel declaration
- Launch configuration `<<<blocks, threads_per_block>>>`
- `blockIdx` and `threadIdx` as 3D coordinates: `(x, y, z)`
- Warp size (`32`) and warp ID via `threadIdx.x / WARP_SIZE`
- `cudaDeviceSynchronize()` to wait for GPU work before continuing

## Launch Configuration

```cpp
print_hello_world<<<1, 1>>>();
print_block_thread_info<<<1, 128>>>();
print_warp_info<<<1, 128>>>();
```

- `BLOCK_SIZE = 1`
- `NUM_OF_THREADS = 128`
- `WARP_SIZE = 32`
- For this 1D launch, `blockIdx.y = blockIdx.z = 0` and `threadIdx.y = threadIdx.z = 0`

Thread distribution for `128` threads with warp size `32`:

- warp `0`: thread `0..31`
- warp `1`: thread `32..63`
- warp `2`: thread `64..95`
- warp `3`: thread `96..127`

## Program Flow

1. CPU prints `Hello, World from CPU!`
2. GPU kernel `print_hello_world` prints one line
3. GPU kernel `print_block_thread_info` prints 128 lines (thread IDs)
4. GPU kernel `print_warp_info` prints 128 lines (thread ID + warp ID)

Each kernel launch is followed by `cudaDeviceSynchronize()`.

## Sample Output

```
Hello, World from CPU!

Hello, World from GPU!

The block ID is (0, 0, 0). The thread ID is (0, 0, 0)
The block ID is (0, 0, 0). The thread ID is (1, 0, 0)
...
The block ID is (0, 0, 0). The thread ID is (127, 0, 0)

Thread ID: 0, Block ID: 0, Warp ID: 0
Thread ID: 1, Block ID: 0, Warp ID: 0
...
Thread ID: 31, Block ID: 0, Warp ID: 0
Thread ID: 32, Block ID: 0, Warp ID: 1
...
Thread ID: 127, Block ID: 0, Warp ID: 3
```

> Output order may vary — GPU scheduling does not guarantee a fixed execution order.

## Build & Run

```bash
make build hello_world.cu
```
