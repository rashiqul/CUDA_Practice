# Parallel Reduction

Reduces an array of N floats to a single sum using GPU parallelism.

## Concepts

- `__shared__` memory — fast on-chip memory shared within a block
- `__syncthreads()` — barrier synchronization across threads in a block
- Reduction tree — iterative halving until one thread holds the block result
- Partial sums — each block produces one output, host finishes the final accumulation

## Pattern

```
Input:  [1, 1, 1, 1, 1, 1, 1, 1]   (N = 8, one block of 8 threads)

Step 1 (stride=1):  [2, _, 2, _, 2, _, 2, _]
Step 2 (stride=2):  [4, _, _, _, 4, _, _, _]
Step 3 (stride=4):  [8, _, _, _, _, _, _, _]

Output: 8  ✓
```

Each block writes its partial sum to `output[blockIdx.x]`. The host sums the partials.

## Expected Output

```
Sum: 1048576  (expected: 1048576)
```

## Build & Run

```bash
make build parallel_reduction.cu
```
