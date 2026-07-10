"""
generate_data.py
Generates plain-text .raw files for the MAC (A * B + C) CUDA kernel.

Usage:
    python3 data/generate_data.py [n]

    n  -- matrix dimension (must be divisible by 64, default 512)

Output files (written to the same directory as this script):
    A.raw      -- first line: "n n", then n rows of n space-separated float32 values
    B.raw      -- first line: "n n", then n rows of n space-separated float32 values
    C.raw      -- first line: "n n", then n rows of n space-separated float32 values
    result.raw -- first line: "n n", then n rows of n space-separated float32 values
"""

import sys
import os
import numpy as np

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 512

    if n % 64 != 0:
        print(f"Error: n={n} is not divisible by 64.", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(seed=42)

    A = rng.random((n, n), dtype=np.float32)
    B = rng.random((n, n), dtype=np.float32)
    C = rng.random((n, n), dtype=np.float32)

    # Reference result: A * B + C  (standard matrix multiply + add)
    result = (A @ B + C).astype(np.float32)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    for name, mat in [("A", A), ("B", B), ("C", C), ("result", result)]:
        path = os.path.join(out_dir, f"{name}.raw")
        with open(path, "w") as f:
            f.write(f"{n} {n}\n")
            for row in mat:
                f.write(" ".join(f"{v:.6f}" for v in row) + "\n")
        size_kb = os.path.getsize(path) / 1024
        print(f"Wrote {path}  ({size_kb:.1f} KB)")

    print(f"\nDone. n={n}, each matrix is {n}x{n} float32.")

if __name__ == "__main__":
    main()
