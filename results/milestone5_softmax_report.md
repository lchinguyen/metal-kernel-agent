# Milestone 5 Softmax Report

## Setup

- Added a real custom Metal Softmax kernel with numerically stable row-wise max subtraction.
- Benchmarked batch sizes `[1, 2, 4, 8, 16, 32, 64]` at hidden size `4096` in float32.
- Torch Softmax is used only as the baseline and correctness reference.
- Two Metal variants were measured: `shared_256` correctness-first and `simd_256` as an optimized reduction variant.

## Best Variant Per Batch

| Batch | Best Variant | Torch Mean | Metal Wall Mean | Metal GPU Mean | Speedup vs Torch | Max Abs Error | Mean Abs Error |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | simd_256 | 0.3314 ms | 0.1800 ms | 0.0121 ms | 1.84x | 0.000000 | 0.000000 |
| 2 | simd_256 | 0.2660 ms | 0.1794 ms | 0.0121 ms | 1.48x | 0.000000 | 0.000000 |
| 4 | shared_256 | 0.1726 ms | 0.2194 ms | 0.0263 ms | 0.79x | 0.000000 | 0.000000 |
| 8 | shared_256 | 0.1992 ms | 0.2004 ms | 0.0166 ms | 0.99x | 0.000000 | 0.000000 |
| 16 | simd_256 | 0.2196 ms | 0.2403 ms | 0.0387 ms | 0.91x | 0.000000 | 0.000000 |
| 32 | shared_256 | 0.2755 ms | 0.2328 ms | 0.0385 ms | 1.18x | 0.000000 | 0.000000 |
| 64 | shared_256 | 0.2862 ms | 0.2477 ms | 0.0507 ms | 1.16x | 0.000000 | 0.000000 |

## All Variants

| Batch | Variant | Torch Mean | Metal Wall Mean | Metal GPU Mean | Speedup vs Torch | Max Abs Error | Mean Abs Error |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | simd_256 | 0.3314 ms | 0.1800 ms | 0.0121 ms | 1.84x | 0.000000 | 0.000000 |
| 1 | shared_256 | 0.3314 ms | 0.3245 ms | 0.0351 ms | 1.02x | 0.000000 | 0.000000 |
| 2 | simd_256 | 0.2660 ms | 0.1794 ms | 0.0121 ms | 1.48x | 0.000000 | 0.000000 |
| 2 | shared_256 | 0.2660 ms | 0.2053 ms | 0.0127 ms | 1.30x | 0.000000 | 0.000000 |
| 4 | shared_256 | 0.1726 ms | 0.2194 ms | 0.0263 ms | 0.79x | 0.000000 | 0.000000 |
| 4 | simd_256 | 0.1726 ms | 0.2201 ms | 0.0267 ms | 0.78x | 0.000000 | 0.000000 |
| 8 | shared_256 | 0.1992 ms | 0.2004 ms | 0.0166 ms | 0.99x | 0.000000 | 0.000000 |
| 8 | simd_256 | 0.1992 ms | 0.2469 ms | 0.0441 ms | 0.81x | 0.000000 | 0.000000 |
| 16 | simd_256 | 0.2196 ms | 0.2403 ms | 0.0387 ms | 0.91x | 0.000000 | 0.000000 |
| 16 | shared_256 | 0.2196 ms | 0.2426 ms | 0.0431 ms | 0.91x | 0.000000 | 0.000000 |
| 32 | shared_256 | 0.2755 ms | 0.2328 ms | 0.0385 ms | 1.18x | 0.000000 | 0.000000 |
| 32 | simd_256 | 0.2755 ms | 0.2516 ms | 0.0299 ms | 1.09x | 0.000000 | 0.000000 |
| 64 | shared_256 | 0.2862 ms | 0.2477 ms | 0.0507 ms | 1.16x | 0.000000 | 0.000000 |
| 64 | simd_256 | 0.2862 ms | 0.2754 ms | 0.0456 ms | 1.04x | 0.000000 | 0.000000 |

## Analysis

- Both Metal variants remain real custom kernels and stay numerically stable by subtracting the row max before exponentiation.
- GPU kernel time is expected to be much smaller than wall time for smaller batches, so command submission overhead still matters here too.
- The single fastest measured configuration was batch `2` with variant `simd_256` at `0.1794 ms` wall time.
- The SIMD reduction variant is intended to reduce barrier and threadgroup-memory overhead relative to the shared-memory baseline.
- Correctness stayed tight against Torch across all measured runs; any nonzero differences are small float32 rounding effects rather than algorithmic drift.
