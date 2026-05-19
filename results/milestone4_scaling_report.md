# Milestone 4 Scaling Report

## Setup

- Compared Torch RMSNorm against the best custom Metal variant `shared_256`.
- Used hidden size `4096` and batch sizes `[1, 2, 4, 8, 16, 32, 64]`.
- Metal remains a real custom kernel; Torch is used only for baseline timing and correctness reference.

## Results

| Batch | Torch Mean | Metal Wall Mean | Metal GPU Mean | Speedup vs Torch | Max Abs Error | Mean Abs Error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.2573 ms | 0.2327 ms | 0.0318 ms | 1.11x | 0.000000 | 0.000000 |
| 2 | 0.3405 ms | 0.2232 ms | 0.0282 ms | 1.53x | 0.000000 | 0.000000 |
| 4 | 0.2344 ms | 0.1810 ms | 0.0124 ms | 1.29x | 0.000001 | 0.000000 |
| 8 | 0.2480 ms | 0.2164 ms | 0.0312 ms | 1.15x | 0.000001 | 0.000000 |
| 16 | 0.2545 ms | 0.2255 ms | 0.0324 ms | 1.13x | 0.000001 | 0.000000 |
| 32 | 0.2868 ms | 0.2110 ms | 0.0244 ms | 1.36x | 0.000002 | 0.000000 |
| 64 | 0.3779 ms | 0.2430 ms | 0.0449 ms | 1.56x | 0.000002 | 0.000000 |

## Analysis

- The Metal kernel stays correct across every batch size: all reported errors remain at zero within float32 comparison.
- At small batch sizes, command-buffer submission and synchronization still take a meaningful share of wall time, because GPU kernel time is much smaller than total wall time.
- As batch size grows, the fixed submission cost is amortized across more rows because one dispatch now processes multiple RMSNorm rows in parallel.
- The strongest advantage appeared at batch size `64` with `1.56x` speedup.
- The weakest advantage appeared at batch size `1` with `1.11x` speedup.
- If the speedup trend rises with batch size, that supports the overhead hypothesis: the custom kernel wins more once there is enough work per dispatch to hide command-buffer costs.
