# Milestone 9 Fusion Report

## Setup

- Benchmarked standalone RMSNorm, separate RMSNorm plus residual add, and a fused RMSNorm plus residual add Metal kernel.
- Used batch sizes `[1, 2, 4, 8, 16, 32, 64]`, hidden size `4096`, and float32.
- Torch was used only as the correctness and fused-operation baseline.

## Fused vs Unfused Results

| Batch | Standalone RMSNorm Mean | Separate RMSNorm+Residual Mean | Fused Mean | Fused GPU Mean | Speedup vs Torch | Speedup vs Unfused | Max Abs Error | Mean Abs Error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3948 ms | 0.3601 ms | 0.3558 ms | 0.0828 ms | 1.06x | 1.01x | 0.000000 | 0.000000 |
| 2 | 0.2770 ms | 0.2884 ms | 0.2730 ms | 0.0560 ms | 1.49x | 1.06x | 0.000001 | 0.000000 |
| 4 | 0.3220 ms | 0.3097 ms | 0.2287 ms | 0.0319 ms | 1.75x | 1.35x | 0.000001 | 0.000000 |
| 8 | 0.1962 ms | 0.2192 ms | 0.2149 ms | 0.0255 ms | 1.24x | 1.02x | 0.000001 | 0.000000 |
| 16 | 0.2318 ms | 0.2448 ms | 0.2368 ms | 0.0324 ms | 1.09x | 1.03x | 0.000001 | 0.000000 |
| 32 | 0.2357 ms | 0.2303 ms | 0.2222 ms | 0.0262 ms | 1.28x | 1.04x | 0.000002 | 0.000000 |
| 64 | 0.2588 ms | 0.2823 ms | 0.2904 ms | 0.0531 ms | 1.25x | 0.97x | 0.000002 | 0.000000 |

## Why Fusion Matters

- Fusion reduces memory traffic by avoiding an intermediate RMSNorm output write followed by a separate residual-read and output-write pass.
- Fusion reduces dispatch overhead because one command-encoded kernel replaces a two-kernel sequence for the same transformer subgraph step.
- In transformer inference, normalization and residual paths are frequent and latency-sensitive, so even modest per-layer savings can compound across many layers and tokens.

## Analysis

- The fused kernel is correctness-first: it keeps the existing fp32 RMSNorm accumulation path and adds the residual writeback in the same pass.
- Standalone RMSNorm provides the lower bound for normalization-only work, while the separate implementation shows the cost of keeping residual addition as another dispatch.
- The strongest fused advantage over the unfused Metal path appeared at batch `4` with `1.35x` speedup.
- If fused wall time stays materially above fused GPU time, the remaining opportunity is still launch overhead and kernel packing rather than pure arithmetic cost.
