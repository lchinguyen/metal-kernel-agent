# Milestone 3 Report

## What Changed

- Kept the optimized path as a real custom Metal kernel and benchmarked three variants on the same `[1, 4096]` float32 input.
- Added a `shared_128` variant to test whether fewer threads and fewer reduction barriers help this tiny workload.
- Added a `simd_128` variant that uses SIMD-group reduction to cut threadgroup-memory traffic and barrier work.
- Added GPU-time collection from Metal command buffers so we can compare on-GPU work against end-to-end wall latency.

## Analysis

- Torch baseline mean latency: 0.3352 ms.
- The tensor is only 4096 float32 values, so memory traffic is small; the work is too small to be bandwidth-bound.
- GPU kernel time is far below end-to-end wall time for every variant, which points to command-buffer submission and synchronization overhead dominating total latency.
- The Swift helper process itself is outside the timed loop; the timed section starts immediately before command buffer creation and ends after GPU completion, so helper startup is not the measured bottleneck.
- The fastest variant was `shared_256` at 0.2115 ms mean wall latency with 0.0291 ms mean GPU time.

## Results

| Variant | Mean | Median | p50 | p95 | GPU Mean | CPU/Submit Overhead Mean | Max Abs Error | Mean Abs Error | Speedup vs Torch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| shared_256 | 0.2115 ms | 0.2097 ms | 0.2097 ms | 0.2298 ms | 0.0291 ms | 0.1824 ms | 0.000000 | 0.000000 | 1.59x |
| simd_128 | 0.2582 ms | 0.2355 ms | 0.2355 ms | 0.4049 ms | 0.0376 ms | 0.2206 ms | 0.000000 | 0.000000 | 1.30x |
| shared_128 | 0.2681 ms | 0.2400 ms | 0.2400 ms | 0.4053 ms | 0.0475 ms | 0.2205 ms | 0.000000 | 0.000000 | 1.25x |
