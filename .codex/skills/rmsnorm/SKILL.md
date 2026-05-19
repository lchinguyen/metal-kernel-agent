---
name: rmsnorm-optimization
description: Use this skill when implementing or optimizing RMSNorm kernels for transformer inference on Apple Silicon Metal.
---

RMSNorm formula:

y = x * rsqrt(mean(x^2) + eps) * weight

Optimization requirements:
- Start with shape [1, 4096].
- Support float32 first, then float16 later.
- Use Torch only as the correctness baseline.
- Optimized path must use a custom Metal kernel.
- Use fp32 accumulation for numerical stability.
- Benchmark latency and correctness after every change.
- Save results to results/results.jsonl.

Do not:
- Replace the optimized kernel with torch, MLX, MPSGraph, Accelerate, or NumPy.
- Claim speedup without benchmark evidence.