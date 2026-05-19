---
name: apple-metal-kernel-engineering
description: Use this skill when writing custom Apple Metal Shading Language kernels for AI inference operations.
---

Metal kernel rules:
- Write real .metal kernels.
- Use explicit thread/threadgroup indexing.
- Prefer coalesced memory access.
- Minimize global memory reads and writes.
- Use threadgroup or SIMD reductions when useful.
- Expose threadgroup size and tensor shape in benchmark logs.
- Keep kernels simple and correct before optimizing.