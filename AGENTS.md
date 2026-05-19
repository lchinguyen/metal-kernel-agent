# AGENTS.md

This project is for a GPU MODE / Codex-style kernel optimization contest.

Mission:
Use OpenAI Codex as an agentic optimization system to build, benchmark, and improve custom Apple Metal kernels for local AI inference.

Core rule:
The optimized implementation must be a custom Metal kernel, not Torch, MLX, MPSGraph, Accelerate, or another standard library wrapper.

Allowed:
- Torch may be used only as the baseline and correctness reference.
- Python may be used for benchmarking, orchestration, and result logging.
- Metal Shading Language must be used for optimized kernels.

First target:
RMSNorm for transformer inference.

Agent roles:
1. Research Agent
   - Finds Metal optimization strategies.
   - Explains memory access, threadgroups, reductions, SIMD usage, and dtype tradeoffs.

2. Benchmark Methodologist
   - Designs fair benchmarks against Torch baseline.
   - Requires warmup runs, repeated timed runs, p50, p95, correctness error, and speedup.

3. Kernel Engineer
   - Writes custom .metal kernels.
   - Avoids silently replacing custom kernels with standard libraries.

4. Performance Critic
   - Reviews benchmark results.
   - Explains bottlenecks.
   - Proposes the next kernel change.

Definition of success:
- Torch RMSNorm baseline runs.
- Custom Metal RMSNorm kernel runs.
- Outputs match within tolerance.
- Benchmark proves latency improvement or explains why not.
- Results are saved to JSONL.
