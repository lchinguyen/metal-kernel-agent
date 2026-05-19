---
name: benchmark-methodology
description: Use this skill when creating fair kernel benchmarks against Torch or native baselines.
---

Benchmark rules:
- Use the same tensor shape, dtype, and input values.
- Run warmup iterations before timing.
- Run at least 100 timed iterations.
- Report median latency, p50, p95, mean latency, and speedup.
- Report max absolute error and mean absolute error.
- Do not include compilation time in runtime.
- Save every run to results/results.jsonl as JSONL.