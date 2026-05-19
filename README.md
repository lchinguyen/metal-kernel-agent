# Metal Kernel Agent: 
An agentic context-engineering framework that uses Codex-guided optimization loops to iteratively design, benchmark, analyze, and improve custom Apple Metal kernels for transformer inference workloads.

## Mission

Metal Kernel Agent is a project for Codex-driven agentic optimization of Apple Metal kernels for local AI inference. The goal is to use an explicit multi-agent workflow to design, benchmark, critique, and improve custom Metal Shading Language kernels for transformer-style operators on Apple Silicon, with Torch used only as the baseline and correctness reference while the optimized paths remain real custom GPU kernels.


## Architecture

```
AGENTS.md + .codex/skills
              ↓
        Orchestrator
              ↓
 ┌────────────┼────────────┐
 ↓            ↓            ↓
Research   Benchmark   Performance
 Agent    Methodologist  Critic
              ↓
       Kernel Engineer
              ↓
      Metal Kernel Variants
              ↓
        Benchmark Results
              ↓
      Optimization Plans
```

## Milestones Completed

- Torch RMSNorm baseline
- Custom Metal RMSNorm kernel
- RMSNorm optimization variants
- RMSNorm batch scaling
- Custom Metal Softmax kernel
- Multi-agent orchestration framework

## Strongest Benchmark Results

| Milestone | Best Result |
|---|---|
| RMSNorm best variant | **1.59x vs Torch** |
| RMSNorm scaling at batch 64 | **1.56x vs Torch** |
| Softmax at batch 1 | **1.84x vs Torch** |
| Softmax at batch 2 | **1.48x vs Torch** |
| Fused RMSNorm + Residual at batch 4 | **1.75x vs Torch** |
| Fused RMSNorm + Residual vs unfused Metal | **1.35x faster** |

All optimized kernels preserved correctness against Torch baselines with only minimal numerical error tolerance.

## Setup

1. Use macOS on Apple Silicon with Metal support.
2. Ensure Xcode command line tools and the Metal toolchain are available.
3. Create or activate a Python environment with PyTorch installed.
4. Run benchmarks from the repository root so result logs and generated reports land in `results/`.

Example environment setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch
```

## Run Commands

```bash
python benchmarks/rmsnorm_torch_baseline.py
python benchmarks/rmsnorm_metal.py
python benchmarks/rmsnorm_scaling.py
python benchmarks/softmax_metal.py
python agents/orchestrator.py
```

## Context Engineering

My role focused on designing the context-engineering and multi-agent optimization framework that guided how Codex performed Metal kernel optimization, benchmarking, and performance analysis. I defined the optimization objectives, benchmarking constraints, and success criteria through AGENTS.md and structured specialized `.codex/skills` modules for Metal kernel engineering, transformer operator optimization, and GPU benchmarking workflows. I also designed the interaction architecture between the Research Agent, Benchmark Methodologist, Kernel Engineer, and Performance Critic to enable structured optimization planning, benchmark-driven iteration, correctness validation against Torch baselines, and hardware-aware performance reasoning aligned with the contest requirements.

[AGENTS.md](/Users/chinguyen/metal-kernel-agent/AGENTS.md) defines the optimization goals, tooling constraints, benchmark rules, and success criteria for the agentic optimization workflow, while the `.codex/skills` modules provide specialized guidance for Metal kernel engineering, transformer operator optimization, GPU benchmarking, and performance analysis. Together, these context-engineering layers guide how Codex generates and evaluates optimization strategies, ensuring all milestones use real custom Metal kernels, reproducible benchmarking, correctness validation against Torch baselines, and explicit reasoning about GPU performance tradeoffs.


## Direct Metal Kernel Optimization 

The optimized implementations in this project are not wrappers around Torch, MLX, MPSGraph, Accelerate, NumPy, or another standard library kernel. Torch is used only as the baseline and correctness reference. The optimized paths compile and dispatch real custom Metal kernels from `.metal` source through explicit benchmark runners and Swift Metal helpers, which makes the results about kernel engineering rather than API selection.

## Current Operator Coverage

- RMSNorm:
  Custom float32 Metal kernels, multiple reduction variants, scaling benchmarks, and correctness checks against Torch.
- Softmax:
  Numerically stable custom float32 Metal kernels with row-max subtraction, multiple reduction variants, and scaling benchmarks against Torch.
- Agent Orchestration:
  Deterministic research, benchmarking, engineering, and critique agents that consume real benchmark logs and emit a structured optimization plan.

## Reports and Artifacts

- `results/results.jsonl`: Append-only benchmark history
- `results/milestone3_report.md`: RMSNorm optimization-variant analysis
- `results/milestone4_scaling_report.md`: RMSNorm workload-scaling analysis
- `results/milestone5_softmax_report.md`: Softmax kernel benchmark report
- `results/optimization_plan.json`: Latest orchestrated optimization plan
- `results/milestone6_agents_report.md`: Agent-framework architecture summary


## Skills Developed

- Designed and optimized custom Apple Metal kernels for transformer inference operators including RMSNorm and Softmax.
- Built benchmark-driven GPU optimization workflows comparing custom Metal kernels against Torch baselines with correctness validation and latency profiling.
- Implemented multi-variant kernel optimization strategies using threadgroup reductions, SIMD reductions, and workload scaling analysis.
- Developed agentic optimization framework with Research, Benchmark, Kernel Engineer, and Performance Critic agents orchestrated through Codex-style context engineering.
- Created reproducible benchmarking infrastructure using Python, Swift Metal helpers, JSONL experiment logging, and automated optimization reporting.
- Analyzed GPU bottlenecks including command-buffer overhead, synchronization costs, dispatch efficiency, and scaling behavior across batch sizes.
- Integrated Python benchmarking infrastructure with Swift Metal helpers, JSONL experiment logging, and modular orchestration pipelines for reproducible GPU optimization experiments.
- Applied hardware-aware performance engineering techniques including memory-access optimization, reduction-strategy benchmarking, and latency-versus-throughput tradeoff analysis on Apple Silicon GPUs.
