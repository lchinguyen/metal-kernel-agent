# Milestone 6 Agents Report

## Why Explicit Agent Decomposition Matters

- Kernel optimization needs different kinds of judgment: benchmark interpretation, low-level kernel design, research synthesis, and iteration planning.
- Splitting those responsibilities into named agents makes the optimization loop auditable instead of hiding everything inside one benchmark script.
- It also makes failure modes easier to localize: if the plan is weak, we can inspect whether the critic, the engineer, or the benchmark context was the limiting factor.

## Context Engineering

- Each agent receives only the benchmark slice it needs rather than the full project state.
- Prompt templates in `agents/prompts/` make the intended role and output format explicit, which is the same surface we would later send to a real LLM call.
- Structured outputs keep the orchestration deterministic today while preserving a clean interface for future model-backed reasoning.

## Iterative Optimization Architecture

- The orchestrator loaded existing results, identified `softmax` as the weakest operator, and routed that context through the critic, research agent, benchmark methodologist, and kernel engineer.
- The performance critic turns raw latency data into a bottleneck diagnosis.
- The research agent maps that diagnosis to Metal-specific optimization ideas.
- The kernel engineer converts those ideas into concrete implementation tasks and target files.
- The resulting `results/optimization_plan.json` is lightweight, deterministic, and ready to be replaced later with real Codex/OpenAI API-backed agent calls.

## Current Outcome

- Weakest operator: `softmax`
- Critic summary: softmax is currently the weakest operator because its mean speedup profile is only 1.19x and some measured configurations still lose to Torch.
- Engineer summary: The next optimization work for softmax should target the specific cost centers surfaced by the critic instead of broad rewrites.
