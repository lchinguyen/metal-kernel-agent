from __future__ import annotations

from .base import BaseAgent, AgentResponse


class KernelEngineer(BaseAgent):
    agent_name = "kernel_engineer"
    system_role = (
        "Translate bottleneck analysis into concrete kernel or dispatch changes that "
        "can be benchmarked next."
    )
    prompt_filename = "kernel_engineer.md"

    def propose(self, critic_output: dict, research_output: dict) -> AgentResponse:
        operator = critic_output["operator"]
        proposals = []

        if operator == "softmax":
            proposals = [
                {
                    "priority": 1,
                    "title": "Cache exponentials to avoid recomputation",
                    "target_files": ["kernels/softmax.metal"],
                    "rationale": "Current Softmax computes exp during the sum pass and again during normalization.",
                    "expected_impact": "Lower GPU math cost and better throughput for medium and large batches.",
                },
                {
                    "priority": 2,
                    "title": "Benchmark multi-row dispatch packing",
                    "target_files": ["benchmarks/softmax_metal_helper.swift", "benchmarks/softmax_metal.py"],
                    "rationale": "Wall time is still much larger than GPU time, so launch overhead likely needs further amortization.",
                    "expected_impact": "Better wall-clock performance on small batches.",
                },
                {
                    "priority": 3,
                    "title": "Tune shared vs SIMD reduction by batch regime",
                    "target_files": ["kernels/softmax.metal"],
                    "rationale": "The best variant flips by batch size, which suggests the dispatch policy should become input-aware.",
                    "expected_impact": "More consistent wins across the full scaling range.",
                },
            ]
        else:
            proposals = [
                {
                    "priority": 1,
                    "title": "Amortize launch overhead further",
                    "target_files": ["benchmarks/"],
                    "rationale": "GPU work is already cheap relative to wall time.",
                    "expected_impact": "Improve small-batch latency.",
                }
            ]

        reasoning_summary = (
            f"The next optimization work for {operator} should target the specific "
            "cost centers surfaced by the critic instead of broad rewrites."
        )
        return self.build_response(
            reasoning_summary=reasoning_summary,
            structured_output={
                "operator": operator,
                "proposals": proposals,
                "research_alignment": research_output["recommendations"],
            },
        )
