from __future__ import annotations

from .base import BaseAgent, AgentResponse


class ResearchAgent(BaseAgent):
    agent_name = "research_agent"
    system_role = (
        "Identify operator-specific Metal optimization strategies and explain "
        "their memory, reduction, and numerical tradeoffs."
    )
    prompt_filename = "research_agent.md"

    def analyze(self, operator_summary: dict) -> AgentResponse:
        operator = operator_summary["operator"]
        bottleneck = operator_summary["dominant_bottleneck"]
        recommendations = []

        if operator == "softmax":
            recommendations = [
                {
                    "idea": "Fuse exponentiation reuse",
                    "why": "The current kernels compute exp twice per element. Caching row-local exponentials or staging them through threadgroup memory can reduce math overhead.",
                },
                {
                    "idea": "Use SIMD reductions for row max and row sum",
                    "why": "Softmax has two row reductions. SIMD-first reductions reduce barrier pressure for small and medium batches.",
                },
                {
                    "idea": "Process multiple rows per command buffer efficiently",
                    "why": "The benchmark data shows wall time remains much larger than GPU time, so launch overhead still matters.",
                },
            ]
        else:
            recommendations = [
                {
                    "idea": "Increase useful work per dispatch",
                    "why": "When GPU time is much lower than wall time, amortizing command submission overhead is usually the highest-leverage move.",
                }
            ]

        reasoning_summary = (
            f"{operator} is limited more by {bottleneck} than raw memory bandwidth, "
            "so the next useful ideas should either reduce reduction overhead or pack "
            "more useful work into each dispatch."
        )
        return self.build_response(
            reasoning_summary=reasoning_summary,
            structured_output={
                "operator": operator,
                "bottleneck": bottleneck,
                "recommendations": recommendations,
            },
        )
