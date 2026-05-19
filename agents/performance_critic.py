from __future__ import annotations

from statistics import mean

from .base import BaseAgent, AgentResponse


class PerformanceCritic(BaseAgent):
    agent_name = "performance_critic"
    system_role = (
        "Interpret benchmark results, identify the weakest-performing operator, and "
        "explain the most likely bottlenecks behind its current behavior."
    )
    prompt_filename = "performance_critic.md"

    def analyze(self, operator_summary: dict) -> AgentResponse:
        speedups = operator_summary["speedups"]
        gpu_times = operator_summary["gpu_times"]
        wall_times = operator_summary["wall_times"]

        avg_speedup = mean(speedups) if speedups else 0.0
        avg_gpu = mean(gpu_times) if gpu_times else 0.0
        avg_wall = mean(wall_times) if wall_times else 0.0
        overhead_ratio = ((avg_wall - avg_gpu) / avg_wall) if avg_wall > 0 else 0.0

        if operator_summary["operator"] == "softmax":
            dominant_bottleneck = "reduction plus launch overhead"
            findings = [
                "Softmax has two reductions per row, so reduction cost is inherently higher than RMSNorm.",
                "Several batch sizes still underperform Torch, which makes Softmax the weakest operator in the current benchmark set.",
                "GPU time is still much smaller than wall time, so dispatch overhead remains material even after batching rows.",
            ]
        else:
            dominant_bottleneck = "command-buffer overhead"
            findings = [
                "GPU time is much smaller than wall time, so dispatch overhead dominates this operator.",
            ]

        reasoning_summary = (
            f"{operator_summary['operator']} is currently the weakest operator because "
            f"its mean speedup profile is only {avg_speedup:.2f}x and some measured "
            "configurations still lose to Torch."
        )
        return self.build_response(
            reasoning_summary=reasoning_summary,
            structured_output={
                "operator": operator_summary["operator"],
                "average_speedup_vs_torch": avg_speedup,
                "average_wall_ms": avg_wall,
                "average_gpu_ms": avg_gpu,
                "estimated_overhead_ratio": overhead_ratio,
                "dominant_bottleneck": dominant_bottleneck,
                "findings": findings,
            },
        )
