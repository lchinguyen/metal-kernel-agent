from __future__ import annotations

from statistics import mean

from .base import BaseAgent, AgentResponse


class BenchmarkMethodologist(BaseAgent):
    agent_name = "benchmark_methodologist"
    system_role = (
        "Audit benchmark coverage, summarize measurement quality, and propose the "
        "next benchmark design needed to validate optimization work."
    )
    prompt_filename = "benchmark_methodologist.md"

    def review(self, operator_records: list[dict]) -> AgentResponse:
        batch_sizes = sorted({tuple(record["shape"])[0] for record in operator_records})
        speedups = [record.get("speedup_vs_torch") for record in operator_records if "speedup_vs_torch" in record]
        valid_speedups = [value for value in speedups if isinstance(value, (int, float))]
        average_speedup = mean(valid_speedups) if valid_speedups else None

        next_checks = [
            "Keep warmup and 100 timed iterations for continuity with earlier milestones.",
            "Record best-variant-per-batch summaries separately from all-variant raw data.",
            "Add a direct CPU-overhead estimate for Softmax just like RMSNorm if launch overhead remains ambiguous.",
        ]

        reasoning_summary = (
            "The benchmark set is broad enough to compare operators across realistic "
            "batch sizes, but Softmax would benefit from clearer overhead attribution "
            "and a stable best-variant summary."
        )
        return self.build_response(
            reasoning_summary=reasoning_summary,
            structured_output={
                "covered_batch_sizes": batch_sizes,
                "average_speedup_vs_torch": average_speedup,
                "next_checks": next_checks,
            },
        )
