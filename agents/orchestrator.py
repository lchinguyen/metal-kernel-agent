from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .benchmark_methodologist import BenchmarkMethodologist
    from .kernel_engineer import KernelEngineer
    from .performance_critic import PerformanceCritic
    from .research_agent import ResearchAgent
except ImportError:  # pragma: no cover - direct script fallback
    from agents.benchmark_methodologist import BenchmarkMethodologist
    from agents.kernel_engineer import KernelEngineer
    from agents.performance_critic import PerformanceCritic
    from agents.research_agent import ResearchAgent


RESULTS_PATH = ROOT / "results" / "results.jsonl"
PLAN_PATH = ROOT / "results" / "optimization_plan.json"
REPORT_PATH = ROOT / "results" / "milestone6_agents_report.md"
LOG_PATH = ROOT / "results" / "agent_orchestration.log"


@dataclass
class OperatorSummary:
    operator: str
    record_count: int
    speedups: list[float]
    wall_times: list[float]
    gpu_times: list[float]
    best_speedup: float
    worst_speedup: float
    average_speedup: float
    dominant_bottleneck: str
    representative_variants: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "record_count": self.record_count,
            "speedups": self.speedups,
            "wall_times": self.wall_times,
            "gpu_times": self.gpu_times,
            "best_speedup": self.best_speedup,
            "worst_speedup": self.worst_speedup,
            "average_speedup": self.average_speedup,
            "dominant_bottleneck": self.dominant_bottleneck,
            "representative_variants": self.representative_variants,
        }


def configure_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("agent_orchestration")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def load_results(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_operator(records: list[dict[str, Any]], operator: str) -> OperatorSummary | None:
    relevant = [record for record in records if record.get("benchmark") == f"{operator}_scaling"]
    if not relevant:
        return None

    speedups = [float(record["speedup_vs_torch"]) for record in relevant if "speedup_vs_torch" in record]
    if operator == "rmsnorm":
        wall_times = [float(record["metal_wall_mean_ms"]) for record in relevant]
        gpu_times = [float(record["metal_gpu_mean_ms"]) for record in relevant]
        variants = sorted({record.get("variant_name", "unknown") for record in relevant})
    else:
        best_by_batch = {}
        for record in relevant:
            batch_size = int(record["batch_size"])
            current = best_by_batch.get(batch_size)
            if current is None or float(record["metal_wall_mean_ms"]) < float(current["metal_wall_mean_ms"]):
                best_by_batch[batch_size] = record
        wall_times = [float(record["metal_wall_mean_ms"]) for record in best_by_batch.values()]
        gpu_times = [float(record["metal_gpu_mean_ms"]) for record in best_by_batch.values()]
        speedups = [float(record["speedup_vs_torch"]) for record in best_by_batch.values()]
        variants = sorted({record.get("variant_name", "unknown") for record in relevant})

    dominant_bottleneck = "command-buffer overhead"
    if gpu_times and wall_times and mean(gpu_times) / mean(wall_times) > 0.2:
        dominant_bottleneck = "kernel math plus dispatch overhead"
    if operator == "softmax":
        dominant_bottleneck = "reduction plus launch overhead"

    return OperatorSummary(
        operator=operator,
        record_count=len(relevant),
        speedups=speedups,
        wall_times=wall_times,
        gpu_times=gpu_times,
        best_speedup=max(speedups),
        worst_speedup=min(speedups),
        average_speedup=mean(speedups),
        dominant_bottleneck=dominant_bottleneck,
        representative_variants=variants,
    )


def identify_weakest_operator(records: list[dict[str, Any]]) -> OperatorSummary:
    candidates = [
        summarize_operator(records, "rmsnorm"),
        summarize_operator(records, "softmax"),
    ]
    available = [candidate for candidate in candidates if candidate is not None]
    if not available:
        raise RuntimeError("No operator scaling records found in results/results.jsonl.")
    return min(available, key=lambda item: item.average_speedup)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_report(path: Path, plan: dict[str, Any]) -> None:
    weakest = plan["weakest_operator"]
    lines = [
        "# Milestone 6 Agents Report",
        "",
        "## Why Explicit Agent Decomposition Matters",
        "",
        "- Kernel optimization needs different kinds of judgment: benchmark interpretation, low-level kernel design, research synthesis, and iteration planning.",
        "- Splitting those responsibilities into named agents makes the optimization loop auditable instead of hiding everything inside one benchmark script.",
        "- It also makes failure modes easier to localize: if the plan is weak, we can inspect whether the critic, the engineer, or the benchmark context was the limiting factor.",
        "",
        "## Context Engineering",
        "",
        "- Each agent receives only the benchmark slice it needs rather than the full project state.",
        "- Prompt templates in `agents/prompts/` make the intended role and output format explicit, which is the same surface we would later send to a real LLM call.",
        "- Structured outputs keep the orchestration deterministic today while preserving a clean interface for future model-backed reasoning.",
        "",
        "## Iterative Optimization Architecture",
        "",
        f"- The orchestrator loaded existing results, identified `{weakest}` as the weakest operator, and routed that context through the critic, research agent, benchmark methodologist, and kernel engineer.",
        "- The performance critic turns raw latency data into a bottleneck diagnosis.",
        "- The research agent maps that diagnosis to Metal-specific optimization ideas.",
        "- The kernel engineer converts those ideas into concrete implementation tasks and target files.",
        "- The resulting `results/optimization_plan.json` is lightweight, deterministic, and ready to be replaced later with real Codex/OpenAI API-backed agent calls.",
        "",
        "## Current Outcome",
        "",
        f"- Weakest operator: `{weakest}`",
        f"- Critic summary: {plan['critic']['reasoning_summary']}",
        f"- Engineer summary: {plan['kernel_engineer']['reasoning_summary']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> dict[str, Any]:
    logger = configure_logging()
    logger.info("orchestrator_start")

    records = load_results(RESULTS_PATH)
    weakest = identify_weakest_operator(records)
    logger.info("weakest_operator_identified | %s", json.dumps(weakest.to_dict(), sort_keys=True))

    research_agent = ResearchAgent(logger)
    benchmark_methodologist = BenchmarkMethodologist(logger)
    performance_critic = PerformanceCritic(logger)
    kernel_engineer = KernelEngineer(logger)

    critic_response = performance_critic.analyze(weakest.to_dict())
    research_response = research_agent.analyze(weakest.to_dict())
    operator_records = [record for record in records if record.get("benchmark") == f"{weakest.operator}_scaling"]
    benchmark_response = benchmark_methodologist.review(operator_records)
    engineer_response = kernel_engineer.propose(
        critic_output=critic_response.structured_output,
        research_output=research_response.structured_output,
    )

    plan = {
        "framework_version": "milestone6",
        "data_source": str(RESULTS_PATH.relative_to(ROOT)),
        "weakest_operator": weakest.operator,
        "operator_summary": weakest.to_dict(),
        "critic": critic_response.to_dict(),
        "research": research_response.to_dict(),
        "benchmark_methodologist": benchmark_response.to_dict(),
        "kernel_engineer": engineer_response.to_dict(),
        "optimization_plan": {
            "focus_operator": weakest.operator,
            "next_actions": engineer_response.structured_output["proposals"],
            "validation_requirements": benchmark_response.structured_output["next_checks"],
        },
    }

    write_json(PLAN_PATH, plan)
    write_report(REPORT_PATH, plan)
    logger.info("orchestrator_complete | %s", json.dumps({"plan_path": str(PLAN_PATH)}, sort_keys=True))
    return plan


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
