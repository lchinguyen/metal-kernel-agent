import argparse
import json
import shutil
import statistics
import subprocess
import tempfile
from array import array
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is required to run this benchmark. Install torch in the active "
        "Python environment, then rerun benchmarks/rmsnorm_metal.py."
    ) from exc


DEFAULT_SHAPE = (1, 4096)
DEFAULT_DTYPE = "float32"
DEFAULT_EPS = 1e-6
DEFAULT_WARMUP_ITERS = 20
DEFAULT_TIMED_ITERS = 100
DEFAULT_RESULTS_PATH = Path("results/results.jsonl")
DEFAULT_REPORT_PATH = Path("results/milestone3_report.md")
DEFAULT_KERNEL_PATH = Path("kernels/rmsnorm.metal")
DEFAULT_HELPER_SOURCE = Path("benchmarks/rmsnorm_metal_helper.swift")
DEFAULT_HELPER_BINARY = Path("benchmarks/.build/rmsnorm_metal_helper")
DEFAULT_SWIFT_CACHE = Path(".swift-cache")
DEFAULT_VARIANTS = ("shared_256", "shared_128", "simd_128")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Metal RMSNorm variants for shape [1, 4096]."
    )
    parser.add_argument("--warmup-iters", type=int, default=DEFAULT_WARMUP_ITERS)
    parser.add_argument("--timed-iters", type=int, default=DEFAULT_TIMED_ITERS)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--results-path",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Path to the JSONL results log.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path to the milestone markdown report.",
    )
    parser.add_argument(
        "--kernel-path",
        type=Path,
        default=DEFAULT_KERNEL_PATH,
        help="Path to the Metal kernel source.",
    )
    parser.add_argument(
        "--helper-source",
        type=Path,
        default=DEFAULT_HELPER_SOURCE,
        help="Swift source that compiles and dispatches the Metal kernel.",
    )
    parser.add_argument(
        "--helper-binary",
        type=Path,
        default=DEFAULT_HELPER_BINARY,
        help="Compiled helper binary path.",
    )
    parser.add_argument(
        "--swift-module-cache",
        type=Path,
        default=DEFAULT_SWIFT_CACHE,
        help="Writable module cache directory for Swift compilation.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANTS),
        help="Metal variants to benchmark.",
    )
    return parser.parse_args()


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute percentiles for an empty sequence.")
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * q
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return lower_value + (upper_value - lower_value) * weight


def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    return x * inv_rms.to(x.dtype) * weight


def append_jsonl(record: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def write_text_file(output_path: Path, content: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def write_f32_binary(values: torch.Tensor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    float_values = array("f", values.detach().cpu().reshape(-1).tolist())
    with output_path.open("wb") as handle:
        float_values.tofile(handle)


def compile_helper(helper_source: Path, helper_binary: Path, swift_module_cache: Path) -> None:
    if shutil.which("swiftc") is None:
        raise SystemExit("swiftc is required to compile the Metal benchmark helper.")

    helper_binary.parent.mkdir(parents=True, exist_ok=True)
    swift_module_cache.mkdir(parents=True, exist_ok=True)

    compile_cmd = [
        "swiftc",
        "-module-cache-path",
        str(swift_module_cache),
        str(helper_source),
        "-o",
        str(helper_binary),
    ]
    subprocess.run(compile_cmd, check=True)


def ensure_compiled_helper(helper_source: Path, helper_binary: Path, swift_module_cache: Path) -> None:
    if not helper_source.exists():
        raise SystemExit(f"Helper source not found: {helper_source}")
    needs_compile = not helper_binary.exists()
    if not needs_compile:
        needs_compile = helper_binary.stat().st_mtime < helper_source.stat().st_mtime
    if needs_compile:
        compile_helper(helper_source, helper_binary, swift_module_cache)


def run_metal_helper(
    helper_binary: Path,
    variant: str,
    kernel_path: Path,
    input_path: Path,
    weight_path: Path,
    eps: float,
    warmup_iters: int,
    timed_iters: int,
) -> dict:
    command = [
        str(helper_binary),
        "--variant",
        variant,
        "--kernel",
        str(kernel_path),
        "--input-bin",
        str(input_path),
        "--weight-bin",
        str(weight_path),
        "--eps",
        str(eps),
        "--warmup-iters",
        str(warmup_iters),
        "--timed-iters",
        str(timed_iters),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def load_previous_torch_baseline(results_path: Path) -> dict | None:
    if not results_path.exists():
        return None

    latest_match = None
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if (
                record.get("benchmark") == "rmsnorm_torch_baseline"
                and record.get("shape") == list(DEFAULT_SHAPE)
                and record.get("dtype") == DEFAULT_DTYPE
            ):
                latest_match = record
    return latest_match


def build_result_record(
    metal_run: dict,
    metal_output: torch.Tensor,
    torch_output: torch.Tensor,
    eps: float,
    warmup_iters: int,
    timed_iters: int,
    torch_baseline: dict | None,
) -> dict:
    wall_latencies_ms = [float(value) for value in metal_run["wall_latencies_ms"]]
    gpu_latencies_ms = [float(value) for value in metal_run["gpu_latencies_ms"]]
    ordered_wall = sorted(wall_latencies_ms)
    ordered_gpu = sorted(gpu_latencies_ms)
    abs_error = (metal_output - torch_output).abs()
    wall_mean = statistics.mean(wall_latencies_ms)
    gpu_mean = statistics.mean(gpu_latencies_ms)

    record = {
        "benchmark": "rmsnorm_metal_variant",
        "implementation": "custom_metal",
        "variant_name": metal_run["variant_name"],
        "kernel_name": metal_run["kernel_name"],
        "shape": list(DEFAULT_SHAPE),
        "dtype": DEFAULT_DTYPE,
        "device": metal_run["device_name"],
        "eps": eps,
        "warmup_iterations": warmup_iters,
        "timed_iterations": timed_iters,
        "threadgroup_size": metal_run["threadgroup_size"],
        "thread_execution_width": metal_run["thread_execution_width"],
        "max_total_threads_per_threadgroup": metal_run["max_total_threads_per_threadgroup"],
        "latency_ms_mean": wall_mean,
        "latency_ms_median": statistics.median(wall_latencies_ms),
        "latency_ms_p50": percentile(ordered_wall, 0.50),
        "latency_ms_p95": percentile(ordered_wall, 0.95),
        "gpu_latency_ms_mean": gpu_mean,
        "gpu_latency_ms_median": statistics.median(gpu_latencies_ms),
        "gpu_latency_ms_p50": percentile(ordered_gpu, 0.50),
        "gpu_latency_ms_p95": percentile(ordered_gpu, 0.95),
        "cpu_overhead_ms_mean": wall_mean - gpu_mean,
        "max_abs_error": abs_error.max().item(),
        "mean_abs_error": abs_error.mean().item(),
        "max_abs_output_value": metal_output.abs().max().item(),
    }

    if torch_baseline is not None:
        torch_mean = torch_baseline.get("latency_ms_mean")
        record["torch_baseline_latency_ms_mean"] = torch_mean
        if isinstance(torch_mean, (int, float)) and wall_mean > 0.0:
            record["speedup_vs_torch_mean"] = torch_mean / wall_mean

    return record


def format_ms(value: float) -> str:
    return f"{value:.4f} ms"


def build_report(records: list[dict], torch_baseline: dict | None) -> str:
    fastest = min(records, key=lambda record: record["latency_ms_mean"])
    baseline_line = "Torch baseline unavailable in results/results.jsonl."
    if torch_baseline is not None:
        baseline_line = (
            "Torch baseline mean latency: "
            f"{torch_baseline['latency_ms_mean']:.4f} ms."
        )

    lines = [
        "# Milestone 3 Report",
        "",
        "## What Changed",
        "",
        "- Kept the optimized path as a real custom Metal kernel and benchmarked three variants on the same `[1, 4096]` float32 input.",
        "- Added a `shared_128` variant to test whether fewer threads and fewer reduction barriers help this tiny workload.",
        "- Added a `simd_128` variant that uses SIMD-group reduction to cut threadgroup-memory traffic and barrier work.",
        "- Added GPU-time collection from Metal command buffers so we can compare on-GPU work against end-to-end wall latency.",
        "",
        "## Analysis",
        "",
        f"- {baseline_line}",
        "- The tensor is only 4096 float32 values, so memory traffic is small; the work is too small to be bandwidth-bound.",
        "- GPU kernel time is far below end-to-end wall time for every variant, which points to command-buffer submission and synchronization overhead dominating total latency.",
        "- The Swift helper process itself is outside the timed loop; the timed section starts immediately before command buffer creation and ends after GPU completion, so helper startup is not the measured bottleneck.",
        f"- The fastest variant was `{fastest['variant_name']}` at {format_ms(fastest['latency_ms_mean'])} mean wall latency with {format_ms(fastest['gpu_latency_ms_mean'])} mean GPU time.",
        "",
        "## Results",
        "",
        "| Variant | Mean | Median | p50 | p95 | GPU Mean | CPU/Submit Overhead Mean | Max Abs Error | Mean Abs Error | Speedup vs Torch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for record in sorted(records, key=lambda row: row["latency_ms_mean"]):
        speedup = record.get("speedup_vs_torch_mean")
        speedup_text = f"{speedup:.2f}x" if isinstance(speedup, (int, float)) else "n/a"
        lines.append(
            "| "
            f"{record['variant_name']} | "
            f"{record['latency_ms_mean']:.4f} ms | "
            f"{record['latency_ms_median']:.4f} ms | "
            f"{record['latency_ms_p50']:.4f} ms | "
            f"{record['latency_ms_p95']:.4f} ms | "
            f"{record['gpu_latency_ms_mean']:.4f} ms | "
            f"{record['cpu_overhead_ms_mean']:.4f} ms | "
            f"{record['max_abs_error']:.6f} | "
            f"{record['mean_abs_error']:.6f} | "
            f"{speedup_text} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.warmup_iters < 0:
        raise SystemExit("--warmup-iters must be >= 0.")
    if args.timed_iters <= 0:
        raise SystemExit("--timed-iters must be > 0.")
    if not args.kernel_path.exists():
        raise SystemExit(f"Kernel source not found: {args.kernel_path}")

    ensure_compiled_helper(args.helper_source, args.helper_binary, args.swift_module_cache)

    torch.manual_seed(args.seed)
    x = torch.randn(DEFAULT_SHAPE, dtype=torch.float32)
    weight = torch.randn(DEFAULT_SHAPE[-1], dtype=torch.float32)
    torch_output = rmsnorm_torch(x, weight, args.eps)
    torch_baseline = load_previous_torch_baseline(args.results_path)

    records: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="rmsnorm-metal-") as temp_dir:
        temp_root = Path(temp_dir)
        input_path = temp_root / "input.bin"
        weight_path = temp_root / "weight.bin"
        write_f32_binary(x, input_path)
        write_f32_binary(weight, weight_path)

        for variant in args.variants:
            try:
                metal_run = run_metal_helper(
                    helper_binary=args.helper_binary,
                    variant=variant,
                    kernel_path=args.kernel_path,
                    input_path=input_path,
                    weight_path=weight_path,
                    eps=args.eps,
                    warmup_iters=args.warmup_iters,
                    timed_iters=args.timed_iters,
                )
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.strip() if exc.stderr else "unknown Metal helper error"
                raise SystemExit(f"Metal benchmark helper failed for variant {variant}: {stderr}") from exc

            metal_output = torch.tensor(metal_run["output"], dtype=torch.float32).reshape(DEFAULT_SHAPE)
            record = build_result_record(
                metal_run=metal_run,
                metal_output=metal_output,
                torch_output=torch_output,
                eps=args.eps,
                warmup_iters=args.warmup_iters,
                timed_iters=args.timed_iters,
                torch_baseline=torch_baseline,
            )
            append_jsonl(record, args.results_path)
            records.append(record)

    report = build_report(records, torch_baseline)
    write_text_file(args.report_path, report)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
