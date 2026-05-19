import argparse
import json
import shutil
import statistics
import subprocess
import tempfile
import time
from array import array
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is required to run this benchmark. Install torch in the active "
        "Python environment, then rerun benchmarks/fused_rmsnorm_residual.py."
    ) from exc


HIDDEN_SIZE = 4096
DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_EPS = 1e-6
DEFAULT_WARMUP_ITERS = 20
DEFAULT_TIMED_ITERS = 100
DEFAULT_RESULTS_PATH = Path("results/results.jsonl")
DEFAULT_REPORT_PATH = Path("results/milestone9_fusion_report.md")
DEFAULT_KERNEL_PATH = Path("kernels/rmsnorm.metal")
DEFAULT_HELPER_SOURCE = Path("benchmarks/fused_rmsnorm_residual_helper.swift")
DEFAULT_HELPER_BINARY = Path("benchmarks/.build/fused_rmsnorm_residual_helper")
DEFAULT_SWIFT_CACHE = Path(".swift-cache")
MODES = ("standalone_rmsnorm", "separate_rmsnorm_residual", "fused_rmsnorm_residual")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark fused RMSNorm + residual-add Metal kernel across batch sizes."
    )
    parser.add_argument("--warmup-iters", type=int, default=DEFAULT_WARMUP_ITERS)
    parser.add_argument("--timed-iters", type=int, default=DEFAULT_TIMED_ITERS)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=list(DEFAULT_BATCH_SIZES))
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--kernel-path", type=Path, default=DEFAULT_KERNEL_PATH)
    parser.add_argument("--helper-source", type=Path, default=DEFAULT_HELPER_SOURCE)
    parser.add_argument("--helper-binary", type=Path, default=DEFAULT_HELPER_BINARY)
    parser.add_argument("--swift-module-cache", type=Path, default=DEFAULT_SWIFT_CACHE)
    parser.add_argument("--torch-device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser.parse_args()


def resolve_torch_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("Requested Torch device 'mps' is not available.")
        return torch.device("mps")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("Requested Torch device 'cuda' is not available.")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def synchronize_torch(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def percentile(sorted_values: list[float], q: float) -> float:
    position = (len(sorted_values) - 1) * q
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index
    return sorted_values[lower_index] + (sorted_values[upper_index] - sorted_values[lower_index]) * weight


def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    return x * inv_rms.to(x.dtype) * weight


def fused_torch(x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor, eps: float) -> torch.Tensor:
    return residual + rmsnorm_torch(x, weight, eps)


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
    subprocess.run(
        ["swiftc", "-module-cache-path", str(swift_module_cache), str(helper_source), "-o", str(helper_binary)],
        check=True,
    )


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
    mode: str,
    kernel_path: Path,
    input_path: Path,
    weight_path: Path,
    residual_path: Path,
    eps: float,
    warmup_iters: int,
    timed_iters: int,
) -> dict:
    completed = subprocess.run(
        [
            str(helper_binary),
            "--mode",
            mode,
            "--kernel",
            str(kernel_path),
            "--input-bin",
            str(input_path),
            "--weight-bin",
            str(weight_path),
            "--residual-bin",
            str(residual_path),
            "--eps",
            str(eps),
            "--warmup-iters",
            str(warmup_iters),
            "--timed-iters",
            str(timed_iters),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def benchmark_torch(
    x_cpu: torch.Tensor,
    weight_cpu: torch.Tensor,
    residual_cpu: torch.Tensor,
    device: torch.device,
    eps: float,
    warmup_iters: int,
    timed_iters: int,
) -> tuple[list[float], torch.Tensor, torch.Tensor]:
    x = x_cpu.to(device)
    weight = weight_cpu.to(device)
    residual = residual_cpu.to(device)
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = fused_torch(x, weight, residual, eps)
        synchronize_torch(device)
        latencies_ms = []
        fused_output = None
        for _ in range(timed_iters):
            synchronize_torch(device)
            start_ns = time.perf_counter_ns()
            fused_output = fused_torch(x, weight, residual, eps)
            synchronize_torch(device)
            end_ns = time.perf_counter_ns()
            latencies_ms.append((end_ns - start_ns) / 1_000_000.0)
    if fused_output is None:
        raise RuntimeError("Torch fused benchmark did not produce an output tensor.")
    standalone_output = rmsnorm_torch(x_cpu, weight_cpu, eps)
    return latencies_ms, standalone_output.detach().cpu(), fused_output.detach().cpu()


def build_record(
    batch_size: int,
    mode: str,
    torch_device: torch.device,
    torch_latencies_ms: list[float],
    metal_run: dict,
    metal_output: torch.Tensor,
    reference_output: torch.Tensor,
    eps: float,
    unfused_mean_ms: float | None,
    warmup_iters: int,
    timed_iters: int,
) -> dict:
    wall_latencies = [float(value) for value in metal_run["wall_latencies_ms"]]
    gpu_latencies = [float(value) for value in metal_run["gpu_latencies_ms"]]
    ordered_wall = sorted(wall_latencies)
    abs_error = (metal_output - reference_output).abs()
    wall_mean = statistics.mean(wall_latencies)
    record = {
        "benchmark": "fused_rmsnorm_residual",
        "implementation": "custom_metal",
        "mode": mode,
        "primary_kernel_name": metal_run["primary_kernel_name"],
        "secondary_kernel_name": metal_run.get("secondary_kernel_name"),
        "shape": [batch_size, HIDDEN_SIZE],
        "dtype": "float32",
        "torch_device": str(torch_device),
        "metal_device": metal_run["device_name"],
        "eps": eps,
        "warmup_iterations": warmup_iters,
        "timed_iterations": timed_iters,
        "threadgroup_size": metal_run["threadgroup_size"],
        "thread_execution_width": metal_run["thread_execution_width"],
        "batch_size": batch_size,
        "torch_mean_ms": statistics.mean(torch_latencies_ms),
        "mean_latency_ms": wall_mean,
        "p50_latency_ms": percentile(ordered_wall, 0.50),
        "p95_latency_ms": percentile(ordered_wall, 0.95),
        "gpu_mean_ms": statistics.mean(gpu_latencies),
        "speedup_vs_torch": statistics.mean(torch_latencies_ms) / wall_mean,
        "max_abs_error": abs_error.max().item(),
        "mean_abs_error": abs_error.mean().item(),
    }
    if unfused_mean_ms is not None and wall_mean > 0.0:
        record["speedup_vs_unfused"] = unfused_mean_ms / wall_mean
    return record


def build_report(records: list[dict]) -> str:
    separate_by_batch = {
        record["batch_size"]: record
        for record in records
        if record["mode"] == "separate_rmsnorm_residual"
    }
    fused_by_batch = {
        record["batch_size"]: record
        for record in records
        if record["mode"] == "fused_rmsnorm_residual"
    }
    standalone_by_batch = {
        record["batch_size"]: record
        for record in records
        if record["mode"] == "standalone_rmsnorm"
    }

    lines = [
        "# Milestone 9 Fusion Report",
        "",
        "## Setup",
        "",
        "- Benchmarked standalone RMSNorm, separate RMSNorm plus residual add, and a fused RMSNorm plus residual add Metal kernel.",
        "- Used batch sizes `[1, 2, 4, 8, 16, 32, 64]`, hidden size `4096`, and float32.",
        "- Torch was used only as the correctness and fused-operation baseline.",
        "",
        "## Fused vs Unfused Results",
        "",
        "| Batch | Standalone RMSNorm Mean | Separate RMSNorm+Residual Mean | Fused Mean | Fused GPU Mean | Speedup vs Torch | Speedup vs Unfused | Max Abs Error | Mean Abs Error |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for batch_size in sorted(fused_by_batch):
        standalone = standalone_by_batch[batch_size]
        separate = separate_by_batch[batch_size]
        fused = fused_by_batch[batch_size]
        lines.append(
            "| "
            f"{batch_size} | "
            f"{standalone['mean_latency_ms']:.4f} ms | "
            f"{separate['mean_latency_ms']:.4f} ms | "
            f"{fused['mean_latency_ms']:.4f} ms | "
            f"{fused['gpu_mean_ms']:.4f} ms | "
            f"{fused['speedup_vs_torch']:.2f}x | "
            f"{fused.get('speedup_vs_unfused', 0.0):.2f}x | "
            f"{fused['max_abs_error']:.6f} | "
            f"{fused['mean_abs_error']:.6f} |"
        )

    best_fused = max(fused_by_batch.values(), key=lambda record: record["speedup_vs_unfused"])
    lines.extend(
        [
            "",
            "## Why Fusion Matters",
            "",
            "- Fusion reduces memory traffic by avoiding an intermediate RMSNorm output write followed by a separate residual-read and output-write pass.",
            "- Fusion reduces dispatch overhead because one command-encoded kernel replaces a two-kernel sequence for the same transformer subgraph step.",
            "- In transformer inference, normalization and residual paths are frequent and latency-sensitive, so even modest per-layer savings can compound across many layers and tokens.",
            "",
            "## Analysis",
            "",
            "- The fused kernel is correctness-first: it keeps the existing fp32 RMSNorm accumulation path and adds the residual writeback in the same pass.",
            "- Standalone RMSNorm provides the lower bound for normalization-only work, while the separate implementation shows the cost of keeping residual addition as another dispatch.",
            f"- The strongest fused advantage over the unfused Metal path appeared at batch `{best_fused['batch_size']}` with `{best_fused['speedup_vs_unfused']:.2f}x` speedup.",
            "- If fused wall time stays materially above fused GPU time, the remaining opportunity is still launch overhead and kernel packing rather than pure arithmetic cost.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.warmup_iters < 0:
        raise SystemExit("--warmup-iters must be >= 0.")
    if args.timed_iters <= 0:
        raise SystemExit("--timed-iters must be > 0.")
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        raise SystemExit("All batch sizes must be > 0.")
    if not args.kernel_path.exists():
        raise SystemExit(f"Kernel source not found: {args.kernel_path}")

    ensure_compiled_helper(args.helper_source, args.helper_binary, args.swift_module_cache)
    torch_device = resolve_torch_device(args.torch_device)

    torch.manual_seed(args.seed)
    records: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="fused-rmsnorm-residual-") as temp_dir:
        temp_root = Path(temp_dir)
        for batch_size in args.batch_sizes:
            x_cpu = torch.randn((batch_size, HIDDEN_SIZE), dtype=torch.float32)
            weight_cpu = torch.randn(HIDDEN_SIZE, dtype=torch.float32)
            residual_cpu = torch.randn((batch_size, HIDDEN_SIZE), dtype=torch.float32)

            torch_latencies_ms, standalone_reference, fused_reference = benchmark_torch(
                x_cpu=x_cpu,
                weight_cpu=weight_cpu,
                residual_cpu=residual_cpu,
                device=torch_device,
                eps=args.eps,
                warmup_iters=args.warmup_iters,
                timed_iters=args.timed_iters,
            )

            input_path = temp_root / f"input_b{batch_size}.bin"
            weight_path = temp_root / f"weight_b{batch_size}.bin"
            residual_path = temp_root / f"residual_b{batch_size}.bin"
            write_f32_binary(x_cpu, input_path)
            write_f32_binary(weight_cpu, weight_path)
            write_f32_binary(residual_cpu, residual_path)

            separate_mean_ms = None
            batch_records = []
            for mode in MODES:
                try:
                    metal_run = run_metal_helper(
                        helper_binary=args.helper_binary,
                        mode=mode,
                        kernel_path=args.kernel_path,
                        input_path=input_path,
                        weight_path=weight_path,
                        residual_path=residual_path,
                        eps=args.eps,
                        warmup_iters=args.warmup_iters,
                        timed_iters=args.timed_iters,
                    )
                except subprocess.CalledProcessError as exc:
                    stderr = exc.stderr.strip() if exc.stderr else "unknown Metal helper error"
                    raise SystemExit(
                        f"Metal benchmark helper failed for batch size {batch_size}, mode {mode}: {stderr}"
                    ) from exc

                reference_output = standalone_reference if mode == "standalone_rmsnorm" else fused_reference
                metal_output = torch.tensor(metal_run["output"], dtype=torch.float32).reshape(batch_size, HIDDEN_SIZE)
                record = build_record(
                    batch_size=batch_size,
                    mode=mode,
                    torch_device=torch_device,
                    torch_latencies_ms=torch_latencies_ms,
                    metal_run=metal_run,
                    metal_output=metal_output,
                    reference_output=reference_output,
                    eps=args.eps,
                    unfused_mean_ms=separate_mean_ms if mode == "fused_rmsnorm_residual" else None,
                    warmup_iters=args.warmup_iters,
                    timed_iters=args.timed_iters,
                )
                if mode == "separate_rmsnorm_residual":
                    separate_mean_ms = record["mean_latency_ms"]
                batch_records.append(record)

            # Rebuild fused record now that separate mean is known.
            normalized_records = []
            for record in batch_records:
                if record["mode"] == "fused_rmsnorm_residual":
                    record["speedup_vs_unfused"] = separate_mean_ms / record["mean_latency_ms"]
                normalized_records.append(record)

            for record in normalized_records:
                append_jsonl(record, args.results_path)
                records.append(record)

    report = build_report(records)
    write_text_file(args.report_path, report)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
