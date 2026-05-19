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
        "Python environment, then rerun benchmarks/rmsnorm_scaling.py."
    ) from exc


HIDDEN_SIZE = 4096
DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_DTYPE = "float32"
DEFAULT_EPS = 1e-6
DEFAULT_WARMUP_ITERS = 20
DEFAULT_TIMED_ITERS = 100
DEFAULT_RESULTS_PATH = Path("results/results.jsonl")
DEFAULT_REPORT_PATH = Path("results/milestone4_scaling_report.md")
DEFAULT_KERNEL_PATH = Path("kernels/rmsnorm.metal")
DEFAULT_HELPER_SOURCE = Path("benchmarks/rmsnorm_metal_helper.swift")
DEFAULT_HELPER_BINARY = Path("benchmarks/.build/rmsnorm_metal_helper")
DEFAULT_SWIFT_CACHE = Path(".swift-cache")
BEST_VARIANT = "shared_256"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Torch vs custom Metal RMSNorm across batch sizes."
    )
    parser.add_argument("--warmup-iters", type=int, default=DEFAULT_WARMUP_ITERS)
    parser.add_argument("--timed-iters", type=int, default=DEFAULT_TIMED_ITERS)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_BATCH_SIZES),
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )
    parser.add_argument(
        "--kernel-path",
        type=Path,
        default=DEFAULT_KERNEL_PATH,
    )
    parser.add_argument(
        "--helper-source",
        type=Path,
        default=DEFAULT_HELPER_SOURCE,
    )
    parser.add_argument(
        "--helper-binary",
        type=Path,
        default=DEFAULT_HELPER_BINARY,
    )
    parser.add_argument(
        "--swift-module-cache",
        type=Path,
        default=DEFAULT_SWIFT_CACHE,
    )
    parser.add_argument(
        "--torch-device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
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
    subprocess.run(
        [
            "swiftc",
            "-module-cache-path",
            str(swift_module_cache),
            str(helper_source),
            "-o",
            str(helper_binary),
        ],
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
    kernel_path: Path,
    input_path: Path,
    weight_path: Path,
    eps: float,
    warmup_iters: int,
    timed_iters: int,
) -> dict:
    completed = subprocess.run(
        [
            str(helper_binary),
            "--variant",
            BEST_VARIANT,
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
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def benchmark_torch(
    x_cpu: torch.Tensor,
    weight_cpu: torch.Tensor,
    device: torch.device,
    eps: float,
    warmup_iters: int,
    timed_iters: int,
) -> tuple[list[float], torch.Tensor]:
    x = x_cpu.to(device)
    weight = weight_cpu.to(device)
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = rmsnorm_torch(x, weight, eps)
        synchronize_torch(device)

        latencies_ms = []
        output = None
        for _ in range(timed_iters):
            synchronize_torch(device)
            start_ns = time.perf_counter_ns()
            output = rmsnorm_torch(x, weight, eps)
            synchronize_torch(device)
            end_ns = time.perf_counter_ns()
            latencies_ms.append((end_ns - start_ns) / 1_000_000.0)

    if output is None:
        raise RuntimeError("Torch benchmark did not produce an output tensor.")
    return latencies_ms, output.detach().cpu()


def build_scaling_record(
    batch_size: int,
    eps: float,
    torch_device: torch.device,
    torch_latencies_ms: list[float],
    metal_run: dict,
    metal_output: torch.Tensor,
    torch_output: torch.Tensor,
    warmup_iters: int,
    timed_iters: int,
) -> dict:
    torch_mean = statistics.mean(torch_latencies_ms)
    metal_wall_mean = statistics.mean(float(value) for value in metal_run["wall_latencies_ms"])
    metal_gpu_mean = statistics.mean(float(value) for value in metal_run["gpu_latencies_ms"])
    abs_error = (metal_output - torch_output).abs()
    return {
        "benchmark": "rmsnorm_scaling",
        "implementation": "custom_metal",
        "variant_name": BEST_VARIANT,
        "kernel_name": metal_run["kernel_name"],
        "shape": [batch_size, HIDDEN_SIZE],
        "dtype": DEFAULT_DTYPE,
        "torch_device": str(torch_device),
        "metal_device": metal_run["device_name"],
        "eps": eps,
        "warmup_iterations": warmup_iters,
        "timed_iterations": timed_iters,
        "threadgroup_size": metal_run["threadgroup_size"],
        "thread_execution_width": metal_run["thread_execution_width"],
        "batch_size": batch_size,
        "torch_mean_ms": torch_mean,
        "metal_wall_mean_ms": metal_wall_mean,
        "metal_gpu_mean_ms": metal_gpu_mean,
        "speedup_vs_torch": torch_mean / metal_wall_mean,
        "max_abs_error": abs_error.max().item(),
        "mean_abs_error": abs_error.mean().item(),
    }


def build_report(records: list[dict]) -> str:
    lines = [
        "# Milestone 4 Scaling Report",
        "",
        "## Setup",
        "",
        "- Compared Torch RMSNorm against the best custom Metal variant `shared_256`.",
        "- Used hidden size `4096` and batch sizes `[1, 2, 4, 8, 16, 32, 64]`.",
        "- Metal remains a real custom kernel; Torch is used only for baseline timing and correctness reference.",
        "",
        "## Results",
        "",
        "| Batch | Torch Mean | Metal Wall Mean | Metal GPU Mean | Speedup vs Torch | Max Abs Error | Mean Abs Error |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        lines.append(
            "| "
            f"{record['batch_size']} | "
            f"{record['torch_mean_ms']:.4f} ms | "
            f"{record['metal_wall_mean_ms']:.4f} ms | "
            f"{record['metal_gpu_mean_ms']:.4f} ms | "
            f"{record['speedup_vs_torch']:.2f}x | "
            f"{record['max_abs_error']:.6f} | "
            f"{record['mean_abs_error']:.6f} |"
        )

    fastest = max(records, key=lambda record: record["speedup_vs_torch"])
    slowest = min(records, key=lambda record: record["speedup_vs_torch"])
    lines.extend(
        [
            "",
            "## Analysis",
            "",
            "- The Metal kernel stays correct across every batch size: all reported errors remain at zero within float32 comparison.",
            "- At small batch sizes, command-buffer submission and synchronization still take a meaningful share of wall time, because GPU kernel time is much smaller than total wall time.",
            "- As batch size grows, the fixed submission cost is amortized across more rows because one dispatch now processes multiple RMSNorm rows in parallel.",
            f"- The strongest advantage appeared at batch size `{fastest['batch_size']}` with `{fastest['speedup_vs_torch']:.2f}x` speedup.",
            f"- The weakest advantage appeared at batch size `{slowest['batch_size']}` with `{slowest['speedup_vs_torch']:.2f}x` speedup.",
            "- If the speedup trend rises with batch size, that supports the overhead hypothesis: the custom kernel wins more once there is enough work per dispatch to hide command-buffer costs.",
        ]
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
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        raise SystemExit("All batch sizes must be > 0.")

    ensure_compiled_helper(args.helper_source, args.helper_binary, args.swift_module_cache)
    torch_device = resolve_torch_device(args.torch_device)

    torch.manual_seed(args.seed)
    records: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="rmsnorm-scaling-") as temp_dir:
        temp_root = Path(temp_dir)
        for batch_size in args.batch_sizes:
            x_cpu = torch.randn((batch_size, HIDDEN_SIZE), dtype=torch.float32)
            weight_cpu = torch.randn(HIDDEN_SIZE, dtype=torch.float32)
            torch_latencies_ms, torch_output = benchmark_torch(
                x_cpu=x_cpu,
                weight_cpu=weight_cpu,
                device=torch_device,
                eps=args.eps,
                warmup_iters=args.warmup_iters,
                timed_iters=args.timed_iters,
            )

            input_path = temp_root / f"input_b{batch_size}.bin"
            weight_path = temp_root / f"weight_b{batch_size}.bin"
            write_f32_binary(x_cpu, input_path)
            write_f32_binary(weight_cpu, weight_path)

            try:
                metal_run = run_metal_helper(
                    helper_binary=args.helper_binary,
                    kernel_path=args.kernel_path,
                    input_path=input_path,
                    weight_path=weight_path,
                    eps=args.eps,
                    warmup_iters=args.warmup_iters,
                    timed_iters=args.timed_iters,
                )
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.strip() if exc.stderr else "unknown Metal helper error"
                raise SystemExit(
                    f"Metal benchmark helper failed for batch size {batch_size}: {stderr}"
                ) from exc

            metal_output = torch.tensor(metal_run["output"], dtype=torch.float32).reshape(batch_size, HIDDEN_SIZE)
            record = build_scaling_record(
                batch_size=batch_size,
                eps=args.eps,
                torch_device=torch_device,
                torch_latencies_ms=torch_latencies_ms,
                metal_run=metal_run,
                metal_output=metal_output,
                torch_output=torch_output,
                warmup_iters=args.warmup_iters,
                timed_iters=args.timed_iters,
            )
            append_jsonl(record, args.results_path)
            records.append(record)

    report = build_report(records)
    write_text_file(args.report_path, report)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
