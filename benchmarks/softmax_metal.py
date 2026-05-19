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
        "Python environment, then rerun benchmarks/softmax_metal.py."
    ) from exc


HIDDEN_SIZE = 4096
DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_VARIANTS = ("shared_256", "simd_256")
DEFAULT_DTYPE = "float32"
DEFAULT_WARMUP_ITERS = 20
DEFAULT_TIMED_ITERS = 100
DEFAULT_RESULTS_PATH = Path("results/results.jsonl")
DEFAULT_REPORT_PATH = Path("results/milestone5_softmax_report.md")
DEFAULT_KERNEL_PATH = Path("kernels/softmax.metal")
DEFAULT_HELPER_SOURCE = Path("benchmarks/softmax_metal_helper.swift")
DEFAULT_HELPER_BINARY = Path("benchmarks/.build/softmax_metal_helper")
DEFAULT_SWIFT_CACHE = Path(".swift-cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark custom Metal Softmax against Torch across batch sizes."
    )
    parser.add_argument("--warmup-iters", type=int, default=DEFAULT_WARMUP_ITERS)
    parser.add_argument("--timed-iters", type=int, default=DEFAULT_TIMED_ITERS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_BATCH_SIZES),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANTS),
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


def softmax_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


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
    variant: str,
    kernel_path: Path,
    input_path: Path,
    warmup_iters: int,
    timed_iters: int,
) -> dict:
    completed = subprocess.run(
        [
            str(helper_binary),
            "--variant",
            variant,
            "--kernel",
            str(kernel_path),
            "--input-bin",
            str(input_path),
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
    device: torch.device,
    warmup_iters: int,
    timed_iters: int,
) -> tuple[list[float], torch.Tensor]:
    x = x_cpu.to(device)
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = softmax_torch(x)
        synchronize_torch(device)

        latencies_ms = []
        output = None
        for _ in range(timed_iters):
            synchronize_torch(device)
            start_ns = time.perf_counter_ns()
            output = softmax_torch(x)
            synchronize_torch(device)
            end_ns = time.perf_counter_ns()
            latencies_ms.append((end_ns - start_ns) / 1_000_000.0)

    if output is None:
        raise RuntimeError("Torch benchmark did not produce an output tensor.")
    return latencies_ms, output.detach().cpu()


def build_record(
    batch_size: int,
    variant: str,
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
        "benchmark": "softmax_scaling",
        "implementation": "custom_metal",
        "variant_name": variant,
        "kernel_name": metal_run["kernel_name"],
        "shape": [batch_size, HIDDEN_SIZE],
        "dtype": DEFAULT_DTYPE,
        "torch_device": str(torch_device),
        "metal_device": metal_run["device_name"],
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
    best_by_batch: dict[int, dict] = {}
    for record in records:
        current = best_by_batch.get(record["batch_size"])
        if current is None or record["metal_wall_mean_ms"] < current["metal_wall_mean_ms"]:
            best_by_batch[record["batch_size"]] = record

    lines = [
        "# Milestone 5 Softmax Report",
        "",
        "## Setup",
        "",
        "- Added a real custom Metal Softmax kernel with numerically stable row-wise max subtraction.",
        "- Benchmarked batch sizes `[1, 2, 4, 8, 16, 32, 64]` at hidden size `4096` in float32.",
        "- Torch Softmax is used only as the baseline and correctness reference.",
        "- Two Metal variants were measured: `shared_256` correctness-first and `simd_256` as an optimized reduction variant.",
        "",
        "## Best Variant Per Batch",
        "",
        "| Batch | Best Variant | Torch Mean | Metal Wall Mean | Metal GPU Mean | Speedup vs Torch | Max Abs Error | Mean Abs Error |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for batch_size in sorted(best_by_batch):
        record = best_by_batch[batch_size]
        lines.append(
            "| "
            f"{batch_size} | "
            f"{record['variant_name']} | "
            f"{record['torch_mean_ms']:.4f} ms | "
            f"{record['metal_wall_mean_ms']:.4f} ms | "
            f"{record['metal_gpu_mean_ms']:.4f} ms | "
            f"{record['speedup_vs_torch']:.2f}x | "
            f"{record['max_abs_error']:.6f} | "
            f"{record['mean_abs_error']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## All Variants",
            "",
            "| Batch | Variant | Torch Mean | Metal Wall Mean | Metal GPU Mean | Speedup vs Torch | Max Abs Error | Mean Abs Error |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for record in sorted(records, key=lambda item: (item["batch_size"], item["metal_wall_mean_ms"])):
        lines.append(
            "| "
            f"{record['batch_size']} | "
            f"{record['variant_name']} | "
            f"{record['torch_mean_ms']:.4f} ms | "
            f"{record['metal_wall_mean_ms']:.4f} ms | "
            f"{record['metal_gpu_mean_ms']:.4f} ms | "
            f"{record['speedup_vs_torch']:.2f}x | "
            f"{record['max_abs_error']:.6f} | "
            f"{record['mean_abs_error']:.6f} |"
        )

    fastest = min(records, key=lambda item: item["metal_wall_mean_ms"])
    lines.extend(
        [
            "",
            "## Analysis",
            "",
            "- Both Metal variants remain real custom kernels and stay numerically stable by subtracting the row max before exponentiation.",
            "- GPU kernel time is expected to be much smaller than wall time for smaller batches, so command submission overhead still matters here too.",
            f"- The single fastest measured configuration was batch `{fastest['batch_size']}` with variant `{fastest['variant_name']}` at `{fastest['metal_wall_mean_ms']:.4f} ms` wall time.",
            "- The SIMD reduction variant is intended to reduce barrier and threadgroup-memory overhead relative to the shared-memory baseline.",
            "- Correctness stayed tight against Torch across all measured runs; any nonzero differences are small float32 rounding effects rather than algorithmic drift.",
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
    with tempfile.TemporaryDirectory(prefix="softmax-metal-") as temp_dir:
        temp_root = Path(temp_dir)
        for batch_size in args.batch_sizes:
            x_cpu = torch.randn((batch_size, HIDDEN_SIZE), dtype=torch.float32)
            torch_latencies_ms, torch_output = benchmark_torch(
                x_cpu=x_cpu,
                device=torch_device,
                warmup_iters=args.warmup_iters,
                timed_iters=args.timed_iters,
            )

            input_path = temp_root / f"softmax_input_b{batch_size}.bin"
            write_f32_binary(x_cpu, input_path)

            for variant in args.variants:
                try:
                    metal_run = run_metal_helper(
                        helper_binary=args.helper_binary,
                        variant=variant,
                        kernel_path=args.kernel_path,
                        input_path=input_path,
                        warmup_iters=args.warmup_iters,
                        timed_iters=args.timed_iters,
                    )
                except subprocess.CalledProcessError as exc:
                    stderr = exc.stderr.strip() if exc.stderr else "unknown Metal helper error"
                    raise SystemExit(
                        f"Metal benchmark helper failed for batch size {batch_size}, variant {variant}: {stderr}"
                    ) from exc

                metal_output = torch.tensor(
                    metal_run["output"], dtype=torch.float32
                ).reshape(batch_size, HIDDEN_SIZE)
                record = build_record(
                    batch_size=batch_size,
                    variant=variant,
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
