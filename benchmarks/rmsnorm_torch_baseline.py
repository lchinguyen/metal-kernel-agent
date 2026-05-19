import argparse
import json
import statistics
import time
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is required to run this benchmark. Install torch in the active "
        "Python environment, then rerun benchmarks/rmsnorm_torch_baseline.py."
    ) from exc


DEFAULT_SHAPE = (1, 4096)
DEFAULT_DTYPE = "float32"
DEFAULT_EPS = 1e-6
DEFAULT_WARMUP_ITERS = 20
DEFAULT_TIMED_ITERS = 100
DEFAULT_RESULTS_PATH = Path("results/results.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the Torch RMSNorm baseline for shape [1, 4096]."
    )
    parser.add_argument("--warmup-iters", type=int, default=DEFAULT_WARMUP_ITERS)
    parser.add_argument("--timed-iters", type=int, default=DEFAULT_TIMED_ITERS)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Execution device for the Torch baseline.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Path to the JSONL results log.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested device 'mps' is not available.")
        return torch.device("mps")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device 'cuda' is not available.")
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # Keep accumulation in fp32 so the baseline matches the intended numerical path.
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    return x * inv_rms.to(x.dtype) * weight


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


def build_result_record(
    latencies_ms: list[float],
    output: torch.Tensor,
    device: torch.device,
    eps: float,
    warmup_iters: int,
    timed_iters: int,
) -> dict:
    ordered = sorted(latencies_ms)
    return {
        "benchmark": "rmsnorm_torch_baseline",
        "implementation": "torch",
        "shape": list(DEFAULT_SHAPE),
        "dtype": DEFAULT_DTYPE,
        "device": str(device),
        "eps": eps,
        "warmup_iterations": warmup_iters,
        "timed_iterations": timed_iters,
        "latency_ms_mean": statistics.mean(latencies_ms),
        "latency_ms_median": statistics.median(latencies_ms),
        "latency_ms_p50": percentile(ordered, 0.50),
        "latency_ms_p95": percentile(ordered, 0.95),
        "max_abs_output_value": output.abs().max().item(),
    }


def append_jsonl(record: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def main() -> None:
    args = parse_args()
    if args.warmup_iters < 0:
        raise SystemExit("--warmup-iters must be >= 0.")
    if args.timed_iters <= 0:
        raise SystemExit("--timed-iters must be > 0.")

    device = resolve_device(args.device)

    torch.manual_seed(args.seed)
    x = torch.randn(DEFAULT_SHAPE, dtype=torch.float32, device=device)
    weight = torch.randn(DEFAULT_SHAPE[-1], dtype=torch.float32, device=device)

    with torch.no_grad():
        for _ in range(args.warmup_iters):
            _ = rmsnorm_torch(x, weight, args.eps)
        synchronize(device)

        latencies_ms = []
        output = None
        for _ in range(args.timed_iters):
            synchronize(device)
            start_ns = time.perf_counter_ns()
            output = rmsnorm_torch(x, weight, args.eps)
            synchronize(device)
            end_ns = time.perf_counter_ns()
            latencies_ms.append((end_ns - start_ns) / 1_000_000.0)

    if output is None:
        raise RuntimeError("Benchmark did not produce an output tensor.")

    record = build_result_record(
        latencies_ms=latencies_ms,
        output=output,
        device=device,
        eps=args.eps,
        warmup_iters=args.warmup_iters,
        timed_iters=args.timed_iters,
    )
    append_jsonl(record, args.results_path)

    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
