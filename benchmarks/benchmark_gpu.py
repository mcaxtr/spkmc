#!/usr/bin/env python3
"""
GPU vs CPU Performance Benchmark for SPKMC.

This script benchmarks GPU vs CPU performance across different:
- Graph sizes (N)
- Number of samples
- Network types

Usage:
    python benchmarks/benchmark_gpu.py [--quick] [--full] [--output results.json]

Options:
    --quick     Run minimal benchmarks (fast)
    --full      Run comprehensive benchmarks (slow, but thorough)
    --output    Save results to JSON file
    --verbose   Show detailed timing information
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    network_type: str
    N: int
    samples: int
    num_runs: int
    gpu_time_ms: float
    cpu_time_ms: float
    speedup: float
    gpu_available: bool
    notes: str = ""


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    name: str
    network_type: str
    N: int
    samples: int
    num_runs: int
    k_avg: int = 10
    exponent: float = 2.5


# Quick benchmark configurations (for CI/development)
QUICK_CONFIGS = [
    BenchmarkConfig("Small ER", "er", N=500, samples=10, num_runs=1),
    BenchmarkConfig("Medium ER", "er", N=2000, samples=20, num_runs=1),
]

# Standard benchmark configurations
STANDARD_CONFIGS = [
    BenchmarkConfig("Small ER", "er", N=1000, samples=20, num_runs=2),
    BenchmarkConfig("Medium ER", "er", N=5000, samples=50, num_runs=2),
    BenchmarkConfig("Large ER", "er", N=10000, samples=50, num_runs=2),
    BenchmarkConfig("Small CN", "cn", N=1000, samples=20, num_runs=2, exponent=2.5),
    BenchmarkConfig("Medium CN", "cn", N=5000, samples=50, num_runs=2, exponent=2.5),
]

# Full benchmark configurations (comprehensive)
FULL_CONFIGS = [
    # Erdos-Renyi at various scales
    BenchmarkConfig("ER N=1K", "er", N=1000, samples=50, num_runs=3),
    BenchmarkConfig("ER N=5K", "er", N=5000, samples=50, num_runs=3),
    BenchmarkConfig("ER N=10K", "er", N=10000, samples=100, num_runs=3),
    BenchmarkConfig("ER N=20K", "er", N=20000, samples=100, num_runs=2),
    BenchmarkConfig("ER N=50K", "er", N=50000, samples=100, num_runs=2),
    BenchmarkConfig("ER N=100K", "er", N=100000, samples=50, num_runs=1),
    # Complex Networks at various scales
    BenchmarkConfig("CN N=1K γ=2.5", "cn", N=1000, samples=50, num_runs=3, exponent=2.5),
    BenchmarkConfig("CN N=5K γ=2.5", "cn", N=5000, samples=50, num_runs=3, exponent=2.5),
    BenchmarkConfig("CN N=10K γ=2.5", "cn", N=10000, samples=100, num_runs=2, exponent=2.5),
    BenchmarkConfig("CN N=5K γ=3.0", "cn", N=5000, samples=50, num_runs=3, exponent=3.0),
    # Random Regular Networks
    BenchmarkConfig("RRN N=5K", "rrn", N=5000, samples=50, num_runs=3),
    BenchmarkConfig("RRN N=10K", "rrn", N=10000, samples=100, num_runs=2),
]


def check_gpu_available() -> Tuple[bool, str]:
    """Check if GPU is available and return status."""
    try:
        from spkmc.utils.gpu_utils import get_gpu_check_error, is_gpu_available

        available = is_gpu_available()
        error = get_gpu_check_error() or ""
        return available, error
    except ImportError:
        return False, "GPU utilities not available"


def run_benchmark(config: BenchmarkConfig, verbose: bool = False) -> BenchmarkResult:
    """Run a single benchmark with both GPU and CPU."""
    from spkmc.core.distributions import create_distribution
    from spkmc.core.simulation import SPKMC

    print(f"\n{'='*60}")
    print(f"Benchmark: {config.name}")
    print(
        f"  Network: {config.network_type.upper()}, N={config.N}, "
        f"samples={config.samples}, runs={config.num_runs}"
    )
    print(f"{'='*60}")

    # Create distribution (exponential for simplicity)
    dist = create_distribution("exponential", mu=1.0, lambda_param=0.3)
    time_steps = np.linspace(0, 30, 100)

    # Check GPU availability
    gpu_available, gpu_error = check_gpu_available()

    gpu_time_ms = float("inf")
    cpu_time_ms = float("inf")
    notes = ""

    # Warm-up run (helps with JIT compilation, memory allocation)
    print("  Warming up...")
    warmup_sim = SPKMC(dist, use_gpu=False)
    _ = warmup_sim.simulate_erdos_renyi(
        num_runs=1,
        time_steps=time_steps,
        N=100,
        samples=5,
        load_if_exists=False,
        show_progress=False,
    )

    # --- GPU Benchmark ---
    if gpu_available:
        print("  Running GPU benchmark...")
        os.environ["SPKMC_FORCE_GPU"] = "1"
        os.environ["SPKMC_BATCH_GPU"] = "1"
        os.environ.pop("SPKMC_NO_GPU", None)

        try:
            sim_gpu = SPKMC(dist, use_gpu=True)

            # Warm-up GPU
            if config.N >= 1000:
                _ = sim_gpu.simulate_erdos_renyi(
                    num_runs=1,
                    time_steps=time_steps,
                    N=500,
                    samples=5,
                    load_if_exists=False,
                    show_progress=False,
                )

            t_start = time.perf_counter()

            if config.network_type == "er":
                _ = sim_gpu.simulate_erdos_renyi(
                    num_runs=config.num_runs,
                    time_steps=time_steps,
                    N=config.N,
                    k_avg=config.k_avg,
                    samples=config.samples,
                    load_if_exists=False,
                    show_progress=False,
                )
            elif config.network_type == "cn":
                _ = sim_gpu.simulate_complex_network(
                    num_runs=config.num_runs,
                    exponent=config.exponent,
                    time_steps=time_steps,
                    N=config.N,
                    k_avg=config.k_avg,
                    samples=config.samples,
                    load_if_exists=False,
                    show_progress=False,
                )
            elif config.network_type == "rrn":
                _ = sim_gpu.simulate_random_regular_network(
                    num_runs=config.num_runs,
                    time_steps=time_steps,
                    N=config.N,
                    k_avg=config.k_avg,
                    samples=config.samples,
                    load_if_exists=False,
                    show_progress=False,
                )

            t_end = time.perf_counter()
            gpu_time_ms = (t_end - t_start) * 1000
            print(f"  GPU: {gpu_time_ms:.1f}ms")

        except Exception as e:
            notes = f"GPU error: {type(e).__name__}: {e}"
            print(f"  GPU: FAILED - {notes}")
            gpu_time_ms = float("inf")

        finally:
            os.environ.pop("SPKMC_FORCE_GPU", None)
            os.environ.pop("SPKMC_BATCH_GPU", None)
    else:
        notes = f"GPU not available: {gpu_error}"
        print(f"  GPU: SKIPPED - {notes}")

    # --- CPU Benchmark ---
    print("  Running CPU benchmark...")
    os.environ["SPKMC_NO_GPU"] = "1"

    try:
        sim_cpu = SPKMC(dist, use_gpu=False)

        t_start = time.perf_counter()

        if config.network_type == "er":
            _ = sim_cpu.simulate_erdos_renyi(
                num_runs=config.num_runs,
                time_steps=time_steps,
                N=config.N,
                k_avg=config.k_avg,
                samples=config.samples,
                load_if_exists=False,
                show_progress=False,
            )
        elif config.network_type == "cn":
            _ = sim_cpu.simulate_complex_network(
                num_runs=config.num_runs,
                exponent=config.exponent,
                time_steps=time_steps,
                N=config.N,
                k_avg=config.k_avg,
                samples=config.samples,
                load_if_exists=False,
                show_progress=False,
            )
        elif config.network_type == "rrn":
            _ = sim_cpu.simulate_random_regular_network(
                num_runs=config.num_runs,
                time_steps=time_steps,
                N=config.N,
                k_avg=config.k_avg,
                samples=config.samples,
                load_if_exists=False,
                show_progress=False,
            )

        t_end = time.perf_counter()
        cpu_time_ms = (t_end - t_start) * 1000
        print(f"  CPU: {cpu_time_ms:.1f}ms")

    except Exception as e:
        notes += f" CPU error: {type(e).__name__}: {e}"
        print(f"  CPU: FAILED - {e}")
        cpu_time_ms = float("inf")

    finally:
        os.environ.pop("SPKMC_NO_GPU", None)

    # Calculate speedup
    if gpu_time_ms < float("inf") and cpu_time_ms < float("inf") and gpu_time_ms > 0:
        speedup = cpu_time_ms / gpu_time_ms
    else:
        speedup = 0.0

    print(f"  Speedup: {speedup:.2f}x")

    return BenchmarkResult(
        name=config.name,
        network_type=config.network_type,
        N=config.N,
        samples=config.samples,
        num_runs=config.num_runs,
        gpu_time_ms=gpu_time_ms if gpu_time_ms < float("inf") else -1,
        cpu_time_ms=cpu_time_ms if cpu_time_ms < float("inf") else -1,
        speedup=speedup,
        gpu_available=gpu_available,
        notes=notes,
    )


def print_summary(results: List[BenchmarkResult]):
    """Print a summary table of benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    # Header
    print(f"{'Name':<25} {'N':>8} {'Samples':>8} {'GPU (ms)':>10} {'CPU (ms)':>10} {'Speedup':>10}")
    print("-" * 80)

    for r in results:
        gpu_str = f"{r.gpu_time_ms:.1f}" if r.gpu_time_ms > 0 else "N/A"
        cpu_str = f"{r.cpu_time_ms:.1f}" if r.cpu_time_ms > 0 else "N/A"
        speedup_str = f"{r.speedup:.2f}x" if r.speedup > 0 else "N/A"
        print(f"{r.name:<25} {r.N:>8} {r.samples:>8} {gpu_str:>10} {cpu_str:>10} {speedup_str:>10}")

    print("-" * 80)

    # Summary statistics
    valid_results = [r for r in results if r.speedup > 0]
    if valid_results:
        avg_speedup = sum(r.speedup for r in valid_results) / len(valid_results)
        max_speedup = max(r.speedup for r in valid_results)
        min_speedup = min(r.speedup for r in valid_results)
        print(f"\nSpeedup Statistics:")
        print(f"  Average: {avg_speedup:.2f}x")
        print(f"  Best:    {max_speedup:.2f}x")
        print(f"  Worst:   {min_speedup:.2f}x")

    # GPU threshold analysis
    print(f"\nGPU Min Nodes Threshold: {SPKMC.GPU_MIN_NODES}")
    results_above_threshold = [r for r in results if r.N >= SPKMC.GPU_MIN_NODES]
    if results_above_threshold:
        above_avg = sum(r.speedup for r in results_above_threshold if r.speedup > 0) / max(
            1, len([r for r in results_above_threshold if r.speedup > 0])
        )
        print(f"  Avg speedup for N >= {SPKMC.GPU_MIN_NODES}: {above_avg:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="GPU vs CPU Performance Benchmark for SPKMC")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmarks only")
    parser.add_argument("--full", action="store_true", help="Run full comprehensive benchmarks")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--verbose", action="store_true", help="Show detailed timing")
    parser.add_argument("--debug", action="store_true", help="Enable SPKMC debug timing")
    args = parser.parse_args()

    # Select configs
    if args.quick:
        configs = QUICK_CONFIGS
        print("Running QUICK benchmarks...")
    elif args.full:
        configs = FULL_CONFIGS
        print("Running FULL benchmarks...")
    else:
        configs = STANDARD_CONFIGS
        print("Running STANDARD benchmarks...")

    # Enable debug timing if requested
    if args.debug:
        os.environ["SPKMC_DEBUG"] = "1"

    # Import SPKMC after environment setup
    from spkmc.core.simulation import SPKMC

    # Check GPU availability
    gpu_available, gpu_error = check_gpu_available()
    print(f"\nGPU Available: {gpu_available}")
    if not gpu_available:
        print(f"  Reason: {gpu_error}")
    print(f"GPU Min Nodes: {SPKMC.GPU_MIN_NODES}")

    # Run benchmarks
    results = []
    for config in configs:
        try:
            result = run_benchmark(config, verbose=args.verbose)
            results.append(result)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            results.append(
                BenchmarkResult(
                    name=config.name,
                    network_type=config.network_type,
                    N=config.N,
                    samples=config.samples,
                    num_runs=config.num_runs,
                    gpu_time_ms=-1,
                    cpu_time_ms=-1,
                    speedup=0,
                    gpu_available=gpu_available,
                    notes=f"Error: {e}",
                )
            )

    # Print summary
    print_summary(results)

    # Save results if requested
    if args.output:
        output_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu_available": gpu_available,
            "gpu_min_nodes": SPKMC.GPU_MIN_NODES,
            "results": [asdict(r) for r in results],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
