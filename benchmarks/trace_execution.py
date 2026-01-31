#!/usr/bin/env python3
"""
Trace actual execution path and timing for SPKMC simulation.

This script instruments the real code to show exactly what path is taken
and where time is spent.

Usage:
    python benchmarks/trace_execution.py [--N 50000] [--samples 20]
"""

import argparse
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def trace_simulation(N: int, samples: int, k_avg: int = 10, force_gpu: bool = False):
    """Run simulation with detailed tracing."""

    print("=" * 70)
    print("SPKMC EXECUTION TRACE")
    print("=" * 70)
    print(f"Parameters: N={N}, samples={samples}, k_avg={k_avg}")
    print()

    # Set environment
    if force_gpu:
        os.environ["SPKMC_FORCE_GPU"] = "1"
        print("Environment: SPKMC_FORCE_GPU=1")
    else:
        os.environ.pop("SPKMC_FORCE_GPU", None)
        print("Environment: SPKMC_FORCE_GPU not set")

    os.environ["SPKMC_BATCH_GPU"] = "1"
    print("Environment: SPKMC_BATCH_GPU=1")
    print()

    # Import after environment setup
    from spkmc.core.distributions import create_distribution
    from spkmc.core.simulation import SPKMC
    from spkmc.utils.hardware import get_hardware_info

    # Hardware detection
    print("--- Hardware Detection ---")
    hw = get_hardware_info()
    print(f"GPU available: {hw.gpu_available}")
    print(f"GPU name: {hw.gpu_name}")
    print()

    # Create distribution
    dist = create_distribution("exponential", mu=1.0, lambda_param=0.3)
    time_steps = np.linspace(0, 30, 100).astype(np.float32)

    # Create SPKMC with GPU
    print("--- SPKMC Initialization ---")
    sim = SPKMC(dist, use_gpu=True)
    print(f"sim.use_gpu = {sim.use_gpu}")
    print(f"sim._gpu_available = {sim._gpu_available}")
    print(f"sim._use_batched_gpu = {sim._use_batched_gpu}")
    print(f"sim._force_gpu = {sim._force_gpu}")
    print(f"SPKMC.GPU_MIN_NODES = {SPKMC.GPU_MIN_NODES}")
    print()

    # Check which path will be taken
    print("--- Path Decision ---")
    will_use_batched = sim._should_use_batched_gpu(N)
    print(f"_should_use_batched_gpu({N}) = {will_use_batched}")

    if will_use_batched:
        print("→ Will use: BatchedGPUSimulator (optimized GPU path)")
    else:
        # Check if non-batched GPU will be used
        use_gpu_for_graph = (
            sim.use_gpu and sim._gpu_available and (N >= SPKMC.GPU_MIN_NODES or sim._force_gpu)
        )
        if use_gpu_for_graph:
            print("→ Will use: Non-batched GPU (per-sample get_dist_gpu)")
        else:
            print("→ Will use: CPU (SciPy Dijkstra)")
    print()

    # Run simulation with timing
    print("--- Running Simulation ---")

    # Timing breakdown
    timings = {}

    # Network generation
    t_start = time.perf_counter()
    from spkmc.core.networks import NetworkFactory

    use_fast_edges = sim._should_use_batched_gpu(N)
    print(f"Using fast edge generators: {use_fast_edges}")

    if use_fast_edges:
        _, edges = NetworkFactory.create_erdos_renyi_edges(N, k_avg)
    else:
        G = NetworkFactory.create_erdos_renyi(N, k_avg)
        edges = np.array(G.edges())

    t_network = time.perf_counter()
    timings["network_generation"] = (t_network - t_start) * 1000
    print(f"Network generated: {len(edges)} edges in {timings['network_generation']:.1f}ms")

    # Sources
    init_infect = max(1, int(N * 0.01))
    sources = np.random.randint(0, N, init_infect)
    print(f"Initial infected: {init_infect} nodes")

    # Run simulation
    t_sim_start = time.perf_counter()
    S, I, R = sim.run_multiple_simulations_from_edges(
        N, edges, sources, time_steps, samples, show_progress=False
    )
    t_sim_end = time.perf_counter()
    timings["simulation"] = (t_sim_end - t_sim_start) * 1000

    print(f"Simulation completed in {timings['simulation']:.1f}ms")
    print(f"  → Per sample: {timings['simulation']/samples:.1f}ms")
    print()

    # Summary
    print("--- Summary ---")
    total_time = timings["network_generation"] + timings["simulation"]
    print(f"Total time: {total_time:.1f}ms")
    print(
        f"  Network generation: {timings['network_generation']:.1f}ms ({timings['network_generation']/total_time*100:.1f}%)"
    )
    print(
        f"  Simulation: {timings['simulation']:.1f}ms ({timings['simulation']/total_time*100:.1f}%)"
    )
    print()

    # Results sanity check
    print("--- Results Check ---")
    print(f"S[0] = {S[0]:.4f} (should be ~0.99)")
    print(f"R[-1] = {R[-1]:.4f} (should be high if epidemic spread)")
    print(f"S+I+R at t=0: {S[0]+I[0]+R[0]:.6f} (should be 1.0)")

    # Cleanup
    os.environ.pop("SPKMC_FORCE_GPU", None)
    os.environ.pop("SPKMC_BATCH_GPU", None)


def main():
    parser = argparse.ArgumentParser(description="Trace SPKMC execution path")
    parser.add_argument("--N", type=int, default=50000, help="Number of nodes")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples")
    parser.add_argument("--k-avg", type=int, default=10, help="Average degree")
    parser.add_argument("--force-gpu", action="store_true", help="Force GPU mode")
    args = parser.parse_args()

    trace_simulation(args.N, args.samples, args.k_avg, args.force_gpu)


if __name__ == "__main__":
    main()
