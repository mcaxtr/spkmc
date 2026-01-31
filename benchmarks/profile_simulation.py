#!/usr/bin/env python3
"""
Detailed profiling of SPKMC simulation pipeline.

Identifies bottlenecks by timing each stage of the simulation.
Outputs a clean summary table without verbose logs.

Usage:
    python benchmarks/profile_simulation.py [--N 50000] [--samples 20]
"""

import argparse
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class TimingResult:
    """Result of a single timing measurement."""

    name: str
    duration_ms: float
    details: str = ""


@dataclass
class ProfileResults:
    """Collection of profiling results."""

    timings: List[TimingResult] = field(default_factory=list)

    def add(self, name: str, duration_ms: float, details: str = ""):
        self.timings.append(TimingResult(name, duration_ms, details))

    def total_ms(self) -> float:
        return sum(t.duration_ms for t in self.timings)

    def print_summary(self):
        total = self.total_ms()
        print("\n" + "=" * 70)
        print("SIMULATION PROFILING SUMMARY")
        print("=" * 70)
        print(f"{'Stage':<35} {'Time (ms)':>12} {'%':>8}")
        print("-" * 70)

        for t in self.timings:
            pct = (t.duration_ms / total * 100) if total > 0 else 0
            bar = "█" * int(pct / 2.5)  # Max 40 chars at 100%
            details = f" ({t.details})" if t.details else ""
            print(f"{t.name:<35} {t.duration_ms:>12.1f} {pct:>7.1f}% {bar}")

        print("-" * 70)
        print(f"{'TOTAL':<35} {total:>12.1f}")
        print("=" * 70)


@contextmanager
def timed_section(results: ProfileResults, name: str, details: str = ""):
    """Context manager for timing a code section."""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    results.add(name, (end - start) * 1000, details)


def profile_gpu_simulation(N: int, samples: int, k_avg: int = 10):
    """Profile GPU simulation pipeline in detail."""
    results = ProfileResults()

    print(f"\nProfiling GPU simulation: N={N}, samples={samples}, k_avg={k_avg}")

    # Import GPU utilities
    try:
        import cudf
        import cugraph
        import cupy as cp

        from spkmc.utils.gpu_utils import BatchedGPUSimulator, configure_gpu_memory_pool

        gpu_available = True
    except ImportError as e:
        print(f"GPU not available: {e}")
        return

    # Force GPU mode
    os.environ["SPKMC_FORCE_GPU"] = "1"
    os.environ["SPKMC_BATCH_GPU"] = "1"

    # =========================================================================
    # Stage 1: Network Generation
    # =========================================================================
    from spkmc.core.networks import NetworkFactory

    with timed_section(results, "1. Network Generation (fast)", f"N={N}"):
        _, edges = NetworkFactory.create_erdos_renyi_edges(N, k_avg)

    print(f"   Generated {len(edges)} edges")

    # =========================================================================
    # Stage 2: GPU Memory Pool Configuration
    # =========================================================================
    with timed_section(results, "2. GPU Memory Pool Config"):
        configure_gpu_memory_pool()

    # =========================================================================
    # Stage 3: Edge Transfer to GPU
    # =========================================================================
    with timed_section(results, "3. Edge Transfer to GPU", f"{len(edges)} edges"):
        edges_gpu = cp.asarray(edges, dtype=cp.int32)
        cp.cuda.Stream.null.synchronize()  # Ensure transfer complete

    # =========================================================================
    # Stage 4: Time Steps Transfer
    # =========================================================================
    time_steps = np.linspace(0, 30, 100).astype(np.float32)
    with timed_section(results, "4. Time Steps Transfer", f"{len(time_steps)} steps"):
        time_steps_gpu = cp.asarray(time_steps, dtype=cp.float32)
        cp.cuda.Stream.null.synchronize()

    # =========================================================================
    # Stage 5: Batched Random Number Generation
    # =========================================================================
    params = {"distribution": "exponential", "mu": 1.0, "lambda_val": 0.3}

    with timed_section(results, "5. Batched RNG - Recovery", f"{samples}x{N}"):
        recovery_all = cp.random.exponential(
            1.0 / params["mu"], size=(samples, N), dtype=cp.float32
        )
        cp.cuda.Stream.null.synchronize()

    with timed_section(results, "6. Batched RNG - Edges", f"{samples}x{len(edges)}"):
        edge_times_all = cp.random.exponential(
            1.0 / params["lambda_val"], size=(samples, len(edges)), dtype=cp.float32
        )
        cp.cuda.Stream.null.synchronize()

    # =========================================================================
    # Stage 6: Pre-allocate Graph Structure (optimization)
    # =========================================================================
    sources = np.array([0])  # Single source
    sources_gpu = cp.asarray(sources, dtype=cp.int32)
    super_node = N
    num_sources = len(sources_gpu)
    total_edges = len(edges) + num_sources

    with timed_section(results, "7. Pre-allocate Graph Structure"):
        # Pre-allocate src/dst arrays (fixed structure)
        graph_src = cp.empty(total_edges, dtype=cp.int32)
        graph_src[: len(edges)] = edges_gpu[:, 0]
        graph_src[len(edges) :] = super_node

        graph_dst = cp.empty(total_edges, dtype=cp.int32)
        graph_dst[: len(edges)] = edges_gpu[:, 1]
        graph_dst[len(edges) :] = sources_gpu

        # Pre-allocate weights (updated each sample)
        graph_weights = cp.empty(total_edges, dtype=cp.float32)
        graph_weights[len(edges) :] = 0.0  # Super-node weights always 0

        # Pre-create DataFrame (reused across samples)
        graph_df = cudf.DataFrame({"src": graph_src, "dst": graph_dst, "weight": graph_weights})
        cp.cuda.Stream.null.synchronize()

    # =========================================================================
    # Stage 7: SSSP Loop (per-sample)
    # =========================================================================
    sssp_times = []
    graph_build_times = []
    distances_all = []

    with timed_section(results, "8. SSSP Loop", f"{samples} samples"):
        for s in range(samples):
            # Build graph timing (reuses DataFrame, only updates weight column)
            t1 = time.perf_counter()

            u = edges_gpu[:, 0]
            recovery = recovery_all[s]
            edge_times = edge_times_all[s]

            # Update weights in-place in pre-allocated array
            graph_weights[: len(edges)] = cp.where(
                edge_times >= recovery[u], cp.float32(np.inf), edge_times
            )

            # Update weight column in pre-existing DataFrame
            graph_df["weight"] = graph_weights

            G = cugraph.Graph(directed=True)
            G.from_cudf_edgelist(
                graph_df, source="src", destination="dst", edge_attr="weight", renumber=False
            )

            t2 = time.perf_counter()
            graph_build_times.append((t2 - t1) * 1000)

            # SSSP timing
            t3 = time.perf_counter()
            result = cugraph.sssp(G, source=super_node)
            cp.cuda.Stream.null.synchronize()
            t4 = time.perf_counter()
            sssp_times.append((t4 - t3) * 1000)

            # Extract distances
            vertices = result["vertex"].to_numpy()
            dist_values = result["distance"].to_numpy()
            distances = np.full(N, np.inf, dtype=np.float32)
            valid_mask = vertices < N
            distances[vertices[valid_mask]] = dist_values[valid_mask]
            distances_all.append(distances)

    # Add breakdown of SSSP loop
    avg_graph_build = np.mean(graph_build_times)
    avg_sssp = np.mean(sssp_times)
    results.add("   8a. Graph Build (avg/sample)", avg_graph_build, "cuDF+cuGraph")
    results.add("   8b. SSSP Algorithm (avg/sample)", avg_sssp, "Dijkstra")

    # =========================================================================
    # Stage 8: Batched SIR Calculation
    # =========================================================================
    with timed_section(results, "9. Batched SIR Calculation", f"{samples}x{len(time_steps)}"):
        dist_gpu = cp.stack([cp.asarray(d, dtype=cp.float32) for d in distances_all])

        d = dist_gpu[:, cp.newaxis, :]
        r = recovery_all[:, cp.newaxis, :]
        t = time_steps_gpu[cp.newaxis, :, cp.newaxis]

        S_mask = d > t
        I_mask = (d <= t) & (d + r > t)

        S = cp.sum(S_mask, axis=2, dtype=cp.float32) / N
        I = cp.sum(I_mask, axis=2, dtype=cp.float32) / N
        R = 1.0 - S - I

        S_result = S.get()
        I_result = I.get()
        R_result = R.get()
        cp.cuda.Stream.null.synchronize()

    # =========================================================================
    # Stage 9: Mean Calculation
    # =========================================================================
    with timed_section(results, "10. Mean Calculation"):
        S_mean = np.mean(S_result, axis=0)
        I_mean = np.mean(I_result, axis=0)
        R_mean = np.mean(R_result, axis=0)

    # Print summary
    results.print_summary()

    # Print recommendations
    print("\nANALYSIS:")
    total = results.total_ms()
    sssp_loop_time = next(t.duration_ms for t in results.timings if t.name == "8. SSSP Loop")
    sssp_pct = sssp_loop_time / total * 100

    if sssp_pct > 50:
        print(f"  → BOTTLENECK: SSSP loop takes {sssp_pct:.1f}% of total time")
        print(f"  → Average per sample: {sssp_loop_time/samples:.1f}ms")
        print(
            f"     - Graph build: {avg_graph_build:.1f}ms ({avg_graph_build/sssp_loop_time*samples*100:.1f}%)"
        )
        print(
            f"     - SSSP algorithm: {avg_sssp:.1f}ms ({avg_sssp/sssp_loop_time*samples*100:.1f}%)"
        )
        print("")
        print("  POSSIBLE OPTIMIZATIONS:")
        print("    1. Reduce number of samples (current: {})".format(samples))
        print("    2. Graph structure is pre-allocated - weights updated in-place")
        print("    3. cuDF DataFrame still created each sample (cuGraph limitation)")
        print("    4. Consider multi-stream SSSP (limited by GPU memory)")
    else:
        print(f"  → No single bottleneck (SSSP is {sssp_pct:.1f}%)")

    # Cleanup
    os.environ.pop("SPKMC_FORCE_GPU", None)
    os.environ.pop("SPKMC_BATCH_GPU", None)


def profile_cpu_simulation(N: int, samples: int, k_avg: int = 10):
    """Profile CPU simulation pipeline."""
    results = ProfileResults()

    print(f"\nProfiling CPU simulation: N={N}, samples={samples}, k_avg={k_avg}")

    os.environ["SPKMC_NO_GPU"] = "1"

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    from spkmc.core.distributions import create_distribution
    from spkmc.core.networks import NetworkFactory

    dist = create_distribution("exponential", mu=1.0, lambda_param=0.3)
    time_steps = np.linspace(0, 30, 100)

    # Network Generation
    with timed_section(results, "1. Network Generation (NetworkX)", f"N={N}"):
        G = NetworkFactory.create_erdos_renyi(N, k_avg)
        edges = np.array(G.edges())

    print(f"   Generated {len(edges)} edges")

    dijkstra_times = []

    with timed_section(results, "2. Simulation Loop", f"{samples} samples"):
        for s in range(samples):
            # Recovery weights
            recovery_weights = dist.get_recovery_weights(N)

            # Infection times
            infection_times = dist.get_infection_times(recovery_weights, edges)

            # Sparse matrix
            row_indices = edges[:, 0]
            col_indices = edges[:, 1]
            graph_matrix = csr_matrix((infection_times, (row_indices, col_indices)), shape=(N, N))

            # Dijkstra
            t1 = time.perf_counter()
            sources = np.array([0])
            dist_matrix = dijkstra(
                csgraph=graph_matrix, directed=True, indices=sources, return_predecessors=False
            )
            t2 = time.perf_counter()
            dijkstra_times.append((t2 - t1) * 1000)

    avg_dijkstra = np.mean(dijkstra_times)
    results.add("   2a. Dijkstra (avg/sample)", avg_dijkstra, "scipy.sparse")

    results.print_summary()

    os.environ.pop("SPKMC_NO_GPU", None)


def main():
    parser = argparse.ArgumentParser(description="Profile SPKMC simulation pipeline")
    parser.add_argument("--N", type=int, default=50000, help="Number of nodes")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples")
    parser.add_argument("--k-avg", type=int, default=10, help="Average degree")
    parser.add_argument("--cpu-only", action="store_true", help="Only profile CPU")
    parser.add_argument("--gpu-only", action="store_true", help="Only profile GPU")
    args = parser.parse_args()

    if not args.cpu_only:
        profile_gpu_simulation(args.N, args.samples, args.k_avg)

    if not args.gpu_only:
        profile_cpu_simulation(args.N, args.samples, args.k_avg)


if __name__ == "__main__":
    main()
