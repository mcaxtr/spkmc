"""
GPU utilities for SPKMC acceleration.

Provides optional GPU acceleration using CuPy, cuDF, and cuGraph.
All functions gracefully fall back to raising ImportError if dependencies are missing.

To install GPU dependencies:
    pip install spkmc[gpu]

Optimizations implemented:
- Batched random number generation across all samples
- Single edge transfer per simulation run (reused across samples)
- Float32 precision throughout (reduced memory bandwidth)
- Skip cuGraph renumbering (vertices already in [0, N) range)
- Batched SIR calculation across all samples
"""

import os
import sys
import time as time_module
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Global flag for GPU availability (cached after first check)
_GPU_AVAILABLE: Optional[bool] = None
_GPU_CHECK_ERROR: Optional[str] = None
_MEMORY_POOL_CONFIGURED: bool = False


def is_gpu_available() -> bool:
    """
    Check if GPU acceleration is available.

    Caches result after first check for performance.

    Returns:
        True if all GPU dependencies are available and GPU is accessible
    """
    global _GPU_AVAILABLE, _GPU_CHECK_ERROR

    if _GPU_AVAILABLE is not None:
        return _GPU_AVAILABLE

    try:
        import cudf
        import cugraph
        import cupy as cp

        # Verify all GPU libraries are importable
        _ = (cudf, cugraph)  # Mark as intentionally checked
        # Verify GPU is accessible with a simple operation
        _ = cp.array([1, 2, 3])
        _GPU_AVAILABLE = True

    except ImportError as e:
        _GPU_AVAILABLE = False
        _GPU_CHECK_ERROR = f"Missing dependencies: {e}"
    except Exception as e:
        _GPU_AVAILABLE = False
        _GPU_CHECK_ERROR = f"GPU access error: {e}"

    return _GPU_AVAILABLE


def get_gpu_check_error() -> Optional[str]:
    """Get the error message from the last GPU availability check."""
    return _GPU_CHECK_ERROR


def reset_gpu_cache() -> None:
    """Reset the GPU availability cache (useful for testing)."""
    global _GPU_AVAILABLE, _GPU_CHECK_ERROR, _MEMORY_POOL_CONFIGURED
    _GPU_AVAILABLE = None
    _GPU_CHECK_ERROR = None
    _MEMORY_POOL_CONFIGURED = False


def configure_gpu_memory_pool(fraction: float = 0.8) -> bool:
    """
    Configure CuPy memory pool for better memory reuse.

    Memory pool caches allocations to avoid repeated cudaMalloc calls,
    which significantly reduces overhead in repeated operations.

    Args:
        fraction: Fraction of GPU memory to allow (0.0-1.0)

    Returns:
        True if configuration succeeded, False otherwise
    """
    global _MEMORY_POOL_CONFIGURED

    if _MEMORY_POOL_CONFIGURED:
        return True

    try:
        import cupy as cp

        mempool = cp.get_default_memory_pool()
        mempool.set_limit(fraction=fraction)
        _MEMORY_POOL_CONFIGURED = True
        return True
    except Exception:
        return False


# Conditional imports and GPU implementations
try:
    import cudf
    import cugraph
    import cupy as cp

    def get_dist_gpu(
        N: int, edges: np.ndarray, sources: np.ndarray, params: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate shortest path distances using GPU acceleration.

        Uses cuGraph SSSP with a super-node technique for multiple sources.

        Args:
            N: Number of nodes
            edges: Edge array of shape (E, 2)
            sources: Array of source node indices
            params: Distribution parameters dict with keys:
                    'distribution': 'gamma' or 'exponential'
                    'shape', 'scale': for gamma
                    'mu', 'lambda_val': for exponential

        Returns:
            Tuple of (infection_times, recovery_times) as numpy arrays
        """
        import os
        import sys
        import time as time_module

        debug = os.environ.get("SPKMC_DEBUG") == "1"
        t_start = time_module.perf_counter()

        # Transfer edges to GPU
        edges_gpu = cp.asarray(edges, dtype=cp.int32)
        sources_gpu = cp.asarray(sources, dtype=cp.int32)

        # Sample recovery and infection times on GPU
        distribution = params.get("distribution", "exponential").lower()
        lmbd = params.get("lambda_val", 1.0)

        if distribution == "gamma":
            shape = params.get("shape", 2.0)
            scale = params.get("scale", 1.0)
            recovery_times = cp.random.gamma(shape, scale, size=N)
            # Infection times use exponential even for gamma recovery (matches CPU impl)
            edge_times = cp.random.exponential(1.0 / lmbd, size=edges_gpu.shape[0])
        else:
            mu = params.get("mu", 1.0)
            recovery_times = cp.random.exponential(1.0 / mu, size=N)
            edge_times = cp.random.exponential(1.0 / lmbd, size=edges_gpu.shape[0])

        # Compute infection times (inf if >= recovery time)
        u = edges_gpu[:, 0]
        infection_weights = cp.where(edge_times >= recovery_times[u], cp.inf, edge_times)

        # Create super-node for multi-source SSSP
        super_node = N
        super_edges_src = cp.full(len(sources_gpu), super_node, dtype=cp.int32)
        super_edges_dst = sources_gpu
        super_weights = cp.zeros(len(sources_gpu), dtype=cp.float32)

        # Concatenate edges
        all_src = cp.concatenate([edges_gpu[:, 0], super_edges_src])
        all_dst = cp.concatenate([edges_gpu[:, 1], super_edges_dst])
        all_weights = cp.concatenate([infection_weights.astype(cp.float32), super_weights])

        # Build cuGraph graph
        df = cudf.DataFrame({"src": all_src, "dst": all_dst, "weight": all_weights})
        G = cugraph.Graph(directed=True)
        G.from_cudf_edgelist(df, source="src", destination="dst", edge_attr="weight")

        # Run SSSP from super-node
        result = cugraph.sssp(G, source=super_node)

        # Extract distances properly by vertex ID
        # cuGraph SSSP returns DataFrame with 'vertex' and 'distance' columns
        # We need to map vertex IDs to their distances, excluding super-node
        vertices = result["vertex"].to_numpy()
        dist_values = result["distance"].to_numpy()

        # Initialize distances to infinity (unreachable nodes)
        distances = np.full(N, np.inf, dtype=np.float32)

        # Filter out super-node and use vectorized assignment
        valid_mask = vertices < N
        valid_vertices = vertices[valid_mask]
        valid_distances = dist_values[valid_mask]
        distances[valid_vertices] = valid_distances

        t_end = time_module.perf_counter()
        if debug:
            print(
                f"[GPU TIMING] get_dist_gpu: {(t_end - t_start)*1000:.1f}ms "
                f"(N={N}, edges={len(edges)}, vertices_in_result={len(vertices)})",
                file=sys.stderr,
            )

        return distances, recovery_times.get()

    def calculate_gpu(
        N: int, time_to_infect: np.ndarray, recovery_times: np.ndarray, time_steps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate SIR proportions over time using GPU.

        Args:
            N: Number of nodes
            time_to_infect: Infection time for each node
            recovery_times: Recovery time for each node
            time_steps: Array of time points

        Returns:
            Tuple of (S, I, R) arrays with proportions at each time step
        """
        import os
        import sys
        import time as time_module

        debug = os.environ.get("SPKMC_DEBUG") == "1"
        t_start = time_module.perf_counter()

        # Validate input shapes
        if len(time_to_infect) != N:
            raise ValueError(
                f"time_to_infect shape mismatch: expected {N}, got {len(time_to_infect)}"
            )
        if len(recovery_times) != N:
            raise ValueError(
                f"recovery_times shape mismatch: expected {N}, got {len(recovery_times)}"
            )

        if debug:
            print(
                f"[GPU DEBUG] calculate_gpu: N={N}, time_to_infect={time_to_infect.shape}, "
                f"recovery_times={recovery_times.shape}, steps={len(time_steps)}",
                file=sys.stderr,
            )

        time_to_infect_gpu = cp.asarray(time_to_infect)
        recovery_times_gpu = cp.asarray(recovery_times)
        time_steps_gpu = cp.asarray(time_steps)

        # Vectorized computation: broadcast time_steps against node arrays
        # time_to_infect_gpu: (N,), time_steps_gpu: (steps,)
        # Result: (steps, N) boolean arrays
        time_to_infect_2d = time_to_infect_gpu[cp.newaxis, :]  # (1, N)
        recovery_2d = recovery_times_gpu[cp.newaxis, :]  # (1, N)
        time_steps_2d = time_steps_gpu[:, cp.newaxis]  # (steps, 1)

        # Compute states for all time steps at once
        s_mask = time_to_infect_2d > time_steps_2d  # (steps, N)
        i_mask = (~s_mask) & (time_to_infect_2d + recovery_2d > time_steps_2d)  # (steps, N)
        r_mask = (~s_mask) & (~i_mask)  # (steps, N)

        # Sum across nodes and normalize
        s_time = cp.sum(s_mask, axis=1) / N
        i_time = cp.sum(i_mask, axis=1) / N
        r_time = cp.sum(r_mask, axis=1) / N

        result = s_time.get(), i_time.get(), r_time.get()
        t_end = time_module.perf_counter()
        if debug:
            print(
                f"[GPU TIMING] calculate_gpu: {(t_end - t_start)*1000:.1f}ms "
                f"(N={N}, steps={len(time_steps)})",
                file=sys.stderr,
            )
        return result

    class BatchedGPUSimulator:
        """
        GPU simulator that batches operations across samples for improved performance.

        Optimizations:
        - Transfers edges to GPU once, reuses for all samples
        - Generates all random numbers in single batched calls
        - Computes SIR for all samples in one vectorized operation
        - Uses float32 throughout to reduce memory bandwidth
        - Skips cuGraph renumbering (vertices already in [0, N) range)

        Usage:
            simulator = BatchedGPUSimulator(N, edges, time_steps)
            S_mean, I_mean, R_mean = simulator.run_samples(samples, sources, params)
        """

        def __init__(
            self,
            N: int,
            edges: np.ndarray,
            time_steps: np.ndarray,
            progress_callback: Optional[Callable[[int], None]] = None,
        ):
            """
            Initialize the batched GPU simulator.

            Args:
                N: Number of nodes in the graph
                edges: Edge array of shape (E, 2) with source and destination nodes
                time_steps: Array of time points for SIR calculation
                progress_callback: Optional callback for progress updates
                    (called with 1 after each sample)
            """
            # Configure memory pool on first use
            configure_gpu_memory_pool()

            # Transfer edges ONCE, reuse for all samples
            self._edges_gpu = cp.asarray(edges, dtype=cp.int32)
            self._time_steps_gpu = cp.asarray(time_steps, dtype=cp.float32)
            self._N = N
            self._num_edges = len(edges)
            self._num_steps = len(time_steps)
            self._progress_callback = progress_callback
            self._debug = os.environ.get("SPKMC_DEBUG") == "1"

            # Pre-allocate graph structure arrays (src/dst never change)
            # These will be populated in _prepare_graph_structure when sources are known
            self._graph_src: Optional[Any] = None
            self._graph_dst: Optional[Any] = None
            self._graph_weights: Optional[Any] = None  # Pre-allocated, updated each sample
            self._graph_df: Optional[Any] = None
            self._super_node: int = 0
            self._sources_prepared = False

        def run_samples(
            self, samples: int, sources: np.ndarray, params: Dict[str, Any]
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Run all samples with batched operations.

            Args:
                samples: Number of Monte Carlo samples to run
                sources: Array of initially infected node indices
                params: Distribution parameters dict with keys:
                        'distribution': 'gamma' or 'exponential'
                        'shape', 'scale': for gamma
                        'mu', 'lambda_val': for exponential

            Returns:
                Tuple of (S_mean, I_mean, R_mean) arrays with mean proportions at each time step
            """
            t_total_start = time_module.perf_counter()

            # 1. Batch random generation for ALL samples at once
            t_rng_start = time_module.perf_counter()
            recovery_all = self._generate_recovery_times_batched(samples, params)
            edge_times_all = self._generate_edge_times_batched(samples, params)
            t_rng_end = time_module.perf_counter()

            if self._debug:
                print(
                    f"[GPU TIMING] Batched RNG: {(t_rng_end - t_rng_start)*1000:.1f}ms "
                    f"(samples={samples}, N={self._N}, edges={self._num_edges})",
                    file=sys.stderr,
                )

            # 2. Prepare graph structure (once, reused across samples)
            sources_gpu = cp.asarray(sources, dtype=cp.int32)
            self._prepare_graph_structure(sources_gpu)

            # 3. Run SSSP for each sample (can't batch due to cuGraph limitation)
            t_sssp_start = time_module.perf_counter()
            distances_all = []

            for s in range(samples):
                dist = self._run_sssp_optimized(recovery_all[s], edge_times_all[s])
                distances_all.append(dist)

                # Call progress callback after each sample
                if self._progress_callback is not None:
                    self._progress_callback(1)

            t_sssp_end = time_module.perf_counter()

            if self._debug:
                print(
                    f"[GPU TIMING] SSSP loop: {(t_sssp_end - t_sssp_start)*1000:.1f}ms "
                    f"({samples} samples, {(t_sssp_end - t_sssp_start)*1000/samples:.1f}ms/sample)",
                    file=sys.stderr,
                )

            # 3. Batch SIR calculation for ALL samples at once
            t_sir_start = time_module.perf_counter()
            S_all, I_all, R_all = self._calculate_sir_batched(distances_all, recovery_all)
            t_sir_end = time_module.perf_counter()

            if self._debug:
                print(
                    f"[GPU TIMING] Batched SIR: {(t_sir_end - t_sir_start)*1000:.1f}ms",
                    file=sys.stderr,
                )

            # 4. Compute means across samples
            S_mean = np.mean(S_all, axis=0)
            I_mean = np.mean(I_all, axis=0)
            R_mean = np.mean(R_all, axis=0)

            t_total_end = time_module.perf_counter()
            if self._debug:
                print(
                    f"[GPU TIMING] Total batched: {(t_total_end - t_total_start)*1000:.1f}ms",
                    file=sys.stderr,
                )

            return S_mean, I_mean, R_mean

        def _generate_recovery_times_batched(
            self, samples: int, params: Dict[str, Any]
        ) -> cp.ndarray:
            """
            Generate recovery times for all samples at once.

            Args:
                samples: Number of samples
                params: Distribution parameters

            Returns:
                CuPy array of shape (samples, N) with recovery times
            """
            distribution = params.get("distribution", "exponential").lower()

            if distribution == "gamma":
                shape = params.get("shape", 2.0)
                scale = params.get("scale", 1.0)
                return cp.random.gamma(shape, scale, size=(samples, self._N), dtype=cp.float32)
            else:
                mu = params.get("mu", 1.0)
                return cp.random.exponential(1.0 / mu, size=(samples, self._N), dtype=cp.float32)

        def _generate_edge_times_batched(self, samples: int, params: Dict[str, Any]) -> cp.ndarray:
            """
            Generate edge infection times for all samples at once.

            Args:
                samples: Number of samples
                params: Distribution parameters

            Returns:
                CuPy array of shape (samples, num_edges) with infection times
            """
            lmbd = params.get("lambda_val", 1.0)
            return cp.random.exponential(
                1.0 / lmbd, size=(samples, self._num_edges), dtype=cp.float32
            )

        def _prepare_graph_structure(self, sources_gpu: cp.ndarray) -> None:
            """
            Pre-allocate graph structure arrays for SSSP.

            The src/dst arrays are fixed across all samples - only weights change.
            This method should be called once before the SSSP loop.

            Args:
                sources_gpu: Source node indices on GPU
            """
            if self._sources_prepared:
                return

            super_node = self._N
            num_sources = len(sources_gpu)
            total_edges = self._num_edges + num_sources

            # Pre-allocate src array: [original edges, super-node edges]
            self._graph_src = cp.empty(total_edges, dtype=cp.int32)
            self._graph_src[: self._num_edges] = self._edges_gpu[:, 0]
            self._graph_src[self._num_edges :] = super_node

            # Pre-allocate dst array: [original edges, source nodes]
            self._graph_dst = cp.empty(total_edges, dtype=cp.int32)
            self._graph_dst[: self._num_edges] = self._edges_gpu[:, 1]
            self._graph_dst[self._num_edges :] = sources_gpu

            # Pre-allocate weights array (will be updated each sample)
            self._graph_weights = cp.empty(total_edges, dtype=cp.float32)
            # Super-node weights are always 0
            self._graph_weights[self._num_edges :] = 0.0

            # Pre-create DataFrame with structure (weight column updated each sample)
            # This avoids DataFrame creation overhead in the loop
            self._graph_df = cudf.DataFrame(
                {"src": self._graph_src, "dst": self._graph_dst, "weight": self._graph_weights}
            )

            self._super_node = super_node
            self._sources_prepared = True

        def _run_sssp_optimized(
            self, recovery_times: cp.ndarray, edge_times: cp.ndarray
        ) -> np.ndarray:
            """
            Run optimized SSSP for a single sample.

            Optimizations:
            - Reuses pre-allocated src/dst arrays (only updates weights)
            - Reuses DataFrame structure, updates weight column in-place
            - Uses float32 throughout
            - Skips renumbering (vertices already in [0, N) range)

            Args:
                recovery_times: Recovery times for this sample (N,)
                edge_times: Edge infection times for this sample (E,)

            Returns:
                NumPy array of shortest path distances from sources to all nodes
            """
            # Compute infection weights (inf if edge_time >= recovery_time of source node)
            # Update weights in-place in pre-allocated array
            assert self._graph_weights is not None, "_prepare_graph_structure must be called first"
            assert self._graph_df is not None, "_prepare_graph_structure must be called first"
            u = self._edges_gpu[:, 0]
            self._graph_weights[: self._num_edges] = cp.where(
                edge_times >= recovery_times[u], cp.float32(np.inf), edge_times
            )

            # Update weight column in pre-existing DataFrame
            # cuDF allows in-place column update via direct assignment
            self._graph_df["weight"] = self._graph_weights

            # Build cuGraph graph using pre-allocated DataFrame
            G = cugraph.Graph(directed=True)
            # renumber=False: vertices already in [0, N] range (including super-node)
            G.from_cudf_edgelist(
                self._graph_df, source="src", destination="dst", edge_attr="weight", renumber=False
            )

            # Run SSSP from super-node
            result = cugraph.sssp(G, source=self._super_node)

            # Extract distances by vertex ID
            vertices = result["vertex"].to_numpy()
            dist_values = result["distance"].to_numpy()

            # Initialize distances to infinity (unreachable nodes)
            distances = np.full(self._N, np.inf, dtype=np.float32)

            # Filter out super-node and assign distances
            valid_mask = vertices < self._N
            valid_vertices = vertices[valid_mask]
            valid_distances = dist_values[valid_mask]
            distances[valid_vertices] = valid_distances

            distances_array: np.ndarray = np.asarray(distances)
            return distances_array

        def _calculate_sir_batched(
            self, distances_all: List[np.ndarray], recovery_all: cp.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Calculate SIR proportions for all samples in one GPU operation.

            For large networks, processes samples in chunks to avoid GPU OOM.
            Memory usage per chunk: chunk_size × steps × N × ~4 bytes (masks + intermediates)

            Args:
                distances_all: List of distance arrays (one per sample)
                recovery_all: Recovery times array of shape (samples, N)

            Returns:
                Tuple of (S, I, R) arrays of shape (samples, steps)
            """
            samples = len(distances_all)
            steps = len(self._time_steps_gpu)

            # Calculate chunk size to stay within GPU memory limits
            # Memory per sample: steps × N × 4 bytes (for masks and intermediates)
            # Target max memory: 512MB for SIR calculation
            MAX_SIR_MEMORY_BYTES = 512 * 1024 * 1024
            bytes_per_sample = steps * self._N * 4
            chunk_size = max(1, MAX_SIR_MEMORY_BYTES // bytes_per_sample)

            # If we can process all samples at once, use the fast path
            if chunk_size >= samples:
                return self._calculate_sir_batched_single(distances_all, recovery_all)

            # Process in chunks for large networks
            if self._debug:
                print(
                    f"[GPU DEBUG] SIR chunked: {samples} samples in chunks of {chunk_size} "
                    f"(N={self._N}, steps={steps})",
                    file=sys.stderr,
                )

            S_results = []
            I_results = []
            R_results = []

            for start in range(0, samples, chunk_size):
                end = min(start + chunk_size, samples)
                chunk_distances = distances_all[start:end]
                chunk_recovery = recovery_all[start:end]

                S_chunk, I_chunk, R_chunk = self._calculate_sir_batched_single(
                    chunk_distances, chunk_recovery
                )
                S_results.append(S_chunk)
                I_results.append(I_chunk)
                R_results.append(R_chunk)

            return (
                np.concatenate(S_results, axis=0),
                np.concatenate(I_results, axis=0),
                np.concatenate(R_results, axis=0),
            )

        def _calculate_sir_batched_single(
            self, distances_all: List[np.ndarray], recovery_all: cp.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Calculate SIR for a single chunk of samples (no chunking).

            Args:
                distances_all: List of distance arrays for this chunk
                recovery_all: Recovery times array for this chunk (chunk_size, N)

            Returns:
                Tuple of (S, I, R) arrays of shape (chunk_size, steps)
            """
            # Stack distances: (chunk_size, N) - transfer to GPU
            dist_gpu = cp.stack([cp.asarray(d, dtype=cp.float32) for d in distances_all])

            # Reshape for broadcasting:
            # dist_gpu: (chunk_size, N) -> (chunk_size, 1, N)
            # recovery_all: (chunk_size, N) -> (chunk_size, 1, N)
            # time_steps: (steps,) -> (1, steps, 1)
            d = dist_gpu[:, cp.newaxis, :]  # (chunk_size, 1, N)
            r = recovery_all[:, cp.newaxis, :]  # (chunk_size, 1, N)
            t = self._time_steps_gpu[cp.newaxis, :, cp.newaxis]  # (1, steps, 1)

            # Vectorized computation across chunk samples AND time steps
            # Result shapes: (chunk_size, steps, N)
            s_mask = d > t  # Susceptible: not yet infected
            i_mask = (d <= t) & (d + r > t)  # Infected: infected but not recovered

            # Sum across nodes (axis=2) and normalize
            s_frac = cp.sum(s_mask, axis=2, dtype=cp.float32) / self._N  # (chunk_size, steps)
            i_frac = cp.sum(i_mask, axis=2, dtype=cp.float32) / self._N  # (chunk_size, steps)
            r_frac = 1.0 - s_frac - i_frac  # (chunk_size, steps)

            return s_frac.get(), i_frac.get(), r_frac.get()

    # Mark GPU functions as available
    _GPU_FUNCTIONS_AVAILABLE = True

except ImportError:
    # Stub implementations when GPU dependencies are not available
    _GPU_FUNCTIONS_AVAILABLE = False

    def get_dist_gpu(
        N: int, edges: np.ndarray, sources: np.ndarray, params: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stub - GPU dependencies not installed."""
        raise ImportError("GPU dependencies not installed. Install with: pip install spkmc[gpu]")

    def calculate_gpu(
        N: int, time_to_infect: np.ndarray, recovery_times: np.ndarray, time_steps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stub - GPU dependencies not installed."""
        raise ImportError("GPU dependencies not installed. Install with: pip install spkmc[gpu]")

    # Define BatchedGPUSimulator as a stub class when GPU is not available
    class BatchedGPUSimulator:  # type: ignore[no-redef]  # noqa: N801
        """Stub class - GPU dependencies not installed."""

        def __init__(
            self,
            N: int,
            edges: np.ndarray,
            time_steps: np.ndarray,
            progress_callback: Optional[Callable[[int], None]] = None,
        ) -> None:
            raise ImportError(
                "GPU dependencies not installed. Install with: pip install spkmc[gpu]"
            )
