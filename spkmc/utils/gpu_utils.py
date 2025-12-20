"""
GPU utilities for SPKMC acceleration.

Provides optional GPU acceleration using CuPy, cuDF, and cuGraph.
All functions gracefully fall back to raising ImportError if dependencies are missing.

To install GPU dependencies:
    pip install spkmc[gpu]
"""

import warnings
from typing import Tuple, Dict, Any, Optional
import numpy as np

# Global flag for GPU availability (cached after first check)
_GPU_AVAILABLE: Optional[bool] = None
_GPU_CHECK_ERROR: Optional[str] = None


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
        import cupy as cp
        import cudf
        import cugraph

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
    global _GPU_AVAILABLE, _GPU_CHECK_ERROR
    _GPU_AVAILABLE = None
    _GPU_CHECK_ERROR = None


# Conditional imports and GPU implementations
try:
    import cupy as cp
    import cudf
    import cugraph

    def get_dist_gpu(
        N: int,
        edges: np.ndarray,
        sources: np.ndarray,
        params: Dict[str, Any]
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
        # Transfer edges to GPU
        edges_gpu = cp.asarray(edges, dtype=cp.int32)
        sources_gpu = cp.asarray(sources, dtype=cp.int32)

        # Sample recovery and infection times on GPU
        distribution = params.get('distribution', 'exponential').lower()

        if distribution == 'gamma':
            shape = params.get('shape', 2.0)
            scale = params.get('scale', 1.0)
            recovery_times = cp.random.gamma(shape, scale, size=N)
            edge_times = cp.random.gamma(shape, scale, size=edges_gpu.shape[0])
        else:
            mu = params.get('mu', 1.0)
            recovery_times = cp.random.exponential(1.0 / mu, size=N)
            lmbd = params.get('lambda_val', 1.0)
            edge_times = cp.random.exponential(1.0 / lmbd, size=edges_gpu.shape[0])

        # Compute infection times (inf if >= recovery time)
        u = edges_gpu[:, 0]
        infection_weights = cp.where(
            edge_times >= recovery_times[u],
            cp.inf,
            edge_times
        )

        # Create super-node for multi-source SSSP
        super_node = N
        super_edges_src = cp.full(len(sources_gpu), super_node, dtype=cp.int32)
        super_edges_dst = sources_gpu
        super_weights = cp.zeros(len(sources_gpu), dtype=cp.float32)

        # Concatenate edges
        all_src = cp.concatenate([edges_gpu[:, 0], super_edges_src])
        all_dst = cp.concatenate([edges_gpu[:, 1], super_edges_dst])
        all_weights = cp.concatenate([
            infection_weights.astype(cp.float32),
            super_weights
        ])

        # Build cuGraph graph
        df = cudf.DataFrame({
            'src': all_src,
            'dst': all_dst,
            'weight': all_weights
        })
        G = cugraph.Graph(directed=True)
        G.from_cudf_edgelist(df, source='src', destination='dst', edge_attr='weight')

        # Run SSSP from super-node
        result = cugraph.sssp(G, source=super_node)

        # Extract distances (excluding super-node)
        distances = result['distance'].to_numpy()[:N]

        return distances, recovery_times.get()

    def calculate_gpu(
        N: int,
        time_to_infect: np.ndarray,
        recovery_times: np.ndarray,
        time_steps: np.ndarray
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
        time_to_infect_gpu = cp.asarray(time_to_infect)
        recovery_times_gpu = cp.asarray(recovery_times)

        steps = len(time_steps)
        S_time = cp.zeros(steps)
        I_time = cp.zeros(steps)
        R_time = cp.zeros(steps)

        for idx, t in enumerate(time_steps):
            S = time_to_infect_gpu > t
            I = (~S) & (time_to_infect_gpu + recovery_times_gpu > t)
            R = (~S) & (~I)

            S_time[idx] = cp.sum(S) / N
            I_time[idx] = cp.sum(I) / N
            R_time[idx] = cp.sum(R) / N

        return S_time.get(), I_time.get(), R_time.get()

    # Mark GPU functions as available
    _GPU_FUNCTIONS_AVAILABLE = True

except ImportError:
    # Stub implementations when GPU dependencies are not available
    _GPU_FUNCTIONS_AVAILABLE = False

    def get_dist_gpu(
        N: int,
        edges: np.ndarray,
        sources: np.ndarray,
        params: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stub - GPU dependencies not installed."""
        raise ImportError(
            "GPU dependencies not installed. Install with: pip install spkmc[gpu]"
        )

    def calculate_gpu(
        N: int,
        time_to_infect: np.ndarray,
        recovery_times: np.ndarray,
        time_steps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stub - GPU dependencies not installed."""
        raise ImportError(
            "GPU dependencies not installed. Install with: pip install spkmc[gpu]"
        )
