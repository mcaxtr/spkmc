"""
Network classes for the SPKMC algorithm.

This module contains implementations of different network types that can be
used in SPKMC simulations, such as Erdos-Renyi networks, complex networks, and complete graphs.

Includes fast edge-list generators that bypass NetworkX for GPU workflows.
"""

from typing import Any, Dict, Tuple

import networkx as nx
import numpy as np


class NetworkFactory:
    """Factory for creating different network types."""

    NETWORK_TYPES = ["er", "cn", "cg", "rrn"]  # Supported network types

    @staticmethod
    def create_network(network_type: str, **kwargs: Any) -> nx.DiGraph:
        """
        Create a network based on the type and provided parameters.

        Args:
            network_type: Network type ('er', 'cn', 'cg', 'rrn')
            **kwargs: Network-specific parameters

        Returns:
            Directed graph of the requested network

        Raises:
            ValueError: If the network type is unknown
        """
        network_type = network_type.lower()

        if network_type == "er":
            N = kwargs.get("N", 1000)
            k_avg = kwargs.get("k_avg", 10)
            return NetworkFactory.create_erdos_renyi(N, k_avg)

        elif network_type == "cn":
            N = kwargs.get("N", 1000)
            exponent = kwargs.get("exponent", 2.5)
            k_avg = kwargs.get("k_avg", 10)
            return NetworkFactory.create_complex_network(N, exponent, k_avg)

        elif network_type == "cg":
            N = kwargs.get("N", 1000)
            return NetworkFactory.create_complete_graph(N)

        elif network_type == "rrn":
            N = kwargs.get("N", 1000)
            k_avg = kwargs.get("k_avg", 10)
            return NetworkFactory.create_random_regular_network(N, k_avg)

        else:
            raise ValueError(f"Unknown network type: {network_type}")

    @staticmethod
    def create_erdos_renyi(N: int, k_avg: float) -> nx.DiGraph:
        """
        Create an Erdos-Renyi network.

        Args:
            N: Number of nodes
            k_avg: Average degree

        Returns:
            Directed Erdos-Renyi graph
        """
        p = k_avg / (N - 1)
        return nx.erdos_renyi_graph(N, p, directed=True)

    @staticmethod
    def generate_discrete_power_law(n: int, alpha: float, xmin: int, xmax: float) -> np.ndarray:
        """
        Generate a discrete power-law sequence.

        Args:
            n: Number of elements
            alpha: Power-law exponent
            xmin: Minimum value
            xmax: Maximum value

        Returns:
            Array with the power-law sequence
        """
        rand_nums = np.random.uniform(size=n)
        power_law_seq = (xmax ** (1 - alpha) - xmin ** (1 - alpha)) * rand_nums + xmin ** (
            1 - alpha
        )
        power_law_seq = power_law_seq ** (1 / (1 - alpha))
        power_law_seq = power_law_seq.astype(int)

        total_sum = int(sum(power_law_seq))
        if total_sum % 2 == 0:
            result: np.ndarray = power_law_seq
            return result
        else:
            power_law_seq[0] = power_law_seq[0] + 1
            result = power_law_seq
            return result

    @staticmethod
    def create_complex_network(N: int, exponent: float, k_avg: float) -> nx.DiGraph:
        """
        Create a complex network with a power-law degree distribution.

        Args:
            N: Number of nodes
            exponent: Power-law exponent
            k_avg: Average degree

        Returns:
            Directed complex graph
        """
        degree_sequence = NetworkFactory.generate_discrete_power_law(N, exponent, 2, np.sqrt(N))
        degree_sequence = np.round(degree_sequence * (k_avg / np.mean(degree_sequence))).astype(int)

        if sum(degree_sequence) % 2 != 0:
            degree_sequence[np.argmin(degree_sequence)] += 1

        G = nx.configuration_model(degree_sequence)
        G = nx.DiGraph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        return G

    @staticmethod
    def create_complete_graph(N: int) -> nx.DiGraph:
        """
        Create a complete graph.

        Args:
            N: Number of nodes

        Returns:
            Directed complete graph
        """
        return nx.complete_graph(N, create_using=nx.DiGraph())

    @staticmethod
    def create_random_regular_network(N: int, k_avg: float) -> nx.DiGraph:
        """
        Create a random regular network.

        Args:
            N: Number of nodes
            k_avg: Regular degree (connections per node, will be converted to int)

        Returns:
            Directed random regular graph
        """
        # k_avg must be int for random_regular_graph
        G = nx.random_regular_graph(int(k_avg), N)
        return nx.DiGraph(G)

    @staticmethod
    def get_network_info(network_type: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Return network information for metadata.

        Args:
            network_type: Network type ('er', 'cn', 'cg', 'rrn')
            **kwargs: Network-specific parameters

        Returns:
            Dictionary with network information
        """
        network_type = network_type.lower()

        info = {"type": network_type, "N": kwargs.get("N", 1000)}

        if network_type in ["er", "cn", "rrn"]:
            info["k_avg"] = kwargs.get("k_avg", 10)

        if network_type == "cn":
            info["exponent"] = kwargs.get("exponent", 2.5)

        return info

    # =========================================================================
    # Fast Edge-List Generators (bypass NetworkX for GPU workflows)
    # =========================================================================

    @staticmethod
    def create_edges(network_type: str, **kwargs: Any) -> Tuple[int, np.ndarray]:
        """
        Create edge list directly without NetworkX graph object.

        This is significantly faster than NetworkX for large graphs,
        especially for GPU workflows where only the edge array is needed.

        Args:
            network_type: Type of network ('er', 'cn', 'cg', 'rrn')
            **kwargs: Network-specific parameters (N, k_avg, exponent)

        Returns:
            Tuple of (N, edges) where edges is ndarray of shape (E, 2)

        Raises:
            ValueError: If network type is unknown
        """
        network_type = network_type.lower()

        if network_type == "er":
            N = kwargs.get("N", 1000)
            k_avg = kwargs.get("k_avg", 10)
            return NetworkFactory.create_erdos_renyi_edges(N, k_avg)

        elif network_type == "cn":
            N = kwargs.get("N", 1000)
            exponent = kwargs.get("exponent", 2.5)
            k_avg = kwargs.get("k_avg", 10)
            return NetworkFactory.create_complex_network_edges(N, exponent, k_avg)

        elif network_type == "cg":
            N = kwargs.get("N", 1000)
            return NetworkFactory.create_complete_graph_edges(N)

        elif network_type == "rrn":
            N = kwargs.get("N", 1000)
            k_avg = kwargs.get("k_avg", 10)
            return NetworkFactory.create_random_regular_edges(N, k_avg)

        else:
            raise ValueError(f"Unknown network type: {network_type}")

    @staticmethod
    def create_erdos_renyi_edges(N: int, k_avg: float) -> Tuple[int, np.ndarray]:
        """
        Create Erdos-Renyi G(n,p) edge list using Batagelj-Brandes algorithm.

        Uses geometric distribution to skip edges efficiently, giving exact
        G(n,p) distribution in O(n+m) time instead of O(n²).

        Based on: V. Batagelj and U. Brandes, "Efficient generation of large
        random networks", Phys. Rev. E, 71, 036113, 2005.

        Args:
            N: Number of nodes
            k_avg: Average degree (p = k_avg / (N-1))

        Returns:
            Tuple of (N, edges) where edges is ndarray of shape (E, 2)
        """
        p = k_avg / (N - 1)

        if p <= 0:
            return N, np.empty((0, 2), dtype=np.int32)
        if p >= 1:
            return NetworkFactory.create_complete_graph_edges(N)

        # For very dense graphs (p > 0.5), matrix method is faster
        if p > 0.5:
            adj = np.random.random((N, N)) < p
            np.fill_diagonal(adj, False)
            edges = np.column_stack(np.where(adj))
            return N, edges.astype(np.int32)

        # Batagelj-Brandes algorithm for sparse G(n,p)
        # Total possible directed edges (excluding self-loops)
        max_edges = N * (N - 1)
        expected_edges = int(max_edges * p)

        # Pre-allocate with buffer for geometric sampling
        # Expected samples needed ≈ expected_edges, add 50% buffer
        num_samples = int(expected_edges * 1.5) + 100

        # Sample geometric distribution: floor(log(U) / log(1-p))
        # This gives number of edges to skip before next inclusion
        lp = np.log(1.0 - p)
        U = np.random.random(num_samples)
        # Avoid log(0) by using 1-U instead of U
        skips = (np.log(U) / lp).astype(np.int64)

        # Compute cumulative edge indices
        # Start at -1, then idx = prev_idx + skip + 1
        indices = np.cumsum(skips + 1) - 1

        # Keep only valid indices (< max_edges)
        valid_mask = indices < max_edges
        indices = indices[valid_mask]

        if len(indices) == 0:
            return N, np.empty((0, 2), dtype=np.int32)

        # Convert linear index to (v, w) for directed graph
        # Edge enumeration (skipping self-loops):
        # idx 0 -> (0,1), idx 1 -> (0,2), ..., idx N-2 -> (0,N-1)
        # idx N-1 -> (1,0), idx N -> (1,2), ..., idx 2N-3 -> (1,N-1)
        # v = idx // (N-1)
        # w_offset = idx % (N-1)
        # w = w_offset if w_offset < v else w_offset + 1
        v = (indices // (N - 1)).astype(np.int32)
        w_offset = (indices % (N - 1)).astype(np.int32)
        w = np.where(w_offset >= v, w_offset + 1, w_offset).astype(np.int32)

        edges = np.column_stack([v, w])
        return N, edges

    @staticmethod
    def create_complex_network_edges(
        N: int, exponent: float, k_avg: float
    ) -> Tuple[int, np.ndarray]:
        """
        Create scale-free network edge list using configuration model.

        Generates power-law degree sequence and creates edges via stub-pairing.
        Multi-edges and self-loops are removed, which slightly alters the
        degree distribution but preserves the power-law shape.

        Note: This is approximately equivalent to NetworkX's configuration_model
        converted to DiGraph, suitable for epidemic simulations where exact
        degree sequence is less important than overall network structure.

        Args:
            N: Number of nodes
            exponent: Power-law exponent
            k_avg: Average degree

        Returns:
            Tuple of (N, edges) where edges is ndarray of shape (E, 2)
        """
        # Generate power-law degree sequence
        degree_sequence = NetworkFactory.generate_discrete_power_law(N, exponent, 2, np.sqrt(N))
        degree_sequence = np.round(degree_sequence * (k_avg / np.mean(degree_sequence))).astype(
            np.int32
        )

        # Ensure even sum
        if np.sum(degree_sequence) % 2 != 0:
            degree_sequence[np.argmin(degree_sequence)] += 1

        # Create stubs (half-edges)
        stubs = np.repeat(np.arange(N, dtype=np.int32), degree_sequence)

        # Shuffle and pair stubs to create edges
        np.random.shuffle(stubs)

        # Pair adjacent stubs
        num_edges = len(stubs) // 2
        src = stubs[:num_edges]
        dst = stubs[num_edges : 2 * num_edges]

        # Remove self-loops
        valid_mask = src != dst
        src = src[valid_mask]
        dst = dst[valid_mask]

        # Remove multi-edges using unique
        packed = src.astype(np.int64) * N + dst.astype(np.int64)
        unique_packed = np.unique(packed)

        src_unique = (unique_packed // N).astype(np.int32)
        dst_unique = (unique_packed % N).astype(np.int32)

        # Make directed (add reverse edges)
        edges = np.vstack(
            [np.column_stack([src_unique, dst_unique]), np.column_stack([dst_unique, src_unique])]
        )

        # Remove any duplicates from making directed
        packed = edges[:, 0].astype(np.int64) * N + edges[:, 1].astype(np.int64)
        unique_packed = np.unique(packed)
        src_final = (unique_packed // N).astype(np.int32)
        dst_final = (unique_packed % N).astype(np.int32)

        return N, np.column_stack([src_final, dst_final])

    @staticmethod
    def create_complete_graph_edges(N: int) -> Tuple[int, np.ndarray]:
        """
        Create complete graph edge list.

        All N*(N-1) directed edges between N nodes.

        Args:
            N: Number of nodes

        Returns:
            Tuple of (N, edges) where edges is ndarray of shape (N*(N-1), 2)
        """
        # Create all pairs (i, j) where i != j
        src, dst = np.meshgrid(np.arange(N, dtype=np.int32), np.arange(N, dtype=np.int32))
        src = src.ravel()
        dst = dst.ravel()

        # Remove self-loops
        mask = src != dst
        edges = np.column_stack([src[mask], dst[mask]])

        return N, edges

    @staticmethod
    def create_random_regular_edges(N: int, k_avg: int) -> Tuple[int, np.ndarray]:
        """
        Create random regular graph edge list.

        Each node has exactly k_avg neighbors (for undirected base graph,
        then converted to directed with 2*k_avg edges per node).

        Uses stub-pairing algorithm with rejection for invalid pairings.
        Falls back to NetworkX if too many attempts fail.

        Args:
            N: Number of nodes
            k_avg: Degree (must be even if N*k_avg is odd)

        Returns:
            Tuple of (N, edges) where edges is ndarray of shape (E, 2)
        """
        # Ensure N*k_avg is even (required for regular graph)
        if (N * k_avg) % 2 != 0:
            raise ValueError(f"N*k_avg must be even. Got N={N}, k_avg={k_avg}")

        # Create stubs (half-edges): k_avg stubs per node
        stubs = np.repeat(np.arange(N, dtype=np.int32), k_avg)
        num_stubs = len(stubs)
        num_edges = num_stubs // 2

        # Try multiple times to get valid pairing (no self-loops, no multi-edges)
        max_attempts = 100
        for _attempt in range(max_attempts):
            shuffled = stubs.copy()
            np.random.shuffle(shuffled)

            # Pair stubs: first half with second half
            src = shuffled[:num_edges]
            dst = shuffled[num_edges : 2 * num_edges]

            # Check for self-loops: any src[i] == dst[i]?
            has_self_loops = np.any(src == dst)
            if has_self_loops:
                continue  # Retry with new shuffle

            # Check for multi-edges: sort each edge (a,b) -> (min,max), then check unique
            edges_sorted = np.sort(np.column_stack([src, dst]), axis=1)
            packed_int = edges_sorted[:, 0].astype(np.int64) * N + edges_sorted[:, 1].astype(
                np.int64
            )
            unique_packed = np.unique(packed_int)

            if len(unique_packed) == num_edges:
                # Valid simple graph: no self-loops, no multi-edges
                # Make directed by adding both directions
                edges = np.vstack([np.column_stack([src, dst]), np.column_stack([dst, src])])
                return N, edges

        # Fallback to NetworkX if fast method fails after max_attempts
        # This can happen for certain N, k_avg combinations
        G = nx.random_regular_graph(k_avg, N)
        G = nx.DiGraph(G)
        edges = np.array(list(G.edges()), dtype=np.int32)
        return N, edges
