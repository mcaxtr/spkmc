"""
Tests for fast edge-list generators in NetworkFactory.

Verifies that fast generators produce statistically equivalent graphs
to NetworkX implementations.
"""

import numpy as np

from spkmc.core.networks import NetworkFactory


class TestErdosRenyiEdges:
    """Tests for fast Erdos-Renyi edge generation."""

    def test_expected_edge_count(self):
        """Edge count should match G(n,p) expectation."""
        N = 1000
        k_avg = 10
        p = k_avg / (N - 1)
        expected_edges = N * (N - 1) * p

        # Run multiple trials
        edge_counts = []
        for _ in range(50):
            _, edges = NetworkFactory.create_erdos_renyi_edges(N, k_avg)
            edge_counts.append(len(edges))

        mean_edges = np.mean(edge_counts)
        std_edges = np.std(edge_counts)

        # Mean should be close to expected (within 5%)
        assert (
            abs(mean_edges - expected_edges) / expected_edges < 0.05
        ), f"Mean edges {mean_edges} differs from expected {expected_edges}"

        # Standard deviation should match binomial (sqrt(n*p*(1-p)))
        expected_std = np.sqrt(N * (N - 1) * p * (1 - p))
        assert (
            std_edges < expected_std * 2
        ), f"Std {std_edges} much larger than expected {expected_std}"

    def test_no_self_loops(self):
        """Should not contain self-loops."""
        N = 500
        k_avg = 10

        for _ in range(10):
            _, edges = NetworkFactory.create_erdos_renyi_edges(N, k_avg)
            self_loops = edges[:, 0] == edges[:, 1]
            assert not np.any(self_loops), "Found self-loops in ER graph"

    def test_no_duplicate_edges(self):
        """Should not contain duplicate edges."""
        N = 500
        k_avg = 10

        for _ in range(10):
            _, edges = NetworkFactory.create_erdos_renyi_edges(N, k_avg)
            # Pack edges to single int for uniqueness check
            packed = edges[:, 0].astype(np.int64) * N + edges[:, 1].astype(np.int64)
            assert len(np.unique(packed)) == len(packed), "Found duplicate edges"

    def test_degree_distribution(self):
        """Out-degree should follow Binomial(N-1, p)."""
        N = 500
        k_avg = 10
        # p = k_avg / (N - 1)  # theoretical probability (used for mean validation)

        # Aggregate degree distribution over multiple graphs
        all_degrees = []
        for _ in range(20):
            _, edges = NetworkFactory.create_erdos_renyi_edges(N, k_avg)
            # Count out-degree for each node
            out_degrees = np.bincount(edges[:, 0], minlength=N)
            all_degrees.extend(out_degrees)

        # Mean degree should be close to (N-1)*p = k_avg
        mean_degree = np.mean(all_degrees)
        assert (
            abs(mean_degree - k_avg) < 0.5
        ), f"Mean degree {mean_degree} differs from expected {k_avg}"

    def test_handles_edge_cases(self):
        """Should handle edge cases correctly."""
        N = 100

        # Very sparse graph
        _, edges = NetworkFactory.create_erdos_renyi_edges(N, k_avg=0.1)
        assert len(edges) < N  # Should have very few edges

        # Dense graph (p > 0.5 triggers matrix method)
        _, edges = NetworkFactory.create_erdos_renyi_edges(N, k_avg=60)
        assert len(edges) > N * 50  # Should have many edges


class TestCompleteGraphEdges:
    """Tests for complete graph edge generation."""

    def test_edge_count(self):
        """Should have N*(N-1) directed edges."""
        for N in [10, 50, 100]:
            _, edges = NetworkFactory.create_complete_graph_edges(N)
            expected = N * (N - 1)
            assert len(edges) == expected, f"Expected {expected} edges, got {len(edges)}"

    def test_no_self_loops(self):
        """Should not contain self-loops."""
        N = 50
        _, edges = NetworkFactory.create_complete_graph_edges(N)
        self_loops = edges[:, 0] == edges[:, 1]
        assert not np.any(self_loops), "Found self-loops in complete graph"

    def test_all_pairs_present(self):
        """Should contain all (i,j) pairs where i != j."""
        N = 20
        _, edges = NetworkFactory.create_complete_graph_edges(N)

        # Create set of all edges
        edge_set = set(map(tuple, edges))

        # Check all pairs are present
        for i in range(N):
            for j in range(N):
                if i != j:
                    assert (i, j) in edge_set, f"Missing edge ({i}, {j})"


class TestRandomRegularEdges:
    """Tests for random regular graph edge generation."""

    def test_regularity(self):
        """Each node should have exactly k_avg neighbors (undirected degree)."""
        N = 100
        k_avg = 10

        for _ in range(5):
            _, edges = NetworkFactory.create_random_regular_edges(N, k_avg)

            # Count degree (both directions since it's directed)
            # For undirected regularity, count unique neighbors
            out_degree = np.bincount(edges[:, 0], minlength=N)

            # Each node should have out-degree = k_avg
            # (since we add both directions, in-degree = out-degree = k_avg)
            assert np.all(
                out_degree == k_avg
            ), f"Not all nodes have degree {k_avg}: {np.unique(out_degree)}"

    def test_no_self_loops(self):
        """Should not contain self-loops."""
        N = 100
        k_avg = 10

        for _ in range(5):
            _, edges = NetworkFactory.create_random_regular_edges(N, k_avg)
            self_loops = edges[:, 0] == edges[:, 1]
            assert not np.any(self_loops), "Found self-loops"

    def test_symmetric(self):
        """Directed graph should be symmetric (edge (a,b) implies (b,a))."""
        N = 100
        k_avg = 10

        _, edges = NetworkFactory.create_random_regular_edges(N, k_avg)
        edge_set = set(map(tuple, edges))

        for src, dst in edges:
            assert (dst, src) in edge_set, f"Edge ({src},{dst}) but not ({dst},{src})"


class TestComplexNetworkEdges:
    """Tests for scale-free network edge generation."""

    def test_approximate_average_degree(self):
        """Average degree should be approximately k_avg."""
        N = 1000
        k_avg = 10
        exponent = 2.5

        degrees = []
        for _ in range(10):
            _, edges = NetworkFactory.create_complex_network_edges(N, exponent, k_avg)
            # Out-degree (directed)
            out_degree = np.bincount(edges[:, 0], minlength=N)
            degrees.extend(out_degree)

        mean_degree = np.mean(degrees)
        # Allow 20% tolerance due to multi-edge removal
        assert (
            abs(mean_degree - k_avg) / k_avg < 0.2
        ), f"Mean degree {mean_degree} differs too much from {k_avg}"

    def test_no_self_loops(self):
        """Should not contain self-loops."""
        N = 500
        k_avg = 10
        exponent = 2.5

        _, edges = NetworkFactory.create_complex_network_edges(N, exponent, k_avg)
        self_loops = edges[:, 0] == edges[:, 1]
        assert not np.any(self_loops), "Found self-loops"

    def test_power_law_shape(self):
        """Degree distribution should roughly follow power-law."""
        N = 2000
        k_avg = 10
        exponent = 2.5

        _, edges = NetworkFactory.create_complex_network_edges(N, exponent, k_avg)
        out_degree = np.bincount(edges[:, 0], minlength=N)

        # Check that high-degree nodes exist (power-law has heavy tail)
        max_degree = np.max(out_degree)
        assert (
            max_degree > k_avg * 3
        ), f"Max degree {max_degree} too low for power-law (expected heavy tail)"

        # Check degree variance is high (power-law has high variance)
        degree_std = np.std(out_degree)
        assert degree_std > k_avg * 0.5, f"Degree std {degree_std} too low for power-law"


class TestCompareWithNetworkX:
    """Compare fast generators with NetworkX for statistical equivalence."""

    def test_er_vs_networkx(self):
        """Fast ER should produce statistically similar graphs to NetworkX."""
        N = 500
        k_avg = 10
        num_trials = 20

        fast_edge_counts = []
        nx_edge_counts = []

        for _ in range(num_trials):
            # Fast generator
            _, edges = NetworkFactory.create_erdos_renyi_edges(N, k_avg)
            fast_edge_counts.append(len(edges))

            # NetworkX
            G = NetworkFactory.create_erdos_renyi(N, k_avg)
            nx_edge_counts.append(G.number_of_edges())

        # Compare means (should be very close)
        fast_mean = np.mean(fast_edge_counts)
        nx_mean = np.mean(nx_edge_counts)

        # Allow 10% difference
        assert (
            abs(fast_mean - nx_mean) / nx_mean < 0.1
        ), f"Fast mean {fast_mean} differs from NX mean {nx_mean}"

        # Compare standard deviations (should be similar)
        fast_std = np.std(fast_edge_counts)
        nx_std = np.std(nx_edge_counts)

        # Allow 100% difference in std (variance in random generators can differ significantly)
        assert (
            abs(fast_std - nx_std) / max(nx_std, 1) < 1.0
        ), f"Fast std {fast_std} differs from NX std {nx_std}"
