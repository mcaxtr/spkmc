#!/usr/bin/env python3
"""
Basic example of using SPKMC.

This script demonstrates how to use the SPKMC API to run an epidemic spread
simulation on an Erdos-Renyi network with a Gamma distribution.
"""

import matplotlib.pyplot as plt
import numpy as np

from spkmc import SPKMC, GammaDistribution, NetworkFactory


def main():
    """Main function for the example."""
    # Parameter configuration
    N = 1000  # Number of nodes
    k_avg = 10  # Average degree
    samples = 50  # Number of samples
    initial_perc = 0.01  # Initial percentage of infected

    # Time configuration
    t_max = 10.0
    steps = 100
    time_steps = np.linspace(0, t_max, steps)

    # Create the Gamma distribution
    gamma_dist = GammaDistribution(shape=2.0, scale=1.0, lmbd=1.0)

    # Create the SPKMC simulator
    simulator = SPKMC(gamma_dist)

    print("Simulating an Erdos-Renyi network with a Gamma distribution...")

    # Create the Erdos-Renyi network
    G = NetworkFactory.create_erdos_renyi(N, k_avg)

    # Configure initially infected nodes
    init_infect = int(N * initial_perc)
    sources = np.random.randint(0, N, init_infect)

    # Run the simulation
    S, I, R = simulator.run_multiple_simulations(G, sources, time_steps, samples)

    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, S, "b-", label="Susceptible")
    plt.plot(time_steps, I, "r-", label="Infected")
    plt.plot(time_steps, R, "g-", label="Recovered")

    plt.xlabel("Time")
    plt.ylabel("Proportion of Individuals")
    plt.title("SIR Model Dynamics Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
