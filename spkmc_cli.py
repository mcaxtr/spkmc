#!/usr/bin/env python3
"""
SPKMC CLI - Command-line interface for the SPKMC algorithm

This CLI runs simulations of the Shortest Path Kinetic Monte Carlo (SPKMC) algorithm
to model epidemic spread on networks using the SIR model
(Susceptible-Infected-Recovered).

Author: SPKMC Team
"""

from spkmc.cli.commands import cli

if __name__ == "__main__":
    cli()
