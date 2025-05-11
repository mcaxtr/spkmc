#!/usr/bin/env python3
"""
SPKMC CLI - Interface de linha de comando para o algoritmo SPKMC

Esta CLI permite executar simulações do algoritmo Shortest Path Kinetic Monte Carlo (SPKMC)
para modelagem de propagação de epidemias em redes, utilizando o modelo SIR 
(Susceptible-Infected-Recovered).

Autor: SPKMC Team
"""

from spkmc.cli.commands import cli

if __name__ == "__main__":
    cli()