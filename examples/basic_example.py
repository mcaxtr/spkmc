#!/usr/bin/env python3
"""
Exemplo básico de uso do SPKMC.

Este script demonstra como usar a API do SPKMC para executar uma simulação
de propagação de epidemias em uma rede Erdos-Renyi com distribuição Gamma.
"""

import numpy as np
import matplotlib.pyplot as plt

from spkmc import SPKMC, GammaDistribution, NetworkFactory


def main():
    """Função principal do exemplo."""
    # Configuração dos parâmetros
    N = 1000  # Número de nós
    k_avg = 10  # Grau médio
    samples = 50  # Número de amostras
    initial_perc = 0.01  # Porcentagem inicial de infectados
    
    # Configuração do tempo
    t_max = 10.0
    steps = 100
    time_steps = np.linspace(0, t_max, steps)
    
    # Criação da distribuição Gamma
    gamma_dist = GammaDistribution(shape=2.0, scale=1.0, lmbd=1.0)
    
    # Criação do simulador SPKMC
    simulator = SPKMC(gamma_dist)
    
    print("Simulando rede Erdos-Renyi com distribuição Gamma...")
    
    # Criação da rede Erdos-Renyi
    G = NetworkFactory.create_erdos_renyi(N, k_avg)
    
    # Configuração dos nós inicialmente infectados
    init_infect = int(N * initial_perc)
    sources = np.random.randint(0, N, init_infect)
    
    # Execução da simulação
    S, I, R = simulator.run_multiple_simulations(G, sources, time_steps, samples)
    
    # Visualização dos resultados
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, S, 'b-', label='Suscetíveis')
    plt.plot(time_steps, I, 'r-', label='Infectados')
    plt.plot(time_steps, R, 'g-', label='Recuperados')
    
    plt.xlabel('Tempo')
    plt.ylabel('Proporção de Indivíduos')
    plt.title('Dinâmica do Modelo SIR ao Longo do Tempo')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()