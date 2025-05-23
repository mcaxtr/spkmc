"""
Implementação principal do algoritmo SPKMC.

Este módulo contém a implementação do algoritmo Shortest Path Kinetic Monte Carlo (SPKMC)
para simulação de propagação de epidemias em redes, utilizando o modelo SIR 
(Susceptible-Infected-Recovered).
"""

import networkx as nx
import numpy as np
from tqdm import tqdm
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
import os
from typing import Dict, List, Tuple, Union, Optional, Any

from spkmc.core.distributions import Distribution
from spkmc.core.networks import NetworkFactory
from spkmc.io.results import ResultManager
from spkmc.utils.numba_utils import calculate


class SPKMC:
    """
    Implementação do algoritmo Shortest Path Kinetic Monte Carlo (SPKMC).
    
    Esta classe implementa o algoritmo SPKMC para simulação de propagação de epidemias
    em redes, utilizando o modelo SIR (Susceptible-Infected-Recovered).
    """
    
    def __init__(self, distribution: Distribution):
        """
        Inicializa o simulador SPKMC.
        
        Args:
            distribution: Objeto de distribuição a ser usado na simulação
        """
        self.distribution = distribution
    
    def get_dist_sparse(self, N: int, edges: np.ndarray, sources: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula as distâncias mínimas dos nós de origem para todos os outros nós.
        
        Args:
            N: Número de nós
            edges: Arestas do grafo como matriz (u, v)
            sources: Nós de origem
            
        Returns:
            Tupla com (distâncias, tempos de recuperação)
        """
        # Gera os tempos de recuperação
        recovery_weights = self.distribution.get_recovery_weights(N)
        
        # Calcula os tempos de infecção
        infection_times = self.distribution.get_infection_times(recovery_weights, edges)
        
        # Cria a matriz esparsa do grafo
        row_indices = edges[:, 0]
        col_indices = edges[:, 1]
        graph_matrix = csr_matrix((infection_times, (row_indices, col_indices)), shape=(N, N))
        
        # Calcula as distâncias mínimas
        dist_matrix = dijkstra(csgraph=graph_matrix, directed=True, indices=sources, return_predecessors=False)
        dist = np.min(dist_matrix, axis=0)
        
        return dist, recovery_weights
    
    def run_single_simulation(self, N: int, edges: np.ndarray, sources: np.ndarray, 
                             time_steps: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Executa uma única simulação SPKMC.
        
        Args:
            N: Número de nós
            edges: Arestas do grafo como matriz (u, v)
            sources: Nós de origem
            time_steps: Array com os passos de tempo
            
        Returns:
            Tupla com (S, I, R) contendo a proporção de indivíduos em cada estado
        """
        # Calcula os tempos de infecção e recuperação
        time_to_infect, recovery_times = self.get_dist_sparse(N, edges, sources)
        
        # Calcula os estados para cada passo de tempo
        steps = time_steps.shape[0]
        return calculate(N, time_to_infect, recovery_times, time_steps, steps)
    
    def run_multiple_simulations(self, G: nx.DiGraph, sources: np.ndarray, time_steps: np.ndarray, 
                                samples: int, show_progress: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Executa múltiplas simulações SPKMC e retorna a média.
        
        Args:
            G: Grafo da rede
            sources: Nós de origem
            time_steps: Array com os passos de tempo
            samples: Número de amostras
            show_progress: Se True, mostra barra de progresso
            
        Returns:
            Tupla com (S_mean, I_mean, R_mean) contendo a média da proporção de indivíduos em cada estado
        """
        steps = time_steps.shape[0]
        
        S_values = np.zeros((samples, steps))
        I_values = np.zeros((samples, steps))
        R_values = np.zeros((samples, steps))
        
        edges = np.array(G.edges())
        N = G.number_of_nodes()
        
        # Configura a barra de progresso
        if show_progress:
            sample_items = tqdm(range(samples), desc="Amostras")
        else:
            sample_items = range(samples)
        
        # Executa as simulações
        for sample in sample_items:
            S, I, R = self.run_single_simulation(N, edges, sources, time_steps)
            S_values[sample, :] = S
            I_values[sample, :] = I
            R_values[sample, :] = R
        
        # Calcula as médias
        S_mean = np.mean(S_values, axis=0)
        I_mean = np.mean(I_values, axis=0)
        R_mean = np.mean(R_values, axis=0)
        
        return S_mean, I_mean, R_mean
    
    def simulate_erdos_renyi(self, num_runs: int, time_steps: np.ndarray, N: int = 3000, 
                            k_avg: float = 10, samples: int = 100, initial_perc: float = 0.01, 
                            load_if_exists: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                                np.ndarray, np.ndarray, np.ndarray]:
        """
        Simula a propagação em múltiplas redes Erdos-Renyi.
        
        Args:
            num_runs: Número de execuções
            time_steps: Array com os passos de tempo
            N: Número de nós
            k_avg: Grau médio
            samples: Número de amostras por execução
            initial_perc: Porcentagem inicial de infectados
            load_if_exists: Se True, carrega resultados existentes
            
        Returns:
            Tupla com (S_avg, I_avg, R_avg, S_err, I_err, R_err)
        """
        S_list, I_list, R_list = [], [], []
        
        # Verifica se já existem resultados salvos
        if load_if_exists:
            result_path = ResultManager.get_result_path("ER", self.distribution, N, samples)
            if os.path.exists(result_path):
                try:
                    result = ResultManager.load_result(result_path)
                    return (
                        np.array(result.get('S_val', [])), 
                        np.array(result.get('I_val', [])), 
                        np.array(result.get('R_val', [])),
                        np.array(result.get('S_err', [])), 
                        np.array(result.get('I_err', [])), 
                        np.array(result.get('R_err', []))
                    )
                except Exception as e:
                    print(f"Erro ao carregar resultados existentes: {e}")
        
        # Executa as simulações
        for run in tqdm(range(num_runs), desc="Execuções"):
            # Cria a rede
            G = NetworkFactory.create_erdos_renyi(N, k_avg)
            
            # Configura os nós inicialmente infectados
            init_infect = int(N * initial_perc)
            if init_infect < 1:
                raise ValueError(f"Número de nós inicialmente infectados menor que 1: N * initial_perc = {init_infect}")
            sources = np.random.randint(0, N, init_infect)
            
            # Executa a simulação
            S, I, R = self.run_multiple_simulations(G, sources, time_steps, samples, show_progress=False)
            
            S_list.append(S)
            I_list.append(I)
            R_list.append(R)
        
        # Calcula médias e erros
        S_avg = np.mean(np.array(S_list), axis=0)
        I_avg = np.mean(np.array(I_list), axis=0)
        R_avg = np.mean(np.array(R_list), axis=0)
        
        S_err = np.std(np.array(S_list) / np.sqrt(N), axis=0)
        I_err = np.std(np.array(I_list) / np.sqrt(N), axis=0)
        R_err = np.std(np.array(R_list) / np.sqrt(N), axis=0)
        
        # Salva os resultados
        result = {
            "S_val": list(S_avg),
            "S_err": list(S_err),
            "I_val": list(I_avg),
            "I_err": list(I_err),
            "R_val": list(R_avg),
            "R_err": list(R_err),
            "time": list(time_steps),
            "metadata": {
                "network_type": "ER",
                "distribution": self.distribution.get_distribution_name(),
                "distribution_params": self.distribution.get_params_dict(),
                "N": N,
                "k_avg": k_avg,
                "samples": samples,
                "num_runs": num_runs,
                "initial_perc": initial_perc
            }
        }
        
        result_path = ResultManager.get_result_path("ER", self.distribution, N, samples)
        ResultManager.save_result(result_path, result)
        
        return S_avg, I_avg, R_avg, S_err, I_err, R_err
    
    def simulate_complex_network(self, num_runs: int, exponent: float, time_steps: np.ndarray, 
                               N: int = 3000, k_avg: float = 10, samples: int = 100, 
                               initial_perc: float = 0.01, load_if_exists: bool = True) -> Tuple[np.ndarray, np.ndarray, 
                                                                                              np.ndarray, np.ndarray, 
                                                                                              np.ndarray, np.ndarray]:
        """
        Simula a propagação em múltiplas redes complexas.
        
        Args:
            num_runs: Número de execuções
            exponent: Expoente da lei de potência
            time_steps: Array com os passos de tempo
            N: Número de nós
            k_avg: Grau médio
            samples: Número de amostras por execução
            initial_perc: Porcentagem inicial de infectados
            load_if_exists: Se True, carrega resultados existentes
            
        Returns:
            Tupla com (S_avg, I_avg, R_avg, S_err, I_err, R_err)
        """
        S_list, I_list, R_list = [], [], []
        
        # Verifica se já existem resultados salvos
        if load_if_exists:
            result_path = ResultManager.get_result_path("CN", self.distribution, N, samples, exponent)
            if os.path.exists(result_path):
                try:
                    result = ResultManager.load_result(result_path)
                    return (
                        np.array(result.get('S_val', [])), 
                        np.array(result.get('I_val', [])), 
                        np.array(result.get('R_val', [])),
                        np.array(result.get('S_err', [])), 
                        np.array(result.get('I_err', [])), 
                        np.array(result.get('R_err', []))
                    )
                except Exception as e:
                    print(f"Erro ao carregar resultados existentes: {e}")
        
        # Executa as simulações
        for run in tqdm(range(num_runs), desc=f"Execuções (expoente={exponent})"):
            # Cria a rede
            G = NetworkFactory.create_complex_network(N, exponent, k_avg)
            
            # Configura os nós inicialmente infectados
            init_infect = int(N * initial_perc)
            if init_infect < 1:
                raise ValueError(f"Número de nós inicialmente infectados menor que 1: N * initial_perc = {init_infect}")
            sources = np.random.randint(0, N, init_infect)
            
            # Executa a simulação
            S, I, R = self.run_multiple_simulations(G, sources, time_steps, samples, show_progress=False)
            
            S_list.append(S)
            I_list.append(I)
            R_list.append(R)
        
        # Calcula médias e erros
        S_avg = np.mean(np.array(S_list), axis=0)
        I_avg = np.mean(np.array(I_list), axis=0)
        R_avg = np.mean(np.array(R_list), axis=0)
        
        S_err = np.std(np.array(S_list) / np.sqrt(N), axis=0)
        I_err = np.std(np.array(I_list) / np.sqrt(N), axis=0)
        R_err = np.std(np.array(R_list) / np.sqrt(N), axis=0)
        
        # Salva os resultados
        result = {
            "S_val": list(S_avg),
            "S_err": list(S_err),
            "I_val": list(I_avg),
            "I_err": list(I_err),
            "R_val": list(R_avg),
            "R_err": list(R_err),
            "time": list(time_steps),
            "metadata": {
                "network_type": "CN",
                "distribution": self.distribution.get_distribution_name(),
                "distribution_params": self.distribution.get_params_dict(),
                "exponent": exponent,
                "N": N,
                "k_avg": k_avg,
                "samples": samples,
                "num_runs": num_runs,
                "initial_perc": initial_perc
            }
        }
        
        result_path = ResultManager.get_result_path("CN", self.distribution, N, samples, exponent)
        ResultManager.save_result(result_path, result)
        
        return S_avg, I_avg, R_avg, S_err, I_err, R_err
    
    def simulate_complete_graph(self, time_steps: np.ndarray, N: int = 3000, samples: int = 100, 
                              initial_perc: float = 0.01, overwrite: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simula a propagação em um grafo completo.
        
        Args:
            time_steps: Array com os passos de tempo
            N: Número de nós
            samples: Número de amostras
            initial_perc: Porcentagem inicial de infectados
            overwrite: Se True, sobrescreve resultados existentes
            
        Returns:
            Tupla com (S, I, R) contendo a proporção de indivíduos em cada estado
        """
        # Verifica se já existem resultados salvos
        if not overwrite:
            result_path = ResultManager.get_result_path("CG", self.distribution, N, samples)
            if os.path.exists(result_path):
                try:
                    result = ResultManager.load_result(result_path)
                    return (
                        np.array(result.get('S_val', [])), 
                        np.array(result.get('I_val', [])), 
                        np.array(result.get('R_val', []))
                    )
                except Exception as e:
                    print(f"Erro ao carregar resultados existentes: {e}")
        
        # Cria a rede
        G = NetworkFactory.create_complete_graph(N)
        
        # Configura os nós inicialmente infectados
        init_infect = int(N * initial_perc)
        if init_infect < 1:
            raise ValueError(f"Número de nós inicialmente infectados menor que 1: N * initial_perc = {init_infect}")
        sources = np.random.randint(0, N, init_infect)
        
        # Executa a simulação
        S, I, R = self.run_multiple_simulations(G, sources, time_steps, samples, show_progress=True)
        
        # Salva os resultados
        result = {
            "S_val": list(S),
            "I_val": list(I),
            "R_val": list(R),
            "time": list(time_steps),
            "metadata": {
                "network_type": "CG",
                "distribution": self.distribution.get_distribution_name(),
                "distribution_params": self.distribution.get_params_dict(),
                "N": N,
                "samples": samples,
                "initial_perc": initial_perc
            }
        }
        
        result_path = ResultManager.get_result_path("CG", self.distribution, N, samples)
        ResultManager.save_result(result_path, result)
        
        return S, I, R
    
    def run_simulation(self, network_type: str, time_steps: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Executa uma simulação com base no tipo de rede e parâmetros fornecidos.
        
        Args:
            network_type: Tipo de rede ('er', 'cn', 'cg')
            time_steps: Array com os passos de tempo
            **kwargs: Parâmetros adicionais para a simulação
        
        Returns:
            Dicionário com os resultados da simulação
        
        Raises:
            ValueError: Se o tipo de rede for desconhecido
        """
        network_type = network_type.lower()
        
        # Parâmetros comuns
        N = kwargs.get("N", 1000)
        samples = kwargs.get("samples", 50)
        initial_perc = kwargs.get("initial_perc", 0.01)
        load_if_exists = not kwargs.get("overwrite", False)
        
        if network_type == "er":
            k_avg = kwargs.get("k_avg", 10)
            num_runs = kwargs.get("num_runs", 2)
            
            S, I, R, S_err, I_err, R_err = self.simulate_erdos_renyi(
                num_runs=num_runs,
                time_steps=time_steps,
                N=N,
                k_avg=k_avg,
                samples=samples,
                initial_perc=initial_perc,
                load_if_exists=load_if_exists
            )
            
            return {
                "S_val": S,
                "I_val": I,
                "R_val": R,
                "S_err": S_err,
                "I_err": I_err,
                "R_err": R_err,
                "time": time_steps,
                "has_error": True
            }
            
        elif network_type == "cn":
            k_avg = kwargs.get("k_avg", 10)
            exponent = kwargs.get("exponent", 2.5)
            num_runs = kwargs.get("num_runs", 2)
            
            S, I, R, S_err, I_err, R_err = self.simulate_complex_network(
                num_runs=num_runs,
                exponent=exponent,
                time_steps=time_steps,
                N=N,
                k_avg=k_avg,
                samples=samples,
                initial_perc=initial_perc,
                load_if_exists=load_if_exists
            )
            
            return {
                "S_val": S,
                "I_val": I,
                "R_val": R,
                "S_err": S_err,
                "I_err": I_err,
                "R_err": R_err,
                "time": time_steps,
                "has_error": True
            }
            
        elif network_type == "cg":
            S, I, R = self.simulate_complete_graph(
                time_steps=time_steps,
                N=N,
                samples=samples,
                initial_perc=initial_perc,
                overwrite=not load_if_exists
            )
            
            return {
                "S_val": S,
                "I_val": I,
                "R_val": R,
                "time": time_steps,
                "has_error": False
            }
            
        else:
            raise ValueError(f"Tipo de rede desconhecido: {network_type}")