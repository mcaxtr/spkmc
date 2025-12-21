"""
Implementação principal do algoritmo SPKMC.

Este módulo contém a implementação do algoritmo Shortest Path Kinetic Monte Carlo (SPKMC)
para simulação de propagação de epidemias em redes, utilizando o modelo SIR 
(Susceptible-Infected-Recovered).
"""

import networkx as nx
import numpy as np
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
import os
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

from spkmc.core.distributions import Distribution

# Type alias for progress callback: called with (completed_units, total_units)
ProgressCallback = Optional[Callable[[int, int], None]]
from spkmc.core.networks import NetworkFactory
from spkmc.io.results import ResultManager
from spkmc.utils.numba_utils import calculate


def _create_progress(description: str, total: int, show: bool = True):
    """Create a Rich progress bar context manager."""
    if not show:
        return _DummyProgress()
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        transient=True  # Remove progress bar when done
    )


class _DummyProgress:
    """Dummy progress context manager that does nothing."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def add_task(self, description, total):
        return 0
    def update(self, task_id, advance=1):
        pass


class SPKMC:
    """
    Implementação do algoritmo Shortest Path Kinetic Monte Carlo (SPKMC).

    Esta classe implementa o algoritmo SPKMC para simulação de propagação de epidemias
    em redes, utilizando o modelo SIR (Susceptible-Infected-Recovered).

    Suporta aceleração GPU automática quando disponível.
    """

    def __init__(self, distribution: Distribution, use_gpu: bool = False):
        """
        Inicializa o simulador SPKMC.

        Args:
            distribution: Objeto de distribuição a ser usado na simulação
            use_gpu: Usar aceleração GPU se disponível (padrão: False)
        """
        self.distribution = distribution
        self.use_gpu = use_gpu
        self._gpu_available = None

        # Check GPU availability if requested
        if use_gpu:
            try:
                from spkmc.utils.gpu_utils import is_gpu_available
                self._gpu_available = is_gpu_available()
            except ImportError:
                self._gpu_available = False
    
    def get_dist_sparse(self, N: int, edges: np.ndarray, sources: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula as distâncias mínimas dos nós de origem para todos os outros nós.

        Usa GPU para aceleração quando disponível e habilitado.

        Args:
            N: Número de nós
            edges: Arestas do grafo como matriz (u, v)
            sources: Nós de origem

        Returns:
            Tupla com (distâncias, tempos de recuperação)
        """
        # Try GPU acceleration if enabled and available
        if self.use_gpu and self._gpu_available:
            try:
                from spkmc.utils.gpu_utils import get_dist_gpu
                params = {
                    'distribution': self.distribution.__class__.__name__.lower().replace('distribution', ''),
                    'shape': getattr(self.distribution, 'shape', 2.0),
                    'scale': getattr(self.distribution, 'scale', 1.0),
                    'mu': getattr(self.distribution, 'mu', 1.0),
                    'lambda_val': getattr(self.distribution, 'lambda_val', 1.0),
                }
                return get_dist_gpu(N, edges, sources, params)
            except Exception:
                # Fall back to CPU on any GPU error
                pass

        # CPU implementation (original)
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
                                samples: int, show_progress: bool = True,
                                progress_callback: ProgressCallback = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Executa múltiplas simulações SPKMC e retorna a média.

        Args:
            G: Grafo da rede
            sources: Nós de origem
            time_steps: Array com os passos de tempo
            samples: Número de amostras
            show_progress: Se True, mostra barra de progresso
            progress_callback: Optional callback called after each sample

        Returns:
            Tupla com (S_mean, I_mean, R_mean) contendo a média da proporção de indivíduos em cada estado
        """
        steps = time_steps.shape[0]

        S_values = np.zeros((samples, steps))
        I_values = np.zeros((samples, steps))
        R_values = np.zeros((samples, steps))

        edges = np.array(G.edges())
        N = G.number_of_nodes()

        # Executa as simulações
        with _create_progress("Amostras", samples, show_progress) as progress:
            task = progress.add_task("Amostras", total=samples)
            for sample in range(samples):
                S, I, R = self.run_single_simulation(N, edges, sources, time_steps)
                S_values[sample, :] = S
                I_values[sample, :] = I
                R_values[sample, :] = R
                progress.update(task, advance=1)
                # Call external progress callback if provided
                if progress_callback is not None:
                    progress_callback(1)

        # Calcula as médias
        S_mean = np.mean(S_values, axis=0)
        I_mean = np.mean(I_values, axis=0)
        R_mean = np.mean(R_values, axis=0)

        return S_mean, I_mean, R_mean
    
    def simulate_erdos_renyi(self, num_runs: int, time_steps: np.ndarray, N: int = 3000,
                            k_avg: float = 10, samples: int = 100, initial_perc: float = 0.01,
                            load_if_exists: bool = True, show_progress: bool = True,
                            progress_callback: ProgressCallback = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
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
            show_progress: Se True, mostra barra de progresso
            progress_callback: Optional callback for progress updates

        Returns:
            Tupla com (S_avg, I_avg, R_avg, S_err, I_err, R_err)
        """
        S_list, I_list, R_list = [], [], []

        # Verifica se já existem resultados salvos
        if load_if_exists:
            result_path = ResultManager.get_result_path("ER", self.distribution, N, samples, k_avg=k_avg)
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
        with _create_progress("Execuções", num_runs, show_progress) as progress:
            task = progress.add_task("Execuções (ER)", total=num_runs)
            for run in range(num_runs):
                # Cria a rede
                G = NetworkFactory.create_erdos_renyi(N, k_avg)

                # Configura os nós inicialmente infectados
                init_infect = int(N * initial_perc)
                if init_infect < 1:
                    raise ValueError(f"Número de nós inicialmente infectados menor que 1: N * initial_perc = {init_infect}")
                sources = np.random.randint(0, N, init_infect)

                # Executa a simulação
                S, I, R = self.run_multiple_simulations(G, sources, time_steps, samples,
                                                        show_progress=False, progress_callback=progress_callback)

                S_list.append(S)
                I_list.append(I)
                R_list.append(R)
                progress.update(task, advance=1)
        
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
        
        result_path = ResultManager.get_result_path("ER", self.distribution, N, samples, k_avg=k_avg)
        ResultManager.save_result(result_path, result)
        
        return S_avg, I_avg, R_avg, S_err, I_err, R_err
    
    def simulate_complex_network(self, num_runs: int, exponent: float, time_steps: np.ndarray,
                               N: int = 3000, k_avg: float = 10, samples: int = 100,
                               initial_perc: float = 0.01, load_if_exists: bool = True,
                               show_progress: bool = True,
                               progress_callback: ProgressCallback = None) -> Tuple[np.ndarray, np.ndarray,
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
            show_progress: Se True, mostra barra de progresso
            progress_callback: Optional callback for progress updates

        Returns:
            Tupla com (S_avg, I_avg, R_avg, S_err, I_err, R_err)
        """
        S_list, I_list, R_list = [], [], []

        # Verifica se já existem resultados salvos
        if load_if_exists:
            result_path = ResultManager.get_result_path("CN", self.distribution, N, samples, exponent=exponent, k_avg=k_avg)
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
        with _create_progress("Execuções", num_runs, show_progress) as progress:
            task = progress.add_task(f"Execuções (CN γ={exponent})", total=num_runs)
            for run in range(num_runs):
                # Cria a rede
                G = NetworkFactory.create_complex_network(N, exponent, k_avg)

                # Configura os nós inicialmente infectados
                init_infect = int(N * initial_perc)
                if init_infect < 1:
                    raise ValueError(f"Número de nós inicialmente infectados menor que 1: N * initial_perc = {init_infect}")
                sources = np.random.randint(0, N, init_infect)

                # Executa a simulação
                S, I, R = self.run_multiple_simulations(G, sources, time_steps, samples,
                                                        show_progress=False, progress_callback=progress_callback)

                S_list.append(S)
                I_list.append(I)
                R_list.append(R)
                progress.update(task, advance=1)

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
        
        result_path = ResultManager.get_result_path("CN", self.distribution, N, samples, exponent=exponent, k_avg=k_avg)
        ResultManager.save_result(result_path, result)

        return S_avg, I_avg, R_avg, S_err, I_err, R_err

    def simulate_complete_graph(self, time_steps: np.ndarray, N: int = 3000, samples: int = 100,
                              initial_perc: float = 0.01, overwrite: bool = False,
                              progress_callback: ProgressCallback = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simula a propagação em um grafo completo.

        Args:
            time_steps: Array com os passos de tempo
            N: Número de nós
            samples: Número de amostras
            initial_perc: Porcentagem inicial de infectados
            overwrite: Se True, sobrescreve resultados existentes
            progress_callback: Optional callback for progress updates

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
        S, I, R = self.run_multiple_simulations(G, sources, time_steps, samples,
                                                show_progress=True, progress_callback=progress_callback)
        
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
    
    def run_simulation(self, network_type: str, time_steps: np.ndarray,
                       progress_callback: ProgressCallback = None, **kwargs) -> Dict[str, Any]:
        """
        Executa uma simulação com base no tipo de rede e parâmetros fornecidos.

        Args:
            network_type: Tipo de rede ('er', 'cn', 'cg', 'rrn')
            time_steps: Array com os passos de tempo
            progress_callback: Optional callback for granular progress updates
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
        show_progress = kwargs.get("show_progress", True)

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
                load_if_exists=load_if_exists,
                show_progress=show_progress,
                progress_callback=progress_callback
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
                load_if_exists=load_if_exists,
                show_progress=show_progress,
                progress_callback=progress_callback
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
                overwrite=not load_if_exists,
                progress_callback=progress_callback
            )
            
            return {
                "S_val": S,
                "I_val": I,
                "R_val": R,
                "time": time_steps,
                "has_error": False
            }
            
        elif network_type == "rrn":
            k_avg = kwargs.get("k_avg", 10)
            num_runs = kwargs.get("num_runs", 2)

            S, I, R, S_err, I_err, R_err = self.simulate_random_regular_network(
                num_runs=num_runs,
                time_steps=time_steps,
                N=N,
                k_avg=k_avg,
                samples=samples,
                initial_perc=initial_perc,
                load_if_exists=load_if_exists,
                show_progress=show_progress,
                progress_callback=progress_callback
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
            
        else:
            raise ValueError(f"Tipo de rede desconhecido: {network_type}")
            
    def simulate_random_regular_network(self, num_runs: int, time_steps: np.ndarray, N: int = 3000,
                                      k_avg: int = 10, samples: int = 100, initial_perc: float = 0.01,
                                      load_if_exists: bool = True, show_progress: bool = True,
                                      progress_callback: ProgressCallback = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                          np.ndarray, np.ndarray, np.ndarray]:
        """
        Simula a propagação em múltiplas redes regulares aleatórias.

        Args:
            num_runs: Número de execuções
            time_steps: Array com os passos de tempo
            N: Número de nós
            k_avg: Grau regular (número de conexões por nó)
            samples: Número de amostras por execução
            initial_perc: Porcentagem inicial de infectados
            load_if_exists: Se True, carrega resultados existentes
            show_progress: Se True, mostra barra de progresso
            progress_callback: Optional callback for progress updates

        Returns:
            Tupla com (S_avg, I_avg, R_avg, S_err, I_err, R_err)
        """
        S_list, I_list, R_list = [], [], []
        
        # Verifica se já existem resultados salvos
        if load_if_exists:
            result_path = ResultManager.get_result_path("RRN", self.distribution, N, samples, k_avg=k_avg)
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
        with _create_progress("Execuções", num_runs, show_progress) as progress:
            task = progress.add_task("Execuções (RRN)", total=num_runs)
            for run in range(num_runs):
                # Cria a rede
                G = NetworkFactory.create_random_regular_network(N, k_avg)

                # Configura os nós inicialmente infectados
                init_infect = int(N * initial_perc)
                if init_infect < 1:
                    raise ValueError(f"Número de nós inicialmente infectados menor que 1: N * initial_perc = {init_infect}")
                sources = np.random.randint(0, N, init_infect)

                # Executa a simulação
                S, I, R = self.run_multiple_simulations(G, sources, time_steps, samples,
                                                        show_progress=False, progress_callback=progress_callback)

                S_list.append(S)
                I_list.append(I)
                R_list.append(R)
                progress.update(task, advance=1)

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
                "network_type": "RRN",
                "distribution": self.distribution.get_distribution_name(),
                "distribution_params": self.distribution.get_params_dict(),
                "N": N,
                "k_avg": k_avg,
                "samples": samples,
                "num_runs": num_runs,
                "initial_perc": initial_perc
            }
        }

        result_path = ResultManager.get_result_path("RRN", self.distribution, N, samples, k_avg=k_avg)
        ResultManager.save_result(result_path, result)

        return S_avg, I_avg, R_avg, S_err, I_err, R_err