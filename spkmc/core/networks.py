"""
Classes de redes para o algoritmo SPKMC.

Este módulo contém implementações de diferentes tipos de redes que podem ser
utilizadas nas simulações SPKMC, como redes Erdos-Renyi, redes complexas e grafos completos.
"""

import networkx as nx
import numpy as np
from typing import Optional, Dict, Any


class NetworkFactory:
    """Fábrica para criar diferentes tipos de redes."""
    
    NETWORK_TYPES = ["er", "cn", "cg"]  # Tipos de rede suportados
    
    @staticmethod
    def create_network(network_type: str, **kwargs) -> nx.DiGraph:
        """
        Cria uma rede com base no tipo e parâmetros fornecidos.
        
        Args:
            network_type: Tipo de rede ('er', 'cn', 'cg')
            **kwargs: Parâmetros específicos do tipo de rede
        
        Returns:
            Grafo direcionado da rede solicitada
        
        Raises:
            ValueError: Se o tipo de rede for desconhecido
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
        
        else:
            raise ValueError(f"Tipo de rede desconhecido: {network_type}")
    
    @staticmethod
    def create_erdos_renyi(N: int, k_avg: float) -> nx.DiGraph:
        """
        Cria uma rede Erdos-Renyi.
        
        Args:
            N: Número de nós
            k_avg: Grau médio
            
        Returns:
            Grafo direcionado Erdos-Renyi
        """
        p = k_avg / (N - 1)
        return nx.erdos_renyi_graph(N, p, directed=True)
    
    @staticmethod
    def generate_discrete_power_law(n: int, alpha: float, xmin: int, xmax: float) -> np.ndarray:
        """
        Gera uma sequência de lei de potência discreta.
        
        Args:
            n: Número de elementos
            alpha: Expoente da lei de potência
            xmin: Valor mínimo
            xmax: Valor máximo
            
        Returns:
            Array com a sequência de lei de potência
        """
        rand_nums = np.random.uniform(size=n)
        power_law_seq = (xmax**(1-alpha) - xmin**(1-alpha)) * rand_nums + xmin**(1-alpha)
        power_law_seq = power_law_seq**(1/(1-alpha))
        power_law_seq = power_law_seq.astype(int)

        if sum(power_law_seq) % 2 == 0:
            return power_law_seq
        else:
            power_law_seq[0] = power_law_seq[0] + 1
            return power_law_seq
    
    @staticmethod
    def create_complex_network(N: int, exponent: float, k_avg: float) -> nx.DiGraph:
        """
        Cria uma rede complexa com distribuição de grau seguindo lei de potência.
        
        Args:
            N: Número de nós
            exponent: Expoente da lei de potência
            k_avg: Grau médio
            
        Returns:
            Grafo direcionado complexo
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
        Cria um grafo completo.
        
        Args:
            N: Número de nós
            
        Returns:
            Grafo direcionado completo
        """
        return nx.complete_graph(N, create_using=nx.DiGraph())
    
    @staticmethod
    def get_network_info(network_type: str, **kwargs) -> Dict[str, Any]:
        """
        Retorna informações sobre a rede para uso em metadados.
        
        Args:
            network_type: Tipo de rede ('er', 'cn', 'cg')
            **kwargs: Parâmetros específicos do tipo de rede
        
        Returns:
            Dicionário com informações sobre a rede
        """
        network_type = network_type.lower()
        
        info = {
            "type": network_type,
            "N": kwargs.get("N", 1000)
        }
        
        if network_type in ["er", "cn"]:
            info["k_avg"] = kwargs.get("k_avg", 10)
            
        if network_type == "cn":
            info["exponent"] = kwargs.get("exponent", 2.5)
            
        return info