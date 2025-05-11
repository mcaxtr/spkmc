"""
Classes de distribuição para o algoritmo SPKMC.

Este módulo contém as implementações de diferentes distribuições de probabilidade
utilizadas no algoritmo SPKMC para modelar os tempos de recuperação e infecção.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

from spkmc.utils.numba_utils import (
    gamma_sampling,
    get_weight_exponential,
    compute_infection_times_gamma,
    compute_infection_times_exponential
)


class Distribution(ABC):
    """Classe abstrata para distribuições de probabilidade usadas no SPKMC."""
    
    @abstractmethod
    def get_recovery_weights(self, size: int) -> np.ndarray:
        """
        Gera os pesos de recuperação para cada nó.
        
        Args:
            size: Número de nós
            
        Returns:
            Array com os pesos de recuperação
        """
        pass
    
    @abstractmethod
    def get_infection_times(self, recovery_times: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """
        Calcula os tempos de infecção para cada aresta.
        
        Args:
            recovery_times: Tempos de recuperação para cada nó
            edges: Arestas do grafo como matriz (u, v)
            
        Returns:
            Array com os tempos de infecção
        """
        pass
    
    @abstractmethod
    def get_distribution_name(self) -> str:
        """
        Retorna o nome da distribuição.
        
        Returns:
            Nome da distribuição
        """
        pass
    
    @abstractmethod
    def get_params_string(self) -> str:
        """
        Retorna uma string com os parâmetros da distribuição para uso em nomes de arquivos.
        
        Returns:
            String com os parâmetros
        """
        pass
    
    def get_params_dict(self) -> dict:
        """
        Retorna um dicionário com os parâmetros da distribuição.
        
        Returns:
            Dicionário com os parâmetros
        """
        return {}


class GammaDistribution(Distribution):
    """Implementação da distribuição Gamma para o SPKMC."""
    
    def __init__(self, shape: float, scale: float, lmbd: float = 1.0):
        """
        Inicializa a distribuição Gamma.
        
        Args:
            shape: Parâmetro de forma da distribuição Gamma
            scale: Parâmetro de escala da distribuição Gamma
            lmbd: Parâmetro lambda para tempos de infecção (padrão: 1.0)
        """
        self.shape = shape
        self.scale = scale
        self.lmbd = lmbd
    
    def get_recovery_weights(self, size: int) -> np.ndarray:
        """
        Gera os pesos de recuperação usando a distribuição Gamma.
        
        Args:
            size: Número de nós
            
        Returns:
            Array com os pesos de recuperação
        """
        return gamma_sampling(self.shape, self.scale, size)
    
    def get_infection_times(self, recovery_times: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """
        Calcula os tempos de infecção usando a distribuição Gamma.
        
        Args:
            recovery_times: Tempos de recuperação para cada nó
            edges: Arestas do grafo como matriz (u, v)
            
        Returns:
            Array com os tempos de infecção
        """
        # Nota: Atualmente usando exponencial para infecção, mesmo com recuperação gamma
        return compute_infection_times_exponential(self.lmbd, recovery_times, edges)
    
    def get_distribution_name(self) -> str:
        """
        Retorna o nome da distribuição.
        
        Returns:
            Nome da distribuição
        """
        return "gamma"
    
    def get_params_string(self) -> str:
        """
        Retorna uma string com os parâmetros da distribuição para uso em nomes de arquivos.
        
        Returns:
            String com os parâmetros
        """
        return f"{self.shape}"
    
    def get_params_dict(self) -> dict:
        """
        Retorna um dicionário com os parâmetros da distribuição.
        
        Returns:
            Dicionário com os parâmetros
        """
        return {
            "type": "gamma",
            "shape": self.shape,
            "scale": self.scale,
            "lambda": self.lmbd
        }


class ExponentialDistribution(Distribution):
    """Implementação da distribuição Exponencial para o SPKMC."""
    
    def __init__(self, mu: float, lmbd: float):
        """
        Inicializa a distribuição Exponencial.
        
        Args:
            mu: Parâmetro mu para tempos de recuperação
            lmbd: Parâmetro lambda para tempos de infecção
        """
        self.mu = mu
        self.lmbd = lmbd
    
    def get_recovery_weights(self, size: int) -> np.ndarray:
        """
        Gera os pesos de recuperação usando a distribuição Exponencial.
        
        Args:
            size: Número de nós
            
        Returns:
            Array com os pesos de recuperação
        """
        return get_weight_exponential(self.mu, size)
    
    def get_infection_times(self, recovery_times: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """
        Calcula os tempos de infecção usando a distribuição Exponencial.
        
        Args:
            recovery_times: Tempos de recuperação para cada nó
            edges: Arestas do grafo como matriz (u, v)
            
        Returns:
            Array com os tempos de infecção
        """
        return compute_infection_times_exponential(self.lmbd, recovery_times, edges)
    
    def get_distribution_name(self) -> str:
        """
        Retorna o nome da distribuição.
        
        Returns:
            Nome da distribuição
        """
        return "exponential"
    
    def get_params_string(self) -> str:
        """
        Retorna uma string com os parâmetros da distribuição para uso em nomes de arquivos.
        
        Returns:
            String com os parâmetros
        """
        return ""
    
    def get_params_dict(self) -> dict:
        """
        Retorna um dicionário com os parâmetros da distribuição.
        
        Returns:
            Dicionário com os parâmetros
        """
        return {
            "type": "exponential",
            "mu": self.mu,
            "lambda": self.lmbd
        }


def create_distribution(dist_type: str, **kwargs) -> Distribution:
    """
    Cria uma instância de distribuição com base no tipo e parâmetros fornecidos.
    
    Args:
        dist_type: Tipo de distribuição ('gamma' ou 'exponential')
        **kwargs: Parâmetros específicos da distribuição
    
    Returns:
        Instância da distribuição solicitada
    
    Raises:
        ValueError: Se o tipo de distribuição for desconhecido
    """
    if dist_type.lower() == "gamma":
        shape = kwargs.get("shape", 2.0)
        scale = kwargs.get("scale", 1.0)
        lmbd = kwargs.get("lambda", 1.0)
        return GammaDistribution(shape=shape, scale=scale, lmbd=lmbd)
    
    elif dist_type.lower() == "exponential":
        mu = kwargs.get("mu", 1.0)
        lmbd = kwargs.get("lambda", 1.0)
        return ExponentialDistribution(mu=mu, lmbd=lmbd)
    
    else:
        raise ValueError(f"Tipo de distribuição desconhecido: {dist_type}")