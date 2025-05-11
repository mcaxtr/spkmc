"""
Classes de distribuição para o algoritmo SPKMC.

Este módulo contém as implementações de diferentes distribuições de probabilidade
utilizadas no algoritmo SPKMC para modelar os tempos de recuperação e infecção.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union

from spkmc.utils.numba_utils import (
    gamma_sampling,
    get_weight_exponential,
    compute_infection_times_gamma,
    compute_infection_times_exponential
)

# Importação condicional das funções GPU
try:
    from spkmc.utils.gpu_utils import (
        is_gpu_available,
        gamma_sampling_gpu,
        get_weight_exponential_gpu,
        compute_infection_times_gamma_gpu,
        compute_infection_times_exponential_gpu,
        to_numpy,
        to_cupy
    )
    GPU_AVAILABLE = is_gpu_available()
except ImportError:
    GPU_AVAILABLE = False


class Distribution(ABC):
    """Classe abstrata para distribuições de probabilidade usadas no SPKMC."""
    
    def __init__(self, use_gpu: bool = False):
        """
        Inicializa a distribuição.
        
        Args:
            use_gpu: Se True, usa GPU para aceleração (se disponível)
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
    
    @abstractmethod
    def get_recovery_weights(self, size: int) -> Union[np.ndarray, "cp.ndarray"]:
        """
        Gera os pesos de recuperação para cada nó.
        
        Args:
            size: Número de nós
            
        Returns:
            Array com os pesos de recuperação (NumPy ou CuPy)
        """
        pass
    
    @abstractmethod
    def get_infection_times(self, recovery_times: Union[np.ndarray, "cp.ndarray"],
                           edges: np.ndarray) -> Union[np.ndarray, "cp.ndarray"]:
        """
        Calcula os tempos de infecção para cada aresta.
        
        Args:
            recovery_times: Tempos de recuperação para cada nó (NumPy ou CuPy)
            edges: Arestas do grafo como matriz (u, v)
            
        Returns:
            Array com os tempos de infecção (NumPy ou CuPy)
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
    
    def __init__(self, shape: float, scale: float, lmbd: float = 1.0, use_gpu: bool = False):
        """
        Inicializa a distribuição Gamma.
        
        Args:
            shape: Parâmetro de forma da distribuição Gamma
            scale: Parâmetro de escala da distribuição Gamma
            lmbd: Parâmetro lambda para tempos de infecção (padrão: 1.0)
            use_gpu: Se True, usa GPU para aceleração (se disponível)
        """
        super().__init__(use_gpu)
        self.shape = shape
        self.scale = scale
        self.lmbd = lmbd
    
    def get_recovery_weights(self, size: int) -> Union[np.ndarray, "cp.ndarray"]:
        """
        Gera os pesos de recuperação usando a distribuição Gamma.
        
        Args:
            size: Número de nós
            
        Returns:
            Array com os pesos de recuperação (NumPy ou CuPy)
        """
        if self.use_gpu:
            return gamma_sampling_gpu(self.shape, self.scale, size)
        else:
            return gamma_sampling(self.shape, self.scale, size)
    
    @timing_decorator
    def get_infection_times(self, recovery_times: Union[np.ndarray, "cp.ndarray"],
                           edges: np.ndarray) -> Union[np.ndarray, "cp.ndarray"]:
        """
        Calcula os tempos de infecção usando a distribuição Gamma.
        Versão otimizada que mantém os dados na GPU durante todo o processamento quando use_gpu=True.
        
        Args:
            recovery_times: Tempos de recuperação para cada nó (NumPy ou CuPy)
            edges: Arestas do grafo como matriz (u, v)
            
        Returns:
            Array com os tempos de infecção (NumPy ou CuPy)
        """
        # Nota: Atualmente usando exponencial para infecção, mesmo com recuperação gamma
        if self.use_gpu:
            try:
                # Verificar se estamos usando CuPy
                is_cupy = hasattr(recovery_times, "device")
                
                # Converter edges para CuPy se necessário
                if not is_cupy:
                    # Se recovery_times não é CuPy, converter ambos para CuPy
                    recovery_times_gpu = to_cupy(recovery_times)
                    edges_gpu = to_cupy(edges)
                else:
                    # Se recovery_times já é CuPy, apenas converter edges se necessário
                    recovery_times_gpu = recovery_times
                    edges_gpu = to_cupy(edges) if not isinstance(edges, type(recovery_times)) else edges
                
                # Usar a versão GPU otimizada
                result = compute_infection_times_exponential_gpu(self.lmbd, recovery_times_gpu, edges_gpu)
                
                # Não converter de volta para CPU, manter na GPU para processamento posterior
                return result
            except Exception as e:
                print(f"Erro ao calcular tempos de infecção na GPU: {e}")
                print("Revertendo para CPU...")
                # Converter para NumPy se necessário
                if not isinstance(recovery_times, np.ndarray):
                    recovery_times = to_numpy(recovery_times)
                # Usar a versão CPU
                return compute_infection_times_exponential(self.lmbd, recovery_times, edges)
        else:
            # Usar a versão CPU
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
    
    def __init__(self, mu: float, lmbd: float, use_gpu: bool = False):
        """
        Inicializa a distribuição Exponencial.
        
        Args:
            mu: Parâmetro mu para tempos de recuperação
            lmbd: Parâmetro lambda para tempos de infecção
            use_gpu: Se True, usa GPU para aceleração (se disponível)
        """
        super().__init__(use_gpu)
        self.mu = mu
        self.lmbd = lmbd
    
    def get_recovery_weights(self, size: int) -> Union[np.ndarray, "cp.ndarray"]:
        """
        Gera os pesos de recuperação usando a distribuição Exponencial.
        
        Args:
            size: Número de nós
            
        Returns:
            Array com os pesos de recuperação (NumPy ou CuPy)
        """
        if self.use_gpu:
            return get_weight_exponential_gpu(self.mu, size)
        else:
            return get_weight_exponential(self.mu, size)
    
    @timing_decorator
    def get_infection_times(self, recovery_times: Union[np.ndarray, "cp.ndarray"],
                           edges: np.ndarray) -> Union[np.ndarray, "cp.ndarray"]:
        """
        Calcula os tempos de infecção usando a distribuição Exponencial.
        Versão otimizada que mantém os dados na GPU durante todo o processamento quando use_gpu=True.
        
        Args:
            recovery_times: Tempos de recuperação para cada nó (NumPy ou CuPy)
            edges: Arestas do grafo como matriz (u, v)
            
        Returns:
            Array com os tempos de infecção (NumPy ou CuPy)
        """
        if self.use_gpu:
            try:
                # Verificar se estamos usando CuPy
                is_cupy = hasattr(recovery_times, "device")
                
                # Converter edges para CuPy se necessário
                if not is_cupy:
                    # Se recovery_times não é CuPy, converter ambos para CuPy
                    recovery_times_gpu = to_cupy(recovery_times)
                    edges_gpu = to_cupy(edges)
                else:
                    # Se recovery_times já é CuPy, apenas converter edges se necessário
                    recovery_times_gpu = recovery_times
                    edges_gpu = to_cupy(edges) if not isinstance(edges, type(recovery_times)) else edges
                
                # Usar a versão GPU otimizada
                result = compute_infection_times_exponential_gpu(self.lmbd, recovery_times_gpu, edges_gpu)
                
                # Não converter de volta para CPU, manter na GPU para processamento posterior
                return result
            except Exception as e:
                print(f"Erro ao calcular tempos de infecção na GPU: {e}")
                print("Revertendo para CPU...")
                # Converter para NumPy se necessário
                if not isinstance(recovery_times, np.ndarray):
                    recovery_times = to_numpy(recovery_times)
                # Usar a versão CPU
                return compute_infection_times_exponential(self.lmbd, recovery_times, edges)
        else:
            # Usar a versão CPU
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


def create_distribution(dist_type: str, use_gpu: bool = False, **kwargs) -> Distribution:
    """
    Cria uma instância de distribuição com base no tipo e parâmetros fornecidos.
    
    Args:
        dist_type: Tipo de distribuição ('gamma' ou 'exponential')
        use_gpu: Se True, usa GPU para aceleração (se disponível)
        **kwargs: Parâmetros específicos da distribuição
    
    Returns:
        Instância da distribuição solicitada
    
    Raises:
        ValueError: Se o tipo de distribuição for desconhecido
    """
    if use_gpu and not GPU_AVAILABLE:
        print("GPU solicitada, mas não disponível. Usando CPU.")
        use_gpu = False
    
    # Adicionar informação sobre o modo de execução aos parâmetros
    execution_mode = "GPU" if use_gpu and GPU_AVAILABLE else "CPU"
    print(f"Criando distribuição {dist_type} no modo {execution_mode}")
        
    if dist_type.lower() == "gamma":
        shape = kwargs.get("shape", 2.0)
        scale = kwargs.get("scale", 1.0)
        lmbd = kwargs.get("lambda", 1.0)
        distribution = GammaDistribution(shape=shape, scale=scale, lmbd=lmbd, use_gpu=use_gpu)
        
        # Adicionar informação sobre o modo de execução ao dicionário de parâmetros
        original_get_params_dict = distribution.get_params_dict
        def enhanced_get_params_dict():
            params = original_get_params_dict()
            params["execution_mode"] = execution_mode
            return params
        distribution.get_params_dict = enhanced_get_params_dict
        
        return distribution
    
    elif dist_type.lower() == "exponential":
        mu = kwargs.get("mu", 1.0)
        lmbd = kwargs.get("lambda", 1.0)
        distribution = ExponentialDistribution(mu=mu, lmbd=lmbd, use_gpu=use_gpu)
        
        # Adicionar informação sobre o modo de execução ao dicionário de parâmetros
        original_get_params_dict = distribution.get_params_dict
        def enhanced_get_params_dict():
            params = original_get_params_dict()
            params["execution_mode"] = execution_mode
            return params
        distribution.get_params_dict = enhanced_get_params_dict
        
        return distribution
    
    else:
        raise ValueError(f"Tipo de distribuição desconhecido: {dist_type}")