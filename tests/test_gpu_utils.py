"""
Testes para as funções utilitárias de GPU no SPKMC.

Este módulo contém testes para verificar a disponibilidade de GPU e as funções utilitárias
relacionadas à aceleração GPU no algoritmo SPKMC.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from spkmc.utils.gpu_utils import (
    check_gpu_dependencies, 
    is_gpu_available, 
    get_dist_gpu, 
    get_states_gpu, 
    calculate_gpu
)


def test_check_gpu_dependencies():
    """
    Testa a função check_gpu_dependencies.
    
    Este teste verifica se a função check_gpu_dependencies retorna um valor booleano.
    O resultado depende do ambiente de execução, então não verificamos o valor específico.
    """
    result = check_gpu_dependencies()
    assert isinstance(result, bool)


def test_is_gpu_available():
    """
    Testa a função is_gpu_available.
    
    Este teste verifica se a função is_gpu_available retorna um valor booleano.
    O resultado depende do ambiente de execução, então não verificamos o valor específico.
    """
    result = is_gpu_available()
    assert isinstance(result, bool)


@pytest.mark.skipif(not is_gpu_available(), reason="GPU não disponível")
def test_get_dist_gpu():
    """
    Testa a função get_dist_gpu.
    
    Este teste verifica se a função get_dist_gpu calcula corretamente as distâncias
    usando GPU. O teste é pulado se a GPU não estiver disponível.
    """
    # Configuração do teste
    N = 5
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    sources = np.array([0])
    params = {
        'distribution': 'gamma',
        'shape': 2.0,
        'scale': 1.0,
        'mu': 1.0,
        'lambda': 1.0
    }
    
    # Executa a função
    dist, recovery_times = get_dist_gpu(N, edges, sources, params)
    
    # Verifica os resultados
    assert isinstance(dist, np.ndarray)
    assert isinstance(recovery_times, np.ndarray)
    assert dist.shape == (N,)
    assert recovery_times.shape == (N,)
    assert dist[0] == 0  # A distância do nó 0 para ele mesmo é 0
    assert np.all(dist[1:] >= 0)  # As distâncias para os outros nós são não-negativas


@pytest.mark.skipif(not is_gpu_available(), reason="GPU não disponível")
def test_get_states_gpu():
    """
    Testa a função get_states_gpu.
    
    Este teste verifica se a função get_states_gpu calcula corretamente os estados
    (S, I, R) usando GPU. O teste é pulado se a GPU não estiver disponível.
    """
    # Configuração do teste
    N = 5
    time_to_infect = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    recovery_times = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    t = 1.5
    
    # Importa cupy apenas se disponível
    if is_gpu_available():
        import cupy as cp
        # Executa a função
        S, I, R = get_states_gpu(cp.asarray(time_to_infect), cp.asarray(recovery_times), t)
        
        # Converte para numpy para verificação
        S_np = S.get()
        I_np = I.get()
        R_np = R.get()
        
        # Verifica os resultados
        assert isinstance(S_np, np.ndarray)
        assert isinstance(I_np, np.ndarray)
        assert isinstance(R_np, np.ndarray)
        assert S_np.shape == (N,)
        assert I_np.shape == (N,)
        assert R_np.shape == (N,)
        
        # Verifica a lógica dos estados
        # S: time_to_infect > t
        # I: (~S) & (time_to_infect + recovery_times > t)
        # R: (~S) & (~I)
        expected_S = time_to_infect > t
        expected_I = (~expected_S) & (time_to_infect + recovery_times > t)
        expected_R = (~expected_S) & (~expected_I)
        
        assert np.array_equal(S_np, expected_S)
        assert np.array_equal(I_np, expected_I)
        assert np.array_equal(R_np, expected_R)


@pytest.mark.skipif(not is_gpu_available(), reason="GPU não disponível")
def test_calculate_gpu():
    """
    Testa a função calculate_gpu.
    
    Este teste verifica se a função calculate_gpu calcula corretamente a proporção
    de indivíduos em cada estado (S, I, R) para cada passo de tempo usando GPU.
    O teste é pulado se a GPU não estiver disponível.
    """
    # Configuração do teste
    N = 5
    time_to_infect = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    recovery_times = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    time_steps = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    steps = len(time_steps)
    
    # Executa a função
    S, I, R = calculate_gpu(N, time_to_infect, recovery_times, time_steps, steps)
    
    # Verifica os resultados
    assert isinstance(S, np.ndarray)
    assert isinstance(I, np.ndarray)
    assert isinstance(R, np.ndarray)
    assert S.shape == (steps,)
    assert I.shape == (steps,)
    assert R.shape == (steps,)
    assert np.all(S >= 0) and np.all(S <= 1)
    assert np.all(I >= 0) and np.all(I <= 1)
    assert np.all(R >= 0) and np.all(R <= 1)
    assert np.allclose(S + I + R, 1.0)  # A soma deve ser 1


@pytest.mark.skipif(not is_gpu_available(), reason="GPU não disponível")
def test_get_dist_gpu_exponential():
    """
    Testa a função get_dist_gpu com distribuição exponencial.
    
    Este teste verifica se a função get_dist_gpu calcula corretamente as distâncias
    usando GPU com distribuição exponencial. O teste é pulado se a GPU não estiver disponível.
    """
    # Configuração do teste
    N = 5
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    sources = np.array([0])
    params = {
        'distribution': 'exponential',
        'shape': 2.0,
        'scale': 1.0,
        'mu': 1.0,
        'lambda': 1.0
    }
    
    # Executa a função
    dist, recovery_times = get_dist_gpu(N, edges, sources, params)
    
    # Verifica os resultados
    assert isinstance(dist, np.ndarray)
    assert isinstance(recovery_times, np.ndarray)
    assert dist.shape == (N,)
    assert recovery_times.shape == (N,)
    assert dist[0] == 0  # A distância do nó 0 para ele mesmo é 0
    assert np.all(dist[1:] >= 0)  # As distâncias para os outros nós são não-negativas


def test_gpu_dependencies_not_available():
    """
    Testa o comportamento quando as dependências GPU não estão disponíveis.
    
    Este teste simula a ausência das dependências GPU e verifica se as funções
    levantam a exceção ImportError apropriada.
    """
    with patch('spkmc.utils.gpu_utils.check_gpu_dependencies', return_value=False):
        # Verifica se is_gpu_available retorna False
        assert not is_gpu_available()
        
        # Verifica se as funções levantam ImportError
        with pytest.raises(ImportError):
            get_dist_gpu(5, np.array([[0, 1]]), np.array([0]), {'distribution': 'gamma'})
        
        with pytest.raises(ImportError):
            get_states_gpu(np.array([0.0]), np.array([1.0]), 0.5)
        
        with pytest.raises(ImportError):
            calculate_gpu(5, np.array([0.0]), np.array([1.0]), np.array([0.0]), 1)