"""
Gerenciamento de resultados para o algoritmo SPKMC.

Este módulo contém funções e classes para o gerenciamento de resultados de simulações,
incluindo salvamento, carregamento e manipulação de dados.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np

from spkmc.core.distributions import Distribution


class ResultManager:
    """Gerencia o salvamento e carregamento de resultados de simulações."""
    
    # Diretório base para resultados
    BASE_DIR = "data/spkmc"
    
    @staticmethod
    def get_result_path(network_type: str, distribution: Distribution, N: int, samples: int, 
                       exponent: Optional[float] = None) -> str:
        """
        Gera o caminho para o arquivo de resultados.
        
        Args:
            network_type: Tipo de rede (ER, CN, CG)
            distribution: Objeto de distribuição
            N: Número de nós
            samples: Número de amostras
            exponent: Expoente para redes complexas
            
        Returns:
            Caminho para o arquivo de resultados
        """
        base_path = f"{ResultManager.BASE_DIR}/{distribution.get_distribution_name()}/{network_type}/"
        
        # Cria diretórios se não existirem
        Path(base_path).mkdir(parents=True, exist_ok=True)
        
        if network_type == "CN" and exponent is not None:
            return f"{base_path}results_{str(exponent).replace('.', '')}_{N}_{samples}_{distribution.get_params_string()}.json"
        else:
            return f"{base_path}results_{N}_{samples}_{distribution.get_params_string()}.json"
    
    @staticmethod
    def load_result(file_path: str) -> Dict[str, Any]:
        """
        Carrega resultados de um arquivo JSON.
        
        Args:
            file_path: Caminho para o arquivo de resultados
            
        Returns:
            Dicionário com os resultados
        
        Raises:
            FileNotFoundError: Se o arquivo não existir
            json.JSONDecodeError: Se o arquivo não for um JSON válido
        """
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise e
    
    @staticmethod
    def save_result(file_path: str, result: Dict[str, Any]) -> None:
        """
        Salva resultados em um arquivo JSON.
        
        Args:
            file_path: Caminho para o arquivo de resultados
            result: Dicionário com os resultados
        
        Raises:
            IOError: Se ocorrer um erro ao salvar o arquivo
        """
        # Cria diretório se não existir
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        try:
            with open(file_path, 'w') as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            raise IOError(f"Erro ao salvar resultados: {e}")
    
    @staticmethod
    def list_results(filter_by: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Lista os arquivos de resultados disponíveis.
        
        Args:
            filter_by: Dicionário com critérios de filtragem (opcional)
            
        Returns:
            Lista de caminhos para arquivos de resultados
        """
        results = []
        base_path = Path(ResultManager.BASE_DIR)
        
        if not base_path.exists():
            return results
        
        for dist_dir in base_path.iterdir():
            if dist_dir.is_dir():
                for network_dir in dist_dir.iterdir():
                    if network_dir.is_dir():
                        for result_file in network_dir.glob("*.json"):
                            # Se não houver filtro, adiciona todos os resultados
                            if filter_by is None:
                                results.append(str(result_file))
                            else:
                                # Verifica se o arquivo atende aos critérios de filtragem
                                try:
                                    result_data = ResultManager.load_result(str(result_file))
                                    
                                    # Verifica se todos os critérios de filtragem são atendidos
                                    match = True
                                    for key, value in filter_by.items():
                                        if key not in result_data or result_data[key] != value:
                                            match = False
                                            break
                                    
                                    if match:
                                        results.append(str(result_file))
                                except:
                                    # Ignora arquivos que não podem ser carregados
                                    pass
        
        return results
    
    @staticmethod
    def get_metadata_from_path(file_path: str) -> Dict[str, Any]:
        """
        Extrai metadados do caminho do arquivo.
        
        Args:
            file_path: Caminho para o arquivo de resultados
            
        Returns:
            Dicionário com metadados extraídos
        """
        metadata = {}
        
        try:
            # Extrai informações do caminho
            parts = Path(file_path).parts
            
            # Encontra o índice do diretório base
            base_idx = -1
            for i, part in enumerate(parts):
                if part == "spkmc":
                    base_idx = i
                    break
            
            if base_idx >= 0 and len(parts) >= base_idx + 3:
                metadata["distribution"] = parts[base_idx + 1]
                metadata["network_type"] = parts[base_idx + 2]
                
                # Extrai informações do nome do arquivo
                filename = Path(file_path).stem
                if filename.startswith("results_"):
                    params = filename.replace("results_", "").split("_")
                    
                    if metadata["network_type"].lower() == "cn" and len(params) >= 3:
                        # Para redes complexas: exponent, N, samples, [params]
                        try:
                            metadata["exponent"] = float("." + params[0]) if params[0].isdigit() else float(params[0])
                            metadata["N"] = int(params[1])
                            metadata["samples"] = int(params[2])
                        except:
                            pass
                    elif len(params) >= 2:
                        # Para outras redes: N, samples, [params]
                        try:
                            metadata["N"] = int(params[0])
                            metadata["samples"] = int(params[1])
                        except:
                            pass
        except:
            pass
        
        return metadata
    
    @staticmethod
    def format_result_for_cli(result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formata o resultado para exibição na CLI.
        
        Args:
            result: Dicionário com os resultados
            
        Returns:
            Dicionário formatado para exibição
        """
        formatted = {}
        
        # Copia os metadados
        if "metadata" in result:
            formatted["metadata"] = result["metadata"]
        
        # Formata os dados de simulação
        if "S_val" in result and "I_val" in result and "R_val" in result:
            formatted["max_infected"] = max(result["I_val"]) if result["I_val"] else 0
            formatted["final_recovered"] = result["R_val"][-1] if result["R_val"] else 0
            formatted["data_points"] = len(result["S_val"])
        
        # Adiciona informações sobre erros
        if "S_err" in result and "I_err" in result and "R_err" in result:
            formatted["has_error_data"] = True
        else:
            formatted["has_error_data"] = False
        
        return formatted