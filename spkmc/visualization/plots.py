"""
Visualização de resultados para o algoritmo SPKMC.

Este módulo contém funções para visualização dos resultados de simulações SPKMC,
incluindo gráficos de evolução temporal dos estados SIR e comparações entre diferentes
simulações.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import os


class Visualizer:
    """Classe para visualização de resultados."""
    
    @staticmethod
    def plot_result_with_error(S: np.ndarray, I: np.ndarray, R: np.ndarray, 
                              S_err: np.ndarray, I_err: np.ndarray, R_err: np.ndarray, 
                              time: np.ndarray, title: Optional[str] = None,
                              save_path: Optional[str] = None) -> None:
        """
        Plota os resultados com barras de erro.
        
        Args:
            S: Proporção de suscetíveis
            I: Proporção de infectados
            R: Proporção de recuperados
            S_err: Erro para suscetíveis
            I_err: Erro para infectados
            R_err: Erro para recuperados
            time: Passos de tempo
            title: Título do gráfico (opcional)
            save_path: Caminho para salvar o gráfico (opcional)
        """
        plt.figure(figsize=(10, 6))
        plt.errorbar(time, R, yerr=R_err, label='Recuperados', capsize=2, color='g')
        plt.errorbar(time, I, yerr=I_err, label='Infectados', capsize=2, color='r')
        plt.errorbar(time, S, yerr=S_err, label='Suscetíveis', capsize=2, color='b')
        
        plt.xlabel('Tempo')
        plt.ylabel('Proporção de Indivíduos')
        
        if title:
            plt.title(title)
        else:
            plt.title('Dinâmica do Modelo SIR ao Longo do Tempo com Barras de Erro')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_result(S: np.ndarray, I: np.ndarray, R: np.ndarray, time: np.ndarray,
                   title: Optional[str] = None, save_path: Optional[str] = None) -> None:
        """
        Plota os resultados sem barras de erro.
        
        Args:
            S: Proporção de suscetíveis
            I: Proporção de infectados
            R: Proporção de recuperados
            time: Passos de tempo
            title: Título do gráfico (opcional)
            save_path: Caminho para salvar o gráfico (opcional)
        """
        plt.figure(figsize=(10, 6))
        plt.plot(time, R, 'g-', label='Recuperados')
        plt.plot(time, I, 'r-', label='Infectados')
        plt.plot(time, S, 'b-', label='Suscetíveis')
        
        plt.xlabel('Tempo')
        plt.ylabel('Proporção de Indivíduos')
        
        if title:
            plt.title(title)
        else:
            plt.title('Dinâmica do Modelo SIR ao Longo do Tempo')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def compare_results(results: List[Dict[str, Any]], labels: List[str],
                       title: Optional[str] = None, save_path: Optional[str] = None) -> None:
        """
        Compara resultados de múltiplas simulações.
        
        Args:
            results: Lista de dicionários com resultados
            labels: Lista de rótulos para cada resultado
            title: Título do gráfico (opcional)
            save_path: Caminho para salvar o gráfico (opcional)
        """
        if not results:
            raise ValueError("A lista de resultados está vazia")
        
        if len(results) != len(labels):
            raise ValueError("O número de resultados e rótulos deve ser igual")
        
        plt.figure(figsize=(12, 8))
        
        # Cores para cada conjunto de dados
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
        
        for i, (result, label) in enumerate(zip(results, labels)):
            if not all(key in result for key in ["S_val", "I_val", "R_val", "time"]):
                raise ValueError(f"Resultado {i} não contém todos os dados necessários")
            
            S = np.array(result["S_val"])
            I = np.array(result["I_val"])
            R = np.array(result["R_val"])
            time = np.array(result["time"])
            
            color = colors[i % len(colors)]
            linestyle = '-' if i < len(colors) else '--'
            
            plt.plot(time, S, f'{color}', linestyle=linestyle, alpha=0.7, label=f'S - {label}')
            plt.plot(time, I, f'{color}', linestyle=linestyle, alpha=0.7, label=f'I - {label}')
            plt.plot(time, R, f'{color}', linestyle=linestyle, alpha=0.7, label=f'R - {label}')
        
        plt.xlabel('Tempo')
        plt.ylabel('Proporção de Indivíduos')
        
        if title:
            plt.title(title)
        else:
            plt.title('Comparação de Simulações SPKMC')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_network(G, title: Optional[str] = None, save_path: Optional[str] = None) -> None:
        """
        Plota a rede utilizada na simulação.
        
        Args:
            G: Grafo da rede
            title: Título do gráfico (opcional)
            save_path: Caminho para salvar o gráfico (opcional)
        """
        plt.figure(figsize=(10, 8))
        
        # Limita o número de nós para visualização
        if G.number_of_nodes() > 100:
            import warnings
            warnings.warn(f"A rede tem {G.number_of_nodes()} nós. Limitando a visualização a 100 nós.")
            import networkx as nx
            G = nx.DiGraph(G.subgraph(list(G.nodes())[:100]))
        
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx(G, pos, with_labels=False, node_size=30, 
                         node_color='skyblue', edge_color='gray', 
                         arrows=True, arrowsize=10, alpha=0.8)
        
        if title:
            plt.title(title)
        else:
            plt.title(f'Visualização da Rede ({G.number_of_nodes()} nós, {G.number_of_edges()} arestas)')
        
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def create_summary_plot(result_path: str, output_dir: Optional[str] = None) -> str:
        """
        Cria um gráfico resumo a partir de um arquivo de resultados.
        
        Args:
            result_path: Caminho para o arquivo de resultados
            output_dir: Diretório para salvar o gráfico (opcional)
        
        Returns:
            Caminho para o gráfico gerado
        """
        import json
        
        # Carrega os resultados
        with open(result_path, 'r') as f:
            result = json.load(f)
        
        # Extrai os dados
        S = np.array(result.get("S_val", []))
        I = np.array(result.get("I_val", []))
        R = np.array(result.get("R_val", []))
        time = np.array(result.get("time", []))
        
        # Verifica se há dados de erro
        has_error = "S_err" in result and "I_err" in result and "R_err" in result
        
        # Cria o diretório de saída se não existir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.basename(result_path).replace(".json", ".png")
            save_path = os.path.join(output_dir, base_name)
        else:
            save_path = result_path.replace(".json", ".png")
        
        # Extrai metadados para o título
        metadata = result.get("metadata", {})
        network_type = metadata.get("network_type", "").upper()
        dist_type = metadata.get("distribution", "").capitalize()
        N = metadata.get("N", "")
        
        title = f"Simulação SPKMC - Rede {network_type}, Distribuição {dist_type}, N={N}"
        
        # Plota os resultados
        if has_error:
            S_err = np.array(result.get("S_err", []))
            I_err = np.array(result.get("I_err", []))
            R_err = np.array(result.get("R_err", []))
            Visualizer.plot_result_with_error(S, I, R, S_err, I_err, R_err, time, title, save_path)
        else:
            Visualizer.plot_result(S, I, R, time, title, save_path)
        
        return save_path