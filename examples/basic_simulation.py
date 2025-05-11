#!/usr/bin/env python3
"""
Exemplo de uso programático da CLI do SPKMC.

Este script demonstra como usar a CLI do SPKMC programaticamente a partir de outro script Python,
permitindo automatizar simulações e análises.
"""

import os
import sys
import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def run_cli_command(command):
    """
    Executa um comando da CLI do SPKMC.
    
    Args:
        command: Lista com o comando e seus argumentos
        
    Returns:
        Saída do comando
    """
    print(f"Executando: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Erro ao executar o comando: {result.stderr}")
        return None
    
    return result.stdout


def run_simulation(network_type, dist_type, output_file, **kwargs):
    """
    Executa uma simulação usando a CLI do SPKMC.
    
    Args:
        network_type: Tipo de rede ('er', 'cn', 'cg')
        dist_type: Tipo de distribuição ('gamma', 'exponential')
        output_file: Caminho para salvar os resultados
        **kwargs: Parâmetros adicionais para a simulação
        
    Returns:
        Caminho para o arquivo de resultados
    """
    # Comando base
    command = ["python", "spkmc_cli.py", "run", 
               "--network-type", network_type, 
               "--dist-type", dist_type,
               "--output", output_file,
               "--no-plot"]  # Não mostrar o gráfico durante a execução
    
    # Adiciona parâmetros adicionais
    for key, value in kwargs.items():
        # Converte underscores para hífens no nome do parâmetro
        param_name = f"--{key.replace('_', '-')}"
        command.extend([param_name, str(value)])
    
    # Executa o comando
    run_cli_command(command)
    
    # Verifica se o arquivo foi criado
    if os.path.exists(output_file):
        return output_file
    else:
        print(f"Erro: O arquivo de resultados não foi criado: {output_file}")
        return None


def plot_results(result_file, output_file=None, with_error=False):
    """
    Plota os resultados de uma simulação usando a CLI do SPKMC.
    
    Args:
        result_file: Caminho para o arquivo de resultados
        output_file: Caminho para salvar o gráfico (opcional)
        with_error: Se True, mostra barras de erro (se disponíveis)
        
    Returns:
        Caminho para o arquivo de gráfico (se output_file for fornecido)
    """
    # Comando base
    command = ["python", "spkmc_cli.py", "plot", result_file]
    
    # Adiciona opções
    if with_error:
        command.append("--with-error")
    
    if output_file:
        command.extend(["--output", output_file])
    
    # Executa o comando
    run_cli_command(command)
    
    # Verifica se o arquivo foi criado
    if output_file and os.path.exists(output_file):
        return output_file
    else:
        return None


def compare_results(result_files, labels=None, output_file=None):
    """
    Compara os resultados de múltiplas simulações usando a CLI do SPKMC.
    
    Args:
        result_files: Lista de caminhos para arquivos de resultados
        labels: Lista de rótulos para cada arquivo (opcional)
        output_file: Caminho para salvar o gráfico (opcional)
        
    Returns:
        Caminho para o arquivo de gráfico (se output_file for fornecido)
    """
    # Comando base
    command = ["python", "spkmc_cli.py", "compare"] + result_files
    
    # Adiciona rótulos
    if labels:
        for label in labels:
            command.extend(["-l", label])
    
    # Adiciona opção de saída
    if output_file:
        command.extend(["--output", output_file])
    
    # Executa o comando
    run_cli_command(command)
    
    # Verifica se o arquivo foi criado
    if output_file and os.path.exists(output_file):
        return output_file
    else:
        return None


def get_info(result_file):
    """
    Obtém informações sobre uma simulação usando a CLI do SPKMC.
    
    Args:
        result_file: Caminho para o arquivo de resultados
        
    Returns:
        Informações sobre a simulação
    """
    # Comando
    command = ["python", "spkmc_cli.py", "info", "--result-file", result_file]
    
    # Executa o comando
    return run_cli_command(command)


def main():
    """Função principal do exemplo."""
    # Cria diretório para resultados
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Parâmetros comuns
    common_params = {
        "nodes": 1000,
        "samples": 50,
        "initial_perc": 0.01,
        "t_max": 10.0,
        "steps": 100,
        "num_runs": 2
    }
    
    # Executa simulação com rede Erdos-Renyi e distribuição Gamma
    er_gamma_file = str(results_dir / "er_gamma.json")
    er_gamma_params = {
        **common_params,
        "k_avg": 10,
        "shape": 2.0,
        "scale": 1.0,
        "lambda_val": 1.0
    }
    er_gamma_result = run_simulation("er", "gamma", er_gamma_file, **er_gamma_params)
    
    # Executa simulação com rede Erdos-Renyi e distribuição Exponencial
    er_exp_file = str(results_dir / "er_exponential.json")
    er_exp_params = {
        **common_params,
        "k_avg": 10,
        "mu": 1.0,
        "lambda_val": 1.0
    }
    er_exp_result = run_simulation("er", "exponential", er_exp_file, **er_exp_params)
    
    # Executa simulação com rede Complexa e distribuição Gamma
    cn_gamma_file = str(results_dir / "cn_gamma.json")
    cn_gamma_params = {
        **common_params,
        "k_avg": 10,
        "exponent": 2.5,
        "shape": 2.0,
        "scale": 1.0,
        "lambda_val": 1.0
    }
    cn_gamma_result = run_simulation("cn", "gamma", cn_gamma_file, **cn_gamma_params)
    
    # Plota os resultados individuais
    if er_gamma_result:
        plot_results(er_gamma_result, str(plots_dir / "er_gamma.png"), with_error=True)
        print(f"Informações sobre a simulação ER-Gamma:")
        print(get_info(er_gamma_result))
    
    if er_exp_result:
        plot_results(er_exp_result, str(plots_dir / "er_exponential.png"), with_error=True)
    
    if cn_gamma_result:
        plot_results(cn_gamma_result, str(plots_dir / "cn_gamma.png"), with_error=True)
    
    # Compara os resultados
    if er_gamma_result and er_exp_result:
        compare_results(
            [er_gamma_result, er_exp_result],
            labels=["ER-Gamma", "ER-Exponential"],
            output_file=str(plots_dir / "comparison_er.png")
        )
    
    if er_gamma_result and cn_gamma_result:
        compare_results(
            [er_gamma_result, cn_gamma_result],
            labels=["ER-Gamma", "CN-Gamma"],
            output_file=str(plots_dir / "comparison_networks.png")
        )
    
    print("\nSimulações concluídas. Resultados salvos em:")
    print(f"- Arquivos de resultados: {results_dir}")
    print(f"- Gráficos: {plots_dir}")


if __name__ == "__main__":
    main()