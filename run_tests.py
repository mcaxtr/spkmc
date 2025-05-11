#!/usr/bin/env python3
"""
Script para executar os testes do SPKMC.

Este script executa os testes unitários e de integração do SPKMC,
gerando relatórios de cobertura de código.
"""

import os
import sys
import subprocess
import argparse


def parse_args():
    """Analisa os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description="Executa os testes do SPKMC")
    parser.add_argument("--unit", action="store_true", help="Executar apenas testes unitários")
    parser.add_argument("--integration", action="store_true", help="Executar apenas testes de integração")
    parser.add_argument("--coverage", action="store_true", help="Gerar relatório de cobertura HTML")
    parser.add_argument("--verbose", "-v", action="store_true", help="Modo verboso")
    return parser.parse_args()


def run_tests(args):
    """Executa os testes com base nos argumentos fornecidos."""
    # Configurar comando base
    cmd = ["pytest"]
    
    # Adicionar opções
    if args.verbose:
        cmd.append("-v")
    
    if args.coverage:
        cmd.extend(["--cov=spkmc", "--cov-report=html"])
    
    # Selecionar testes
    if args.unit:
        cmd.append("tests/test_distributions.py tests/test_networks.py tests/test_cli_validators.py tests/test_results.py tests/test_export.py")
    elif args.integration:
        cmd.append("tests/test_cli_commands.py tests/test_simulation.py tests/test_integration.py")
    
    # Executar comando
    cmd_str = " ".join(cmd)
    print(f"Executando: {cmd_str}")
    return subprocess.call(cmd_str, shell=True)


def main():
    """Função principal."""
    args = parse_args()
    
    # Se nenhum tipo de teste for especificado, executar todos
    if not (args.unit or args.integration):
        args.unit = True
        args.integration = True
    
    # Executar testes
    result = run_tests(args)
    
    # Exibir mensagem de resultado
    if result == 0:
        print("\n✅ Todos os testes passaram!")
        
        if args.coverage:
            print("\nRelatório de cobertura gerado em: htmlcov/index.html")
    else:
        print("\n❌ Alguns testes falharam.")
    
    return result


if __name__ == "__main__":
    sys.exit(main())