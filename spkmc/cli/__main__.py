"""
Ponto de entrada principal para a CLI do SPKMC.

Este módulo é o ponto de entrada para a CLI quando executada como um módulo Python.
Exemplo de uso: python -m spkmc.cli
"""

from spkmc.cli.commands import cli

if __name__ == "__main__":
    cli()