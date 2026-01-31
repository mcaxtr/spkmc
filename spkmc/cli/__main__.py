"""
Main entry point for the SPKMC CLI.

This module is the entry point for the CLI when run as a Python module.
Example usage: python -m spkmc.cli
"""

from spkmc.cli.commands import cli

if __name__ == "__main__":
    cli()
