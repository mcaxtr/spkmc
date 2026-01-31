#!/usr/bin/env python3
"""
Script to run SPKMC tests.

This script runs SPKMC unit and integration tests,
generating code coverage reports.
"""

import argparse
import os
import subprocess
import sys


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run SPKMC tests")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--coverage", action="store_true", help="Generate HTML coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose mode")
    return parser.parse_args()


def run_tests(args):
    """Run tests based on the provided arguments."""
    # Configure base command
    cmd = ["pytest"]

    # Add options
    if args.verbose:
        cmd.append("-v")

    if args.coverage:
        cmd.extend(["--cov=spkmc", "--cov-report=html"])

    # Select tests
    if args.unit:
        cmd.append(
            "tests/test_distributions.py tests/test_networks.py tests/test_cli_validators.py tests/test_results.py tests/test_export.py"
        )
    elif args.integration:
        cmd.append("tests/test_cli_commands.py tests/test_simulation.py tests/test_integration.py")

    # Execute command
    cmd_str = " ".join(cmd)
    print(f"Running: {cmd_str}")
    return subprocess.call(cmd_str, shell=True)


def main():
    """Main entry point."""
    args = parse_args()

    # If no test type is specified, run all
    if not (args.unit or args.integration):
        args.unit = True
        args.integration = True

    # Run tests
    result = run_tests(args)

    # Show result message
    if result == 0:
        print("\n✅ All tests passed!")

        if args.coverage:
            print("\nCoverage report generated at: htmlcov/index.html")
    else:
        print("\n❌ Some tests failed.")

    return result


if __name__ == "__main__":
    sys.exit(main())
