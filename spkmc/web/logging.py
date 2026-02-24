"""
Structured logging for the SPKMC web interface.

Provides color-coded, timestamped debug output that appears in the terminal
where ``spkmc web --verbose`` is running.  Debug messages are gated behind the
``SPKMC_VERBOSE`` environment variable so they add zero overhead in normal use.
"""

from __future__ import annotations

import os
import sys
import time

_verbose: bool = os.environ.get("SPKMC_VERBOSE", "0") == "1"
_start_time: float = time.monotonic()

# ANSI colour codes (stderr is almost always a terminal when running locally)
_RESET = "\033[0m"
_DIM = "\033[2m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_RED = "\033[31m"

# When verbose mode is active, also enable Python's standard logging so
# that logger.warning()/debug() calls in non-web modules (e.g.
# spkmc.io.experiments) are visible in the terminal.
if _verbose:
    import logging as _logging

    _handler = _logging.StreamHandler(sys.stderr)
    _handler.setFormatter(_logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
    _spkmc_logger = _logging.getLogger("spkmc")
    _spkmc_logger.setLevel(_logging.DEBUG)
    _spkmc_logger.addHandler(_handler)


def is_verbose() -> bool:
    """Return whether verbose/debug logging is enabled."""
    return _verbose


def _elapsed() -> str:
    """Return a human-readable elapsed-time string since process start."""
    return f"{time.monotonic() - _start_time:.3f}s"


def debug(module: str, message: str) -> None:
    """Print a debug message to stderr (only when verbose mode is on)."""
    if _verbose:
        print(
            f"{_DIM}[{_elapsed()}]{_RESET} {_CYAN}[{module}]{_RESET} {message}",
            file=sys.stderr,
        )


def info(module: str, message: str) -> None:
    """Print an info-level message to stderr (always visible)."""
    print(f"[{module}] {message}", file=sys.stderr)


def warn(module: str, message: str) -> None:
    """Print a warning message to stderr (always visible)."""
    print(
        f"{_YELLOW}[{module}] WARNING: {message}{_RESET}",
        file=sys.stderr,
    )


def error(module: str, message: str) -> None:
    """Print an error message to stderr (always visible)."""
    print(
        f"{_RED}[{module}] ERROR: {message}{_RESET}",
        file=sys.stderr,
    )
