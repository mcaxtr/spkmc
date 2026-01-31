"""
Formatting utilities for the SPKMC CLI.

This module provides functions and classes to format and color CLI output,
improving the user experience with colors, styles, and rich formatting.
"""

import os
import platform
import sys
from typing import Any, Dict, List, Optional

# Check whether colors should be disabled
NO_COLOR = "--no-color" in sys.argv


# Check whether the terminal supports colors
def supports_color() -> bool:
    """
    Check whether the current terminal supports ANSI colors.
    Based on Django and Pytest logic.
    """
    if NO_COLOR:
        return False

    # Check environment variables indicating color support
    if os.environ.get("FORCE_COLOR", "0") != "0":
        return True
    if os.environ.get("NO_COLOR", "0") != "0":
        return False
    if os.environ.get("TERM") == "dumb":
        return False

    # Check whether this is an interactive terminal
    is_a_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    # Check platform
    plat = platform.system()
    if plat == "Windows":
        # On Windows, check whether this is a modern terminal
        return is_a_tty and (
            "ANSICON" in os.environ
            or "WT_SESSION" in os.environ  # Windows Terminal
            or os.environ.get("TERM_PROGRAM") == "vscode"  # VS Code
            or "ConEmuANSI" in os.environ
            or os.environ.get("TERM") == "xterm"
        )
    else:
        # On Unix, most terminals support colors
        return is_a_tty


# Check whether colors are enabled
COLORS_ENABLED = supports_color()

# Conditional imports for colorama
try:
    from colorama import Back, Fore, Style, init

    # Initialize colorama for all platforms.
    # Always initialize, even if colors are disabled, so ANSI escape codes are handled.
    init(autoreset=True, strip=not COLORS_ENABLED)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    if COLORS_ENABLED:
        print("Warning: Colorama not found. Colored output will be unavailable.")

    # Define empty classes to avoid errors
    class DummyColor:
        def __getattr__(self, name: str) -> str:
            return ""

    class DummyStyle:
        def __getattr__(self, name: str) -> str:
            return ""

    Fore = DummyColor()
    Back = DummyColor()
    Style = DummyStyle()

# Conditional imports for rich
# Declare console at module level to avoid mypy redefinition error
console: Any = None

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table

    RICH_AVAILABLE = True
    # Rich console for advanced output, respecting color settings
    console = Console(color_system="auto" if COLORS_ENABLED else None, highlight=False)
except ImportError:
    RICH_AVAILABLE = False
    if COLORS_ENABLED:
        print("Warning: Rich not found. Advanced formatting will be unavailable.")

    # Create a simple console class to avoid errors
    class SimpleConsole:
        def print(self, text: str, *args: Any, **kwargs: Any) -> None:
            # Remove Rich formatting tags if present
            import re

            text = re.sub(r"\[.*?\]", "", text)
            print(text)

    console = SimpleConsole()

# Color mapping for SIR states and other elements
COLOR_MAP = {
    "S": Fore.BLUE,
    "I": Fore.RED,
    "R": Fore.GREEN,
    "title": Fore.CYAN,
    "info": Fore.YELLOW,
    "success": Fore.GREEN,
    "error": Fore.RED,
    "warning": Fore.YELLOW,
    "param": Fore.MAGENTA,
    "value": Fore.WHITE,
}


def colorize(text: str, color: str) -> str:
    """
    Colorize text with the specified color.

    Args:
        text: Text to colorize
        color: Color name (must be in COLOR_MAP)

    Returns:
        Colorized text
    """
    if not COLORS_ENABLED:
        return text

    # Use colorama directly to avoid ANSI escape code issues
    if COLORAMA_AVAILABLE and color in COLOR_MAP:
        return f"{COLOR_MAP[color]}{text}{Style.RESET_ALL}"

    return text


def format_title(title: str) -> str:
    """
    Format a title for CLI display.

    Args:
        title: Title to format

    Returns:
        Formatted title
    """
    if not COLORS_ENABLED:
        return f"\n{title}\n{'-' * len(title)}"

    # Use colorama directly to avoid ANSI escape code issues
    if COLORAMA_AVAILABLE:
        return f"\n{Fore.CYAN}{Style.BRIGHT}{title}{Style.RESET_ALL}\n{'-' * len(title)}"
    else:
        return f"\n{title}\n{'-' * len(title)}"


def format_param(name: str, value: Any) -> str:
    """
    Format a parameter and its value for CLI display.

    Args:
        name: Parameter name
        value: Parameter value

    Returns:
        Formatted parameter
    """
    if not COLORS_ENABLED:
        return f"{name}: {value}"

    # Use colorama directly to avoid ANSI escape code issues
    if COLORAMA_AVAILABLE:
        return f"{Fore.MAGENTA}{name}{Style.RESET_ALL}: {Fore.WHITE}{value}{Style.RESET_ALL}"
    else:
        return f"{name}: {value}"


def format_success(message: str) -> str:
    """
    Format a success message.

    Args:
        message: Success message

    Returns:
        Formatted message
    """
    if not COLORS_ENABLED:
        return f"✓ {message}"

    # Use colorama directly to avoid ANSI escape code issues
    if COLORAMA_AVAILABLE:
        return f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}"
    else:
        return f"✓ {message}"


def format_error(message: str) -> str:
    """
    Format an error message.

    Args:
        message: Error message

    Returns:
        Formatted message
    """
    if not COLORS_ENABLED:
        return f"✗ {message}"

    # Use colorama directly to avoid ANSI escape code issues
    if COLORAMA_AVAILABLE:
        return f"{Fore.RED}✗ {message}{Style.RESET_ALL}"
    else:
        return f"✗ {message}"


def format_warning(message: str) -> str:
    """
    Format a warning message.

    Args:
        message: Warning message

    Returns:
        Formatted message
    """
    if not COLORS_ENABLED:
        return f"⚠ {message}"

    # Use colorama directly to avoid ANSI escape code issues
    if COLORAMA_AVAILABLE:
        return f"{Fore.YELLOW}⚠ {message}{Style.RESET_ALL}"
    else:
        return f"⚠ {message}"


def format_info(message: str) -> str:
    """
    Format an informational message.

    Args:
        message: Informational message

    Returns:
        Formatted message
    """
    if not COLORS_ENABLED:
        return f"ℹ {message}"

    # Use colorama directly to avoid ANSI escape code issues
    if COLORAMA_AVAILABLE:
        return f"{Fore.YELLOW}ℹ {message}{Style.RESET_ALL}"
    else:
        return f"ℹ {message}"


def create_progress_bar(description: str, total: int, verbose: bool = False) -> Any:
    """
    Create an advanced progress bar.

    Args:
        description: Task description
        total: Total items
        verbose: If True, show additional details

    Returns:
        Configured Progress object or a dummy object if rich is unavailable
    """
    if not RICH_AVAILABLE:
        # Return a dummy object that mimics the Progress API
        class DummyProgress:
            def __enter__(self) -> "DummyProgress":
                return self

            def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                pass

            def add_task(self, description: str, total: Optional[int] = None) -> int:
                print(f"{description} (Total: {total})")
                return 0

            def update(self, task_id: int, advance: Optional[int] = None) -> None:
                if advance:
                    print(f"Progress: advanced {advance}")

        return DummyProgress()

    if verbose:
        # Verbose mode: show all details including items completed and elapsed time
        return Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            MofNCompleteColumn(),
            "•",
            TimeElapsedColumn(),
            "<",
            TimeRemainingColumn(),
            console=console,
            expand=True,
        )
    else:
        # Default mode: show percentage, elapsed time, and ETA
        return Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            TimeElapsedColumn(),
            "<",
            TimeRemainingColumn(),
            console=console,
            expand=True,
        )


def print_rich_table(data: List[Dict[str, Any]], title: str) -> None:
    """
    Print a formatted table with Rich.

    Args:
        data: List of dictionaries with data
        title: Table title
    """
    if not data:
        console.print(format_warning("No data to display."))
        return

    if not RICH_AVAILABLE:
        # Fallback to simple printing
        print(f"\n{title}\n{'-' * len(title)}")
        # Print headers
        headers = list(data[0].keys())
        print(" | ".join(headers))
        print("-" * (sum(len(h) for h in headers) + 3 * (len(headers) - 1)))
        # Print rows
        for item in data:
            print(" | ".join(str(v) for v in item.values()))
        return

    table = Table(title=title)

    # Add columns
    for key in data[0].keys():
        table.add_column(key, style="cyan")

    # Add rows
    for item in data:
        table.add_row(*[str(v) for v in item.values()])

    console.print(table)


def print_markdown(markdown_text: str) -> None:
    """
    Render and print Markdown text.

    Args:
        markdown_text: Markdown text
    """
    if not RICH_AVAILABLE:
        # Fallback to simple printing
        print(markdown_text)
        return

    md = Markdown(markdown_text)
    console.print(md)


def print_panel(content: str, title: Optional[str] = None) -> None:
    """
    Print a panel with content.

    Args:
        content: Panel content
        title: Panel title (optional)
    """
    if not RICH_AVAILABLE:
        # Fallback to simple printing
        if title:
            print(f"\n{title}\n{'-' * len(title)}")
        print(content)
        return

    panel = Panel(content, title=title)
    console.print(panel)


def is_verbose_mode() -> bool:
    """
    Check whether verbose mode is enabled.

    Returns:
        True if verbose mode is enabled, False otherwise
    """
    return "--verbose" in sys.argv or "-v" in sys.argv


def log_debug(message: str, verbose_only: bool = True) -> None:
    """
    Log a debug message.

    Args:
        message: Debug message
        verbose_only: If True, only show in verbose mode
    """
    if not verbose_only or is_verbose_mode():
        if RICH_AVAILABLE and COLORS_ENABLED:
            console.print(f"[dim][DEBUG] {message}[/dim]")
        else:
            print(f"[DEBUG] {message}")


def log_info(message: str) -> None:
    """
    Log an informational message.

    Args:
        message: Informational message
    """
    if RICH_AVAILABLE and COLORS_ENABLED:
        console.print(f"[blue]ℹ {message}[/blue]")
    else:
        print(f"ℹ {message}")


def log_success(message: str) -> None:
    """
    Log a success message.

    Args:
        message: Success message
    """
    if RICH_AVAILABLE and COLORS_ENABLED:
        console.print(f"[green]✓ {message}[/green]")
    else:
        print(f"✓ {message}")


def log_warning(message: str) -> None:
    """
    Log a warning message.

    Args:
        message: Warning message
    """
    if RICH_AVAILABLE and COLORS_ENABLED:
        console.print(f"[yellow]⚠ {message}[/yellow]")
    else:
        print(f"⚠ {message}")


def log_error(message: str) -> None:
    """
    Log an error message.

    Args:
        message: Error message
    """
    if RICH_AVAILABLE and COLORS_ENABLED:
        console.print(f"[red]✗ {message}[/red]")
    else:
        print(f"✗ {message}")
