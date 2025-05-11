"""
Utilitários de formatação para a CLI do SPKMC.

Este módulo contém funções e classes para formatação e coloração da saída da CLI,
melhorando a experiência do usuário com cores, estilos e formatação avançada.
"""

import os
import sys
import platform
from typing import Dict, List, Tuple, Optional, Any, Union

# Verificar se as cores devem ser desabilitadas
NO_COLOR = "--no-color" in sys.argv

# Verificar se o terminal suporta cores
def supports_color():
    """
    Verifica se o terminal atual suporta cores ANSI.
    Baseado na lógica do Django e do Pytest.
    """
    if NO_COLOR:
        return False
        
    # Verificar variáveis de ambiente que indicam suporte a cores
    if os.environ.get("FORCE_COLOR", "0") != "0":
        return True
    if os.environ.get("NO_COLOR", "0") != "0":
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    
    # Verificar se é um terminal interativo
    is_a_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    
    # Verificar plataforma
    plat = platform.system()
    if plat == "Windows":
        # No Windows, verificar se é um terminal moderno
        return is_a_tty and (
            "ANSICON" in os.environ
            or "WT_SESSION" in os.environ  # Windows Terminal
            or os.environ.get("TERM_PROGRAM") == "vscode"  # VS Code
            or "ConEmuANSI" in os.environ
            or os.environ.get("TERM") == "xterm"
        )
    else:
        # No Unix, a maioria dos terminais suporta cores
        return is_a_tty

# Verificar se as cores estão habilitadas
COLORS_ENABLED = supports_color()

# Importações condicionais para colorama
try:
    from colorama import Fore, Back, Style, init
    # Inicializa o colorama para funcionar em todos os sistemas operacionais
    # Sempre inicializar o colorama, mesmo se as cores estiverem desabilitadas
    # Isso garante que os códigos de escape ANSI sejam processados corretamente
    init(autoreset=True, strip=not COLORS_ENABLED)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    if COLORS_ENABLED:
        print("Aviso: Colorama não encontrado. A saída colorida não estará disponível.")
    
    # Definir classes vazias para evitar erros
    class DummyColor:
        def __getattr__(self, name):
            return ""
    
    class DummyStyle:
        def __getattr__(self, name):
            return ""
    
    Fore = DummyColor()
    Back = DummyColor()
    Style = DummyStyle()

# Importações condicionais para rich
try:
    from rich.console import Console
    from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.panel import Panel
    RICH_AVAILABLE = True
    # Console Rich para saída avançada, respeitando a configuração de cores
    console = Console(color_system="auto" if COLORS_ENABLED else None, highlight=False)
except ImportError:
    RICH_AVAILABLE = False
    if COLORS_ENABLED:
        print("Aviso: Rich não encontrado. A formatação avançada não estará disponível.")
    # Criar uma classe console simples para evitar erros
    class SimpleConsole:
        def print(self, text, *args, **kwargs):
            # Remover tags de formatação Rich se presentes
            import re
            text = re.sub(r'\[.*?\]', '', text)
            print(text)
    
    console = SimpleConsole()

# Mapeamento de cores para estados SIR e outros elementos
COLOR_MAP = {
    'S': Fore.BLUE,
    'I': Fore.RED,
    'R': Fore.GREEN,
    'title': Fore.CYAN,
    'info': Fore.YELLOW,
    'success': Fore.GREEN,
    'error': Fore.RED,
    'warning': Fore.YELLOW,
    'param': Fore.MAGENTA,
    'value': Fore.WHITE
}


def colorize(text: str, color: str) -> str:
    """
    Colore o texto com a cor especificada.
    
    Args:
        text: Texto a ser colorido
        color: Nome da cor (deve estar em COLOR_MAP)
        
    Returns:
        Texto colorido
    """
    if not COLORS_ENABLED:
        return text
    
    # Usar colorama diretamente para evitar problemas com códigos de escape
    if COLORAMA_AVAILABLE and color in COLOR_MAP:
        return f"{COLOR_MAP[color]}{text}{Style.RESET_ALL}"
    
    return text


def format_title(title: str) -> str:
    """
    Formata um título para exibição na CLI.
    
    Args:
        title: Título a ser formatado
        
    Returns:
        Título formatado
    """
    if not COLORS_ENABLED:
        return f"\n{title}\n{'-' * len(title)}"
    
    # Usar colorama diretamente para evitar problemas com códigos de escape
    if COLORAMA_AVAILABLE:
        return f"\n{Fore.CYAN}{Style.BRIGHT}{title}{Style.RESET_ALL}\n{'-' * len(title)}"
    else:
        return f"\n{title}\n{'-' * len(title)}"


def format_param(name: str, value: Any) -> str:
    """
    Formata um parâmetro e seu valor para exibição na CLI.
    
    Args:
        name: Nome do parâmetro
        value: Valor do parâmetro
        
    Returns:
        Parâmetro formatado
    """
    if not COLORS_ENABLED:
        return f"{name}: {value}"
    
    # Usar colorama diretamente para evitar problemas com códigos de escape
    if COLORAMA_AVAILABLE:
        return f"{Fore.MAGENTA}{name}{Style.RESET_ALL}: {Fore.WHITE}{value}{Style.RESET_ALL}"
    else:
        return f"{name}: {value}"


def format_success(message: str) -> str:
    """
    Formata uma mensagem de sucesso.
    
    Args:
        message: Mensagem de sucesso
        
    Returns:
        Mensagem formatada
    """
    if not COLORS_ENABLED:
        return f"✓ {message}"
    
    # Usar colorama diretamente para evitar problemas com códigos de escape
    if COLORAMA_AVAILABLE:
        return f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}"
    else:
        return f"✓ {message}"


def format_error(message: str) -> str:
    """
    Formata uma mensagem de erro.
    
    Args:
        message: Mensagem de erro
        
    Returns:
        Mensagem formatada
    """
    if not COLORS_ENABLED:
        return f"✗ {message}"
    
    # Usar colorama diretamente para evitar problemas com códigos de escape
    if COLORAMA_AVAILABLE:
        return f"{Fore.RED}✗ {message}{Style.RESET_ALL}"
    else:
        return f"✗ {message}"


def format_warning(message: str) -> str:
    """
    Formata uma mensagem de aviso.
    
    Args:
        message: Mensagem de aviso
        
    Returns:
        Mensagem formatada
    """
    if not COLORS_ENABLED:
        return f"⚠ {message}"
    
    # Usar colorama diretamente para evitar problemas com códigos de escape
    if COLORAMA_AVAILABLE:
        return f"{Fore.YELLOW}⚠ {message}{Style.RESET_ALL}"
    else:
        return f"⚠ {message}"


def format_info(message: str) -> str:
    """
    Formata uma mensagem informativa.
    
    Args:
        message: Mensagem informativa
        
    Returns:
        Mensagem formatada
    """
    if not COLORS_ENABLED:
        return f"ℹ {message}"
    
    # Usar colorama diretamente para evitar problemas com códigos de escape
    if COLORAMA_AVAILABLE:
        return f"{Fore.YELLOW}ℹ {message}{Style.RESET_ALL}"
    else:
        return f"ℹ {message}"


def create_progress_bar(description: str, total: int, verbose: bool = False) -> Any:
    """
    Cria uma barra de progresso avançada.
    
    Args:
        description: Descrição da tarefa
        total: Total de itens
        verbose: Se True, mostra informações adicionais
        
    Returns:
        Objeto Progress configurado ou um objeto dummy se rich não estiver disponível
    """
    if not RICH_AVAILABLE:
        # Retorna um objeto dummy que simula a API do Progress
        class DummyProgress:
            def __enter__(self):
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
                
            def add_task(self, description, total=None):
                print(f"{description} (Total: {total})")
                return 0
                
            def update(self, task_id, advance=None):
                if advance:
                    print(f"Progresso: avançou {advance}")
        
        return DummyProgress()
        
    if verbose:
        return Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            TaskProgressColumn(),
            "•",
            TimeRemainingColumn(),
            console=console,
            expand=True
        )
    else:
        return Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            console=console,
            expand=True
        )


def print_rich_table(data: List[Dict[str, Any]], title: str) -> None:
    """
    Imprime uma tabela formatada com Rich.
    
    Args:
        data: Lista de dicionários com os dados
        title: Título da tabela
    """
    if not data:
        console.print(format_warning("Nenhum dado para exibir."))
        return
    
    if not RICH_AVAILABLE:
        # Fallback para impressão simples
        print(f"\n{title}\n{'-' * len(title)}")
        # Imprimir cabeçalhos
        headers = list(data[0].keys())
        print(" | ".join(headers))
        print("-" * (sum(len(h) for h in headers) + 3 * (len(headers) - 1)))
        # Imprimir linhas
        for item in data:
            print(" | ".join(str(v) for v in item.values()))
        return
    
    table = Table(title=title)
    
    # Adiciona as colunas
    for key in data[0].keys():
        table.add_column(key, style="cyan")
    
    # Adiciona as linhas
    for item in data:
        table.add_row(*[str(v) for v in item.values()])
    
    console.print(table)


def print_markdown(markdown_text: str) -> None:
    """
    Renderiza e imprime texto em formato Markdown.
    
    Args:
        markdown_text: Texto em formato Markdown
    """
    if not RICH_AVAILABLE:
        # Fallback para impressão simples
        print(markdown_text)
        return
        
    md = Markdown(markdown_text)
    console.print(md)


def print_panel(content: str, title: str = None) -> None:
    """
    Imprime um painel com conteúdo.
    
    Args:
        content: Conteúdo do painel
        title: Título do painel (opcional)
    """
    if not RICH_AVAILABLE:
        # Fallback para impressão simples
        if title:
            print(f"\n{title}\n{'-' * len(title)}")
        print(content)
        return
        
    panel = Panel(content, title=title)
    console.print(panel)


def is_verbose_mode() -> bool:
    """
    Verifica se o modo verboso está ativado.
    
    Returns:
        True se o modo verboso estiver ativado, False caso contrário
    """
    return '--verbose' in sys.argv or '-v' in sys.argv


def log_debug(message: str, verbose_only: bool = True) -> None:
    """
    Registra uma mensagem de depuração.
    
    Args:
        message: Mensagem de depuração
        verbose_only: Se True, só exibe a mensagem no modo verboso
    """
    if not verbose_only or is_verbose_mode():
        if RICH_AVAILABLE and COLORS_ENABLED:
            console.print(f"[dim][DEBUG] {message}[/dim]")
        else:
            print(f"[DEBUG] {message}")


def log_info(message: str) -> None:
    """
    Registra uma mensagem informativa.
    
    Args:
        message: Mensagem informativa
    """
    if RICH_AVAILABLE and COLORS_ENABLED:
        console.print(f"[blue]ℹ {message}[/blue]")
    else:
        print(f"ℹ {message}")


def log_success(message: str) -> None:
    """
    Registra uma mensagem de sucesso.
    
    Args:
        message: Mensagem de sucesso
    """
    if RICH_AVAILABLE and COLORS_ENABLED:
        console.print(f"[green]✓ {message}[/green]")
    else:
        print(f"✓ {message}")


def log_warning(message: str) -> None:
    """
    Registra uma mensagem de aviso.
    
    Args:
        message: Mensagem de aviso
    """
    if RICH_AVAILABLE and COLORS_ENABLED:
        console.print(f"[yellow]⚠ {message}[/yellow]")
    else:
        print(f"⚠ {message}")


def log_error(message: str) -> None:
    """
    Registra uma mensagem de erro.
    
    Args:
        message: Mensagem de erro
    """
    if RICH_AVAILABLE and COLORS_ENABLED:
        console.print(f"[red]✗ {message}[/red]")
    else:
        print(f"✗ {message}")