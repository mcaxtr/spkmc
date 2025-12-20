"""
Comandos da CLI do SPKMC.

Este módulo contém os comandos da interface de linha de comando para o algoritmo SPKMC,
incluindo comandos para executar simulações, visualizar resultados e obter informações.
"""

import os
import json
import click
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import time
import zipfile
from datetime import datetime
import questionary
from questionary import Style

from spkmc.core.distributions import create_distribution
from spkmc.core.networks import NetworkFactory
from spkmc.core.simulation import SPKMC
from spkmc.io.results import ResultManager
from spkmc.io.experiments import ExperimentManager, Experiment, PlotConfig
from spkmc.visualization.plots import Visualizer
from spkmc.io.export import ExportManager
from spkmc.utils.hardware import (
    get_hardware_info,
    ParallelizationStrategy,
    format_hardware_box,
    HardwareInfo
)
from spkmc.utils.parallel import ParallelBatchExecutor, ScenarioResult
from spkmc.cli.formatting import (
    colorize, format_title, format_param, format_success, format_error,
    format_warning, format_info, create_progress_bar, print_rich_table,
    print_markdown, print_panel, log_debug, log_info, log_success,
    log_warning, log_error, console
)
from spkmc.cli.validators import (
    validate_percentage,
    validate_positive,
    validate_positive_int,
    validate_network_type,
    validate_distribution_type,
    validate_exponent,
    validate_file_exists,
    validate_directory_exists,
    validate_output_file,
    validate_conditional
)


# Constantes para valores padrão
DEFAULT_N = 1000
DEFAULT_K_AVG = 10
DEFAULT_SAMPLES = 50
DEFAULT_INITIAL_PERC = 0.01
DEFAULT_T_MAX = 10.0
DEFAULT_STEPS = 100
DEFAULT_NUM_RUNS = 2
DEFAULT_SHAPE = 2.0
DEFAULT_SCALE = 1.0
DEFAULT_MU = 1.0
DEFAULT_LAMBDA = 1.0
DEFAULT_EXPONENT = 2.5


def display_experiments_menu(experiments: List[Experiment]) -> Optional[int]:
    """
    Exibe um menu interativo para seleção de experimento com navegação por setas.

    Args:
        experiments: Lista de experimentos disponíveis

    Returns:
        Índice do experimento selecionado (1-based) ou None para sair
    """
    # Custom style for the menu
    custom_style = Style([
        ('qmark', 'fg:cyan bold'),
        ('question', 'fg:white bold'),
        ('answer', 'fg:green bold'),
        ('pointer', 'fg:cyan bold'),
        ('highlighted', 'fg:cyan bold'),
        ('selected', 'fg:green'),
        ('separator', 'fg:gray'),
        ('instruction', 'fg:gray italic'),
    ])

    # Build choices with formatted display
    choices = []
    for i, exp in enumerate(experiments):
        scenarios_count = len(exp.scenarios)
        results_count = exp.result_count

        # Status indicator
        if results_count == 0:
            status = "○ Pending"
        elif results_count >= scenarios_count:
            status = "✓ Complete"
        else:
            status = f"◐ {results_count}/{scenarios_count}"

        # Format: "Name  |  scenarios  |  status"
        display = f"{exp.name:<35} │ {scenarios_count} scenarios │ {status}"
        choices.append(questionary.Choice(title=display, value=i + 1))

    # Add cancel option
    choices.append(questionary.Separator("─" * 70))
    choices.append(questionary.Choice(title="⮐ Cancel", value=-1))

    # Display header
    console.print()
    console.print("[bold cyan]╔══════════════════════════════════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]                    [bold white]SPKMC Experiment Runner[/bold white]                         [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]╠══════════════════════════════════════════════════════════════════════╣[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]  [dim]Use ↑/↓ arrows to navigate, Enter to select, Ctrl+C to cancel[/dim]   [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]╚══════════════════════════════════════════════════════════════════════╝[/bold cyan]")
    console.print()

    try:
        result = questionary.select(
            "Select an experiment to run:",
            choices=choices,
            style=custom_style,
            instruction="",
            use_indicator=True,
            use_shortcuts=False,
        ).ask()

        # Handle cancel or Ctrl+C
        if result is None or result == -1:
            return None
        return result
    except KeyboardInterrupt:
        return None


def _execute_single_scenario(
    scenario: Dict[str, Any],
    scenario_index: int,
    results_dir: Path,
    experiment_name: str,
    use_simple: bool,
    create_zip: bool,
    no_plot: bool,
    save_plot: bool,
    use_gpu: bool = False,
    force_rerun: bool = False
) -> Tuple[Optional[Dict[str, Any]], str, str]:
    """
    Execute a single scenario and return the result.

    This function is designed to be called from parallel workers.

    Args:
        scenario: Scenario configuration dictionary
        scenario_index: Index of the scenario in the experiment
        results_dir: Path to results directory
        experiment_name: Name of the experiment
        use_simple: Generate simplified CSV files
        create_zip: Create zip archives
        no_plot: Disable plot generation
        save_plot: Save plots to files
        use_gpu: Use GPU acceleration if available
        force_rerun: Force re-execution even if cached results exist

    Returns:
        Tuple of (result_dict, output_file_path, scenario_label)
    """
    scenario_num = scenario_index + 1

    # Extract parameters
    network_type = scenario.get("network_type", "er")
    dist_type    = scenario.get("distribution", "exponential")
    nodes        = scenario.get("network_size", DEFAULT_N)
    k_avg        = scenario.get("k_avg", DEFAULT_K_AVG)
    shape        = scenario.get("shape", DEFAULT_SHAPE)
    scale        = scenario.get("scale", DEFAULT_SCALE)
    mu           = scenario.get("mu", DEFAULT_MU)
    lambda_val   = scenario.get("lambda", DEFAULT_LAMBDA)
    exponent     = scenario.get("exponent", DEFAULT_EXPONENT)
    samples      = scenario.get("samples", DEFAULT_SAMPLES)
    num_runs     = scenario.get("num_runs", DEFAULT_NUM_RUNS)
    initial_perc = scenario.get("initial_perc", DEFAULT_INITIAL_PERC)
    t_max        = scenario.get("t_max", DEFAULT_T_MAX)
    steps        = scenario.get("steps", DEFAULT_STEPS)

    # Create filename
    scenario_label = f"scenario_{scenario_num:03d}"
    if "label" in scenario:
        scenario_label = scenario['label']

    output_file = str(results_dir / f"{scenario_label}.json")

    # Check if already executed (skip if force_rerun is True)
    if not force_rerun and os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                existing_result = json.load(f)
            existing_metadata = existing_result.get("metadata", {})

            params_match = (
                existing_metadata.get("network_type", "").lower() == network_type.lower() and
                existing_metadata.get("distribution", "").lower() == dist_type.lower() and
                existing_metadata.get("N") == nodes and
                abs(existing_metadata.get("initial_perc", 0) - initial_perc) < 1e-6
            )

            if params_match:
                return (existing_result, output_file, scenario_label)
        except Exception:
            pass

    # Create distribution
    distribution_params = {
        "shape": shape,
        "scale": scale,
        "mu": mu,
        "lambda": lambda_val
    }
    distribution = create_distribution(dist_type, **distribution_params)

    # Create simulator
    simulator = SPKMC(distribution, use_gpu=use_gpu)

    # Create time steps
    time_steps = np.linspace(0, t_max, steps)

    # Simulation parameters
    simulation_params = {
        "N": nodes,
        "samples": samples,
        "initial_perc": initial_perc,
        "overwrite": force_rerun  # Force overwrite when re-running
    }

    if network_type in ["er", "cn", "rrn"]:
        simulation_params["k_avg"] = k_avg
        simulation_params["num_runs"] = num_runs

    if network_type == "cn":
        simulation_params["exponent"] = exponent

    # Disable inner progress bars during batch execution
    simulation_params["show_progress"] = False

    # Execute simulation
    result = simulator.run_simulation(network_type, time_steps, **simulation_params)

    # Extract results
    S = result["S_val"]
    I = result["I_val"]
    R = result["R_val"]
    has_error = result.get("has_error", False)

    if has_error:
        S_err = result["S_err"]
        I_err = result["I_err"]
        R_err = result["R_err"]

    # Prepare output
    output_result = {
        "S_val": list(S),
        "I_val": list(I),
        "R_val": list(R),
        "time": list(time_steps),
        "metadata": {
            "network_type": network_type,
            "distribution": dist_type,
            "N": nodes,
            "initial_perc": initial_perc,
            "scenario_number": scenario_num,
            "scenario_label": scenario_label,
            "experiment_name": experiment_name
        }
    }

    # Add distribution parameters
    if dist_type == "gamma":
        output_result["metadata"]["shape"] = shape
        output_result["metadata"]["scale"] = scale
    else:
        output_result["metadata"]["mu"] = mu

    output_result["metadata"]["lambda"] = lambda_val

    if network_type in ["er", "cn", "rrn"]:
        output_result["metadata"]["k_avg"] = k_avg
        output_result["metadata"]["num_runs"] = num_runs

    if network_type == "cn":
        output_result["metadata"]["exponent"] = exponent

    if has_error:
        output_result.update({
            "S_err": list(S_err),
            "I_err": list(I_err),
            "R_err": list(R_err)
        })

    # Save result
    with open(output_file, 'w') as f:
        json.dump(output_result, f, indent=2)

    # Generate CSV if needed
    if use_simple:
        csv_path = output_file.replace(".json", "_simple.csv")
        with open(csv_path, 'w') as f:
            for j, t in enumerate(time_steps):
                erro = I_err[j] if has_error else 0.0
                f.write(f"{t},{I[j]},{erro}\n")

    # Generate individual plot if needed
    if save_plot and not no_plot:
        plot_path = output_file.replace(".json", ".png")
        title = f"{experiment_name} - {scenario_label}"
        if has_error:
            Visualizer.plot_result_with_error(S, I, R, S_err, I_err, R_err, time_steps, title, plot_path)
        else:
            Visualizer.plot_result(S, I, R, time_steps, title, plot_path)

    # Create zip if needed
    if create_zip:
        zip_path = output_file.replace(".json", ".zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(output_file, os.path.basename(output_file))
            if use_simple:
                csv_p = output_file.replace(".json", "_simple.csv")
                if os.path.exists(csv_p):
                    zipf.write(csv_p, os.path.basename(csv_p))

    return (output_result, output_file, scenario_label)


def _display_hardware_panel(hardware: HardwareInfo, strategy: ParallelizationStrategy) -> None:
    """Display hardware detection panel."""
    from rich.panel import Panel
    from rich.text import Text

    lines = []

    # CPU info
    cpu_info = f"CPU: {hardware.cpu_count} cores ({hardware.cpu_count_physical} physical)"
    if strategy.scenario_workers > 1:
        cpu_info += f" → {strategy.scenario_workers} parallel workers"
    lines.append(cpu_info)

    # GPU info
    if hardware.gpu_available and hardware.gpu_name:
        memory_str = f"{hardware.gpu_memory_mb // 1024}GB" if hardware.gpu_memory_mb and hardware.gpu_memory_mb >= 1024 else f"{hardware.gpu_memory_mb}MB"
        gpu_info = f"GPU: {hardware.gpu_name} ({memory_str}) → Dijkstra acceleration"
    else:
        gpu_info = "GPU: Not available → CPU mode"
    lines.append(gpu_info)

    # Numba info
    numba_info = f"Numba: {strategy.numba_threads} threads (OpenMP)"
    lines.append(numba_info)

    content = "\n".join(f"  {line}" for line in lines)
    panel = Panel(content, title="Hardware Detected", border_style="cyan")
    console.print(panel)
    console.print()


def run_experiment_scenarios(
    experiment: Experiment,
    verbose: bool,
    use_simple: bool,
    create_zip: bool,
    no_plot: bool,
    save_plot: bool,
    force_rerun: bool = False
) -> List[str]:
    """
    Executa todos os cenários de um experimento com paralelização automática.

    Args:
        experiment: Experimento a ser executado
        verbose: Mostrar informações detalhadas
        use_simple: Gerar arquivos CSV simplificados
        create_zip: Criar arquivos zip com os resultados
        no_plot: Desativar geração de gráficos
        save_plot: Salvar gráficos em arquivos
        force_rerun: Forçar re-execução mesmo se resultados em cache existirem

    Returns:
        Lista de caminhos dos arquivos de resultado gerados
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Detect hardware and configure parallelization
    hardware = get_hardware_info()
    strategy = ParallelizationStrategy.auto_configure(
        hardware,
        num_scenarios=len(experiment.scenarios)
    )

    # Display hardware panel
    _display_hardware_panel(hardware, strategy)

    # Ensure results directory exists
    results_dir = experiment.ensure_results_dir()

    result_files = []
    all_results = []
    all_labels = []

    start_time = time.time()
    num_scenarios = len(experiment.scenarios)

    # Determine execution mode
    use_parallel = strategy.scenario_workers > 1 and num_scenarios > 1

    if use_parallel:
        # Parallel execution
        parallel_label = f"Executando {num_scenarios} cenários [{strategy.scenario_workers}x parallel]"
    else:
        parallel_label = f"Executando {num_scenarios} cenários"

    with create_progress_bar(parallel_label, num_scenarios, verbose) as progress:
        task = progress.add_task("Processando cenários...", total=num_scenarios)

        if use_parallel:
            # Parallel execution using ProcessPoolExecutor
            futures_results: List[Optional[Tuple]] = [None] * num_scenarios

            with ProcessPoolExecutor(max_workers=strategy.scenario_workers) as executor:
                future_to_index = {}

                for i, scenario in enumerate(experiment.scenarios):
                    future = executor.submit(
                        _execute_single_scenario,
                        scenario,
                        i,
                        results_dir,
                        experiment.name,
                        use_simple,
                        create_zip,
                        no_plot,
                        save_plot,
                        strategy.use_gpu,
                        force_rerun
                    )
                    future_to_index[future] = i

                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    scenario = experiment.scenarios[index]
                    scenario_label = scenario.get('label', f'scenario_{index+1:03d}')

                    try:
                        result_tuple = future.result()
                        futures_results[index] = result_tuple
                    except Exception as e:
                        log_error(f"Erro no cenário {index+1} ({scenario_label}): {e}")
                        futures_results[index] = None

                    progress.update(task, advance=1)

            # Collect results in order
            for i, result_tuple in enumerate(futures_results):
                if result_tuple is not None:
                    output_result, output_file, scenario_label = result_tuple
                    result_files.append(output_file)
                    all_results.append(output_result)
                    all_labels.append(experiment.scenarios[i].get("label", scenario_label))

        else:
            # Sequential execution (original behavior)
            for i, scenario in enumerate(experiment.scenarios):
                scenario_label = scenario.get('label', f'scenario_{i+1:03d}')

                try:
                    result_tuple = _execute_single_scenario(
                        scenario,
                        i,
                        results_dir,
                        experiment.name,
                        use_simple,
                        create_zip,
                        no_plot,
                        save_plot,
                        strategy.use_gpu,
                        force_rerun
                    )

                    if result_tuple is not None:
                        output_result, output_file, label = result_tuple
                        result_files.append(output_file)
                        all_results.append(output_result)
                        all_labels.append(scenario.get("label", label))

                except Exception as e:
                    log_error(f"Erro ao executar cenário {i+1}: {e}")

                progress.update(task, advance=1)

    # Calculate execution time
    execution_time = time.time() - start_time

    # Generate comparison plot with custom configuration
    if len(all_results) > 1:
        compare_path = str(results_dir / "comparison.png")
        try:
            Visualizer.compare_results_with_config(all_results, all_labels, experiment.plot_config, compare_path)
            log_success(f"Gráfico comparativo salvo em: {compare_path}")
        except Exception as e:
            log_error(f"Erro ao gerar gráfico comparativo: {e}")

    # Show execution summary
    mode_str = f"{strategy.scenario_workers}x parallel" if use_parallel else "sequential"
    log_success(f"Concluído em {execution_time:.1f}s ({mode_str})")

    return result_files


def create_time_steps(t_max: float, steps: int) -> np.ndarray:
    """
    Cria o array de passos de tempo.
    
    Args:
        t_max: Tempo máximo de simulação
        steps: Número de passos de tempo
    
    Returns:
        Array com os passos de tempo
    """
    return np.linspace(0, t_max, steps)


@click.group(help="CLI para o algoritmo SPKMC (Shortest Path Kinetic Monte Carlo)")
@click.version_option(version="1.0.0")
@click.option("--verbose", "-v", is_flag=True, help="Ativar modo verboso para depuração")
@click.option("--no-color", is_flag=True, help="Desativar cores na saída")
@click.option("--simple", is_flag=True, help="Gerar arquivo de resultado simplificado em CSV (tempo, infectados, erro)")
def cli(verbose, no_color, simple):
    """Grupo principal de comandos da CLI SPKMC."""
    # Configura o modo verboso
    os.environ["SPKMC_VERBOSE"] = "1" if verbose else "0"
    
    if verbose:
        log_info("Modo verboso ativado")
        
    if no_color:
        log_info("Cores desativadas na saída")
        
    if simple:
        log_info("Modo de saída simplificada em CSV ativado")


@cli.command(help="Executar uma simulação SPKMC")
@click.option("--simple", is_flag=True, help="Gerar arquivo de resultado simplificado em CSV (tempo, infectados, erro)")
@click.option("--network-type", "-n", type=str, default="er", show_default=True,
              callback=validate_network_type,
              help="Tipo de rede: Erdos-Renyi (er), Complex Network (cn), Complete Graph (cg), Random Regular Network (rrn)")
@click.option("--dist-type", "-d", type=str, default="gamma", show_default=True,
              callback=validate_distribution_type,
              help="Tipo de distribuição: Gamma ou Exponential")
@click.option("--shape", type=float, default=DEFAULT_SHAPE, show_default=True,
              callback=validate_positive,
              help="Parâmetro de forma para distribuição Gamma")
@click.option("--scale", type=float, default=DEFAULT_SCALE, show_default=True,
              callback=validate_positive,
              help="Parâmetro de escala para distribuição Gamma")
@click.option("--mu", type=float, default=DEFAULT_MU, show_default=True,
              callback=validate_positive,
              help="Parâmetro mu para distribuição Exponencial (recuperação)")
@click.option("--lambda", "lambda_val", type=float, default=DEFAULT_LAMBDA, show_default=True,
              callback=validate_positive,
              help="Parâmetro lambda para tempos de infecção")
@click.option("--exponent", type=float, default=DEFAULT_EXPONENT, show_default=True,
              callback=validate_exponent,
              help="Expoente para redes complexas (CN)")
@click.option("--nodes", "-N", type=int, default=DEFAULT_N, show_default=True,
              callback=validate_positive_int,
              help="Número de nós na rede")
@click.option("--k-avg", type=float, default=DEFAULT_K_AVG, show_default=True,
              callback=validate_positive,
              help="Grau médio da rede")
@click.option("--samples", "-s", type=int, default=DEFAULT_SAMPLES, show_default=True,
              callback=validate_positive_int,
              help="Número de amostras por execução")
@click.option("--num-runs", "-r", type=int, default=DEFAULT_NUM_RUNS, show_default=True,
              callback=validate_positive_int,
              help="Número de execuções (para médias)")
@click.option("--initial-perc", "-i", type=float, default=DEFAULT_INITIAL_PERC, show_default=True,
              callback=validate_percentage,
              help="Porcentagem inicial de infectados")
@click.option("--t-max", type=float, default=DEFAULT_T_MAX, show_default=True,
              callback=validate_positive,
              help="Tempo máximo de simulação")
@click.option("--steps", type=int, default=DEFAULT_STEPS, show_default=True,
              callback=validate_positive_int,
              help="Número de passos de tempo")
@click.option("--output", "-o", type=str,
              callback=validate_output_file,
              help="Caminho para salvar os resultados (opcional)")
@click.option("--export-format", "-e", type=click.Choice(['json', 'csv', 'excel', 'md', 'html']),
              help="Formato para exportação dos resultados (opcional)")
@click.option("--no-plot", is_flag=True, default=False,
              help="Não exibir o gráfico dos resultados")
@click.option("--save-plot", type=str,
              help="Salvar o gráfico em um arquivo (formato: png, pdf, svg)")
@click.option("--overwrite", is_flag=True, default=False,
              help="Sobrescrever resultados existentes")
@click.option("--zip", is_flag=True, default=False,
              help="Criar um arquivo zip com os resultados")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Mostrar informações detalhadas durante a simulação")
def run(simple, network_type, dist_type, shape, scale, mu, lambda_val, exponent, nodes, k_avg,
        samples, num_runs, initial_perc, t_max, steps, output, export_format, no_plot,
        save_plot, overwrite, zip, verbose):
    """Executa uma simulação SPKMC com os parâmetros especificados."""
    # Configurar o modo verboso
    if verbose:
        os.environ["SPKMC_VERBOSE"] = "1"
    
    # Registrar início da simulação
    start_time = time.time()
    log_debug(f"Iniciando simulação em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", verbose_only=False)
    
    # Criar a distribuição
    distribution_params = {
        "shape": shape,
        "scale": scale,
        "mu": mu,
        "lambda": lambda_val
    }
    distribution = create_distribution(dist_type, **distribution_params)
    log_debug(f"Distribuição {dist_type.capitalize()} criada com parâmetros: {distribution_params}", verbose_only=True)
    
    # Criar o simulador
    simulator = SPKMC(distribution)
    
    # Criar os passos de tempo
    time_steps = create_time_steps(t_max, steps)
    log_debug(f"Passos de tempo criados: {time_steps[0]} a {time_steps[-1]} ({len(time_steps)} pontos)", verbose_only=True)
    
    # Exibir informações da simulação
    console.print(format_title(f"Simulação SPKMC"))
    console.print(format_info('Configuração:'))
    
    # Usar valores diretamente sem aninhamento de funções de formatação
    network_type_upper = network_type.upper()
    dist_type_cap = dist_type.capitalize()
    
    console.print(f"  {format_param('Rede', network_type_upper)}")
    console.print(f"  {format_param('Distribuição', dist_type_cap)}")
    console.print(f"  {format_param('Nós', nodes)}")
    
    if network_type in ["er", "cn", "rrn"]:
        console.print(f"  {format_param('Grau médio', k_avg)}")
    
    if network_type == "cn":
        console.print(f"  {format_param('Expoente', exponent)}")
    
    console.print(f"  {format_param('Amostras', samples)}")
    console.print(f"  {format_param('Execuções', num_runs)}")
    console.print(f"  {format_param('Infectados iniciais', f'{initial_perc*100:.2f}%')}")
    console.print(f"  {format_param('Tempo máximo', t_max)}")
    console.print(f"  {format_param('Passos', steps)}")
    
    # Iniciar barra de progresso
    with create_progress_bar("Simulação SPKMC", 1, verbose) as progress:
        task = progress.add_task("Executando simulação...", total=1)
    
    # Parâmetros específicos para cada tipo de rede
    simulation_params = {
        "N": nodes,
        "samples": samples,
        "initial_perc": initial_perc,
        "overwrite": overwrite
    }
    
    if network_type in ["er", "cn", "rrn"]:
        simulation_params["k_avg"] = k_avg
        simulation_params["num_runs"] = num_runs
    
    if network_type == "cn":
        simulation_params["exponent"] = exponent
    
    # Executar a simulação
    try:
        log_debug("Iniciando execução da simulação", verbose_only=True)
        result = simulator.run_simulation(network_type, time_steps, **simulation_params)
        progress.update(task, advance=1)
        log_success("Simulação concluída com sucesso!")
    except Exception as e:
        log_error(f"Erro durante a simulação: {e}")
        return
    
    # Extrair os resultados
    S = result["S_val"]
    I = result["I_val"]
    R = result["R_val"]
    has_error = result.get("has_error", False)
    
    if has_error:
        S_err = result["S_err"]
        I_err = result["I_err"]
        R_err = result["R_err"]
    
    # Calcular estatísticas
    max_infected = np.max(I)
    max_infected_time = time_steps[np.argmax(I)]
    final_recovered = R[-1]
    
    # Exibir estatísticas
    console.print(format_title("Estatísticas da Simulação"))
    console.print(f"  {format_param('Máximo de infectados', f'{max_infected:.4f} (em t={max_infected_time:.2f})')}")
    console.print(f"  {format_param('Recuperados finais', f'{final_recovered:.4f}')}")
    
    # Registrar tempo de execução
    end_time = time.time()
    execution_time = end_time - start_time
    log_debug(f"Tempo de execução: {execution_time:.2f} segundos", verbose_only=False)
    
    # Salvar resultados em um arquivo personalizado, se especificado
    if output:
        output_result = {
            "S_val": list(S),
            "I_val": list(I),
            "R_val": list(R),
            "time": list(time_steps),
            "metadata": {
                "network_type": network_type,
                "distribution": dist_type,
                "N": nodes,
                "initial_perc": initial_perc,
                "execution_time": execution_time
            }
        }
        
        # Adicionar parâmetros específicos
        if dist_type == "gamma":
            output_result["metadata"]["shape"] = shape
            output_result["metadata"]["scale"] = scale
        else:
            output_result["metadata"]["mu"] = mu
        
        output_result["metadata"]["lambda"] = lambda_val
        
        if network_type in ["er", "cn", "rrn"]:
            output_result["metadata"]["k_avg"] = k_avg
            output_result["metadata"]["num_runs"] = num_runs
        
        if network_type == "cn":
            output_result["metadata"]["exponent"] = exponent
        
        if has_error:
            output_result.update({
                "S_err": list(S_err),
                "I_err": list(I_err),
                "R_err": list(R_err)
            })
        
        try:
            # Garantir que o diretório exista
            os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)
            
            with open(output, 'w') as f:
                json.dump(output_result, f, indent=2)
            log_success(f"Resultados salvos em: {output}")
            
            # Verificar se o parâmetro --simple foi passado globalmente ou localmente
            ctx = click.get_current_context()
            use_simple = simple or ctx.parent.params.get('simple', False)
            
            # Gerar arquivo CSV simplificado se o parâmetro --simple estiver ativado
            if use_simple:
                csv_path = output.replace(".json", "_simple.csv")
                with open(csv_path, 'w') as f:
                    # Dados sem cabeçalho (tempo, infectados, erro)
                    for i, t in enumerate(time_steps):
                        erro = I_err[i] if has_error else 0.0
                        f.write(f"{t},{I[i]},{erro}\n")
                
                log_success(f"Resultados simplificados salvos em CSV: {csv_path}")
            
            # Exportar em formato adicional, se especificado
            if export_format:
                export_path = output.replace(".json", f".{export_format}")
                exported_file = ExportManager.export_results(output_result, export_path, export_format)
                log_success(f"Resultados exportados em formato {export_format.upper()}: {exported_file}")
            
            # Criar arquivo zip com os resultados, se solicitado
            if zip:
                zip_path = output.replace(".json", ".zip")
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Adicionar o arquivo JSON principal
                    zipf.write(output, os.path.basename(output))
                    
                    # Adicionar o arquivo CSV simplificado, se existir
                    if use_simple and os.path.exists(csv_path):
                        zipf.write(csv_path, os.path.basename(csv_path))
                    
                    # Adicionar o arquivo exportado, se existir
                    if export_format and os.path.exists(export_path):
                        zipf.write(export_path, os.path.basename(export_path))
                    
                    # Adicionar o gráfico, se existir
                    if save_plot and os.path.exists(save_plot):
                        zipf.write(save_plot, os.path.basename(save_plot))
                
                log_success(f"Resultados compactados em: {zip_path}")
        except Exception as e:
            log_error(f"Erro ao salvar resultados: {e}")
    
    # Plotar os resultados, se solicitado
    if not no_plot or save_plot:
        log_info("Gerando visualização...")
        
        title = f"Simulação SPKMC - Rede {network_type.upper()}, Distribuição {dist_type.capitalize()}"
        
        try:
            if save_plot:
                # Salvar o gráfico em um arquivo
                if has_error:
                    Visualizer.plot_result_with_error(S, I, R, S_err, I_err, R_err, time_steps, title, save_plot)
                else:
                    Visualizer.plot_result(S, I, R, time_steps, title, save_plot)
                log_success(f"Gráfico salvo em: {save_plot}")
            elif not no_plot:
                # Mostrar o gráfico na tela
                if has_error:
                    Visualizer.plot_result_with_error(S, I, R, S_err, I_err, R_err, time_steps, title)
                else:
                    Visualizer.plot_result(S, I, R, time_steps, title)
        except Exception as e:
            log_error(f"Erro ao gerar visualização: {e}")


@cli.command(help="Visualizar resultados de simulações anteriores")
@click.option("--simple", is_flag=True, help="Gerar arquivo de resultado simplificado em CSV (tempo, infectados, erro)")
@click.argument("path", type=str)
@click.option("--with-error", "-e", is_flag=True, default=False,
              help="Mostrar barras de erro (se disponíveis)")
@click.option("--output", "-o", type=str,
              callback=validate_output_file,
              help="Salvar o gráfico em um arquivo (opcional)")
@click.option("--format", "-f", type=click.Choice(['png', 'pdf', 'svg', 'jpg']), default='png',
              help="Formato do gráfico (quando usado com --output)")
@click.option("--dpi", type=int, default=300,
              help="Resolução do gráfico em DPI (quando usado com --output)")
@click.option("--export", "-x", type=click.Choice(['json', 'csv', 'excel', 'md', 'html']),
              help="Exportar resultados em formato adicional")
@click.option("--states", "-s", type=str, multiple=True,
              help="Estados específicos para plotar (ex: --states infected --states recovered)")
@click.option("--separate", is_flag=True, default=False,
              help="Plotar múltiplos cenários em gráficos separados (quando um diretório é fornecido)")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Mostrar informações detalhadas")
def plot(simple, path, with_error, output, format, dpi, export, states, separate, verbose):
    # Verificar se o parâmetro --simple foi passado globalmente ou localmente
    ctx = click.get_current_context()
    use_simple = simple or ctx.parent.params.get('simple', False)
    """Visualiza os resultados de uma simulação anterior."""
    # Configurar o modo verboso
    if verbose:
        os.environ["SPKMC_VERBOSE"] = "1"
    
    # Verificar se o caminho é um arquivo ou diretório
    path_obj = Path(path)
    
    if not path_obj.exists():
        log_error(f"Caminho não encontrado: {path}")
        ctx.exit(1)
    
    # Processar estados a serem plotados
    valid_states = {'susceptible', 'infected', 'recovered', 's', 'i', 'r'}
    states_to_plot = set()
    
    if states:
        for state in states:
            state_lower = state.lower()
            if state_lower in valid_states:
                # Normalizar para S, I, R
                if state_lower in ['susceptible', 's']:
                    states_to_plot.add('S')
                elif state_lower in ['infected', 'i']:
                    states_to_plot.add('I')
                elif state_lower in ['recovered', 'r']:
                    states_to_plot.add('R')
            else:
                log_warning(f"Estado inválido ignorado: {state}")
    else:
        # Se nenhum estado foi especificado, plotar todos
        states_to_plot = {'S', 'I', 'R'}
    
    if not states_to_plot:
        log_warning("Nenhum estado válido especificado. Plotando todos os estados.")
        states_to_plot = {'S', 'I', 'R'}
    
    log_info(f"Estados a serem plotados: {', '.join(sorted(states_to_plot))}")
    
    # Coletar arquivos de resultados
    result_files = []
    
    if path_obj.is_file():
        # Arquivo único
        if path_obj.suffix == '.json':
            result_files.append(path_obj)
        else:
            log_error(f"Arquivo deve ser JSON: {path}")
            ctx.exit(1)
    elif path_obj.is_dir():
        # Diretório - buscar todos os arquivos JSON
        json_files = list(path_obj.glob("*.json"))
        if not json_files:
            log_error(f"Nenhum arquivo JSON encontrado no diretório: {path}")
            ctx.exit(1)
        result_files.extend(json_files)
        log_info(f"Encontrados {len(result_files)} arquivos JSON no diretório")
    else:
        log_error(f"Caminho inválido: {path}")
        ctx.exit(1)
    
    # Processar arquivos
    if len(result_files) == 1:
        # Arquivo único - comportamento original
        result_file = result_files[0]
        try:
            log_info(f"Carregando resultados de: {result_file}")
            result = ResultManager.load_result(str(result_file))
        except Exception as e:
            log_error(f"Erro ao carregar o arquivo de resultados: {e}")
            ctx.exit(1)
    
        # Extrair os dados
        S = np.array(result.get("S_val", []))
        I = np.array(result.get("I_val", []))
        R = np.array(result.get("R_val", []))
        time_steps = np.array(result.get("time", []))
        
        if not len(S) or not len(I) or not len(R) or not len(time_steps):
            log_error("Dados incompletos no arquivo de resultados.")
            ctx.exit(1)
        
        # Calcular estatísticas básicas
        max_infected = np.max(I)
        max_infected_time = time_steps[np.argmax(I)]
        final_recovered = R[-1]
        
        console.print(format_title("Estatísticas da Simulação"))
        console.print(f"  {format_param('Máximo de infectados', f'{max_infected:.4f} (em t={max_infected_time:.2f})')}")
        console.print(f"  {format_param('Recuperados finais', f'{final_recovered:.4f}')}")
        
        # Verificar se há dados de erro disponíveis
        has_error = "S_err" in result and "I_err" in result and "R_err" in result
        
        # Extrair metadados para o título
        metadata = result.get("metadata", {})
        network_type = metadata.get("network_type", "").upper()
        dist_type = metadata.get("distribution", "").capitalize()
        N = metadata.get("N", "")
        
        # Exibir metadados
        console.print(format_title("Metadados da Simulação"))
        for key, value in metadata.items():
            console.print(f"  {format_param(key, value)}")
        
        title = f"Simulação SPKMC - Rede {network_type}, Distribuição {dist_type}, N={N}"
        
        try:
            log_info("Gerando visualização...")
            
            if with_error and has_error:
                S_err = np.array(result.get("S_err", []))
                I_err = np.array(result.get("I_err", []))
                R_err = np.array(result.get("R_err", []))
                Visualizer.plot_result_with_error(S, I, R, S_err, I_err, R_err, time_steps, title, output, states_to_plot)
            else:
                if with_error and not has_error:
                    log_warning("Dados de erro não disponíveis. Mostrando gráfico sem barras de erro.")
                Visualizer.plot_result(S, I, R, time_steps, title, output, states_to_plot)
            
            if output:
                log_success(f"Gráfico salvo em: {output}")
                
            # Exportar em formato adicional, se especificado
            if export:
                export_path = str(result_file).replace(".json", f".{export}")
                exported_file = ExportManager.export_results(result, export_path, export)
                log_success(f"Resultados exportados em formato {export.upper()}: {exported_file}")
            
            # Gerar arquivo CSV simplificado se o parâmetro --simple estiver ativado
            if use_simple:
                csv_path = str(result_file).replace(".json", "_simple.csv")
                with open(csv_path, 'w') as f:
                    # Dados sem cabeçalho (tempo, infectados, erro)
                    for i, t in enumerate(time_steps):
                        erro = I_err[i] if has_error else 0.0
                        f.write(f"{t},{I[i]},{erro}\n")
                
                log_success(f"Resultados simplificados salvos em CSV: {csv_path}")
                
        except Exception as e:
            log_error(f"Erro ao gerar visualização: {e}")
    
    else:
        # Múltiplos arquivos
        log_info(f"Processando {len(result_files)} arquivos de resultados...")
        
        if separate:
            # Plotar cada arquivo separadamente
            for i, result_file in enumerate(result_files):
                log_info(f"Processando arquivo {i+1}/{len(result_files)}: {result_file.name}")
                
                try:
                    result = ResultManager.load_result(str(result_file))
                    
                    # Extrair os dados
                    S = np.array(result.get("S_val", []))
                    I = np.array(result.get("I_val", []))
                    R = np.array(result.get("R_val", []))
                    time_steps = np.array(result.get("time", []))
                    
                    if not len(S) or not len(I) or not len(R) or not len(time_steps):
                        log_warning(f"Dados incompletos em {result_file.name}, pulando...")
                        continue
                    
                    # Verificar se há dados de erro
                    has_error = "S_err" in result and "I_err" in result and "R_err" in result
                    
                    # Extrair metadados
                    metadata = result.get("metadata", {})
                    network_type = metadata.get("network_type", "").upper()
                    dist_type = metadata.get("distribution", "").capitalize()
                    N = metadata.get("N", "")
                    
                    title = f"Simulação SPKMC - {result_file.stem}"
                    
                    # Determinar nome do arquivo de saída
                    if output:
                        base_output = Path(output)
                        output_file = base_output.parent / f"{base_output.stem}_{result_file.stem}{base_output.suffix}"
                    else:
                        output_file = None
                    
                    if with_error and has_error:
                        S_err = np.array(result.get("S_err", []))
                        I_err = np.array(result.get("I_err", []))
                        R_err = np.array(result.get("R_err", []))
                        Visualizer.plot_result_with_error(S, I, R, S_err, I_err, R_err, time_steps, title, str(output_file) if output_file else None, states_to_plot)
                    else:
                        Visualizer.plot_result(S, I, R, time_steps, title, str(output_file) if output_file else None, states_to_plot)
                    
                    if output_file:
                        log_success(f"Gráfico salvo em: {output_file}")
                        
                except Exception as e:
                    log_error(f"Erro ao processar {result_file.name}: {e}")
                    continue
        
        else:
            # Plotar todos os arquivos em um único gráfico comparativo
            results_data = []
            labels = []
            
            for result_file in result_files:
                try:
                    result = ResultManager.load_result(str(result_file))
                    
                    # Verificar se tem os dados necessários
                    if all(key in result for key in ["S_val", "I_val", "R_val", "time"]):
                        results_data.append(result)
                        labels.append(result_file.stem)
                    else:
                        log_warning(f"Dados incompletos em {result_file.name}, pulando...")
                        
                except Exception as e:
                    log_error(f"Erro ao carregar {result_file.name}: {e}")
                    continue
            
            if not results_data:
                log_error("Nenhum arquivo válido encontrado para plotar.")
                ctx.exit(1)
            
            try:
                log_info(f"Gerando visualização comparativa de {len(results_data)} cenários...")
                
                title = f"Comparação de Simulações SPKMC - {path_obj.name}"
                Visualizer.compare_results(results_data, labels, title, output, states_to_plot)
                
                if output:
                    log_success(f"Gráfico comparativo salvo em: {output}")
                    
            except Exception as e:
                log_error(f"Erro ao gerar visualização comparativa: {e}")


@cli.command(help="Mostrar informações sobre simulações salvas")
@click.option("--simple", is_flag=True, help="Gerar arquivo de resultado simplificado em CSV (tempo, infectados, erro)")
@click.option("--result-file", "-f", type=str, callback=validate_file_exists,
              help="Arquivo de resultados específico (opcional)")
@click.option("--list", "-l", "list_files", is_flag=True, default=False,
              help="Listar todos os arquivos de resultados disponíveis")
@click.option("--export", "-e", type=click.Choice(['json', 'csv', 'excel', 'md', 'html']),
              help="Exportar informações em formato específico")
@click.option("--output", "-o", type=str,
              callback=validate_output_file,
              help="Caminho para salvar a exportação (usado com --export)")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Mostrar informações detalhadas")
def info(simple, result_file, list_files, export, output, verbose):
    # Verificar se o parâmetro --simple foi passado globalmente ou localmente
    ctx = click.get_current_context()
    use_simple = simple or ctx.parent.params.get('simple', False)
    """Mostra informações sobre simulações salvas."""
    # Configurar o modo verboso
    if verbose:
        os.environ["SPKMC_VERBOSE"] = "1"
    
    if list_files:
        files = ResultManager.list_results()
        if not files:
            log_info("Nenhum arquivo de resultados encontrado.")
            return
        
        console.print(format_title("Arquivos de Resultados Disponíveis"))
        
        # Criar uma tabela formatada
        data = []
        for i, file in enumerate(files, 1):
            # Extrair metadados do caminho
            metadata = ResultManager.get_metadata_from_path(file)
            network = metadata.get("network_type", "").upper() if metadata else ""
            dist = metadata.get("distribution", "").capitalize() if metadata else ""
            nodes = metadata.get("N", "") if metadata else ""
            
            data.append({
                "Índice": i,
                "Arquivo": os.path.basename(file),
                "Rede": network,
                "Distribuição": dist,
                "Nós": nodes
            })
        
        print_rich_table(data, "Resultados Disponíveis")
        return
    
    if not result_file:
        log_error("Especifique um arquivo de resultados ou use --list para ver os disponíveis.")
        return
    
    # Carregar os resultados
    try:
        log_info(f"Carregando resultados de: {result_file}")
        result = ResultManager.load_result(result_file)
    except Exception as e:
        log_error(f"Erro ao carregar o arquivo de resultados: {e}")
        return
    
    # Gerar arquivo CSV simplificado se o parâmetro --simple estiver ativado e um arquivo de resultados foi especificado
    if use_simple and result_file:
        try:
            # Extrair os dados necessários
            time_steps = np.array(result.get("time", []))
            I = np.array(result.get("I_val", []))
            has_error = "I_err" in result
            I_err = np.array(result.get("I_err", [])) if has_error else None
            
            csv_path = result_file.replace(".json", "_simple.csv")
            with open(csv_path, 'w') as f:
                # Dados sem cabeçalho (tempo, infectados, erro)
                for i, t in enumerate(time_steps):
                    erro = I_err[i] if has_error else 0.0
                    f.write(f"{t},{I[i]},{erro}\n")
            
            log_success(f"Resultados simplificados salvos em CSV: {csv_path}")
        except Exception as e:
            log_error(f"Erro ao gerar arquivo CSV simplificado: {e}")
    
    # Extrair e mostrar informações
    metadata = result.get("metadata", {})
    if metadata:
        console.print(format_title("Parâmetros da Simulação"))
        for key, value in metadata.items():
            if isinstance(value, dict):
                console.print(f"  {format_param(key, '')}")
                for subkey, subvalue in value.items():
                    console.print(f"    {format_param(subkey, subvalue)}")
            else:
                console.print(f"  {format_param(key, value)}")
    else:
        # Tentar inferir informações do arquivo
        console.print(format_title("Informações do Arquivo"))
        console.print(f"  {format_param('Arquivo', os.path.basename(result_file))}")
        console.print(f"  {format_param('Tamanho', f'{os.path.getsize(result_file)/1024:.2f} KB')}")
        
        # Formatar a data de modificação
        mod_time = os.path.getmtime(result_file)
        mod_time_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
        console.print(f"  {format_param('Última modificação', mod_time_str)}")
        
        # Extrair metadados do caminho
        path_metadata = ResultManager.get_metadata_from_path(result_file)
        if path_metadata:
            console.print(format_title("Metadados Inferidos do Caminho"))
            for key, value in path_metadata.items():
                console.print(f"  {format_param(key, value)}")
        
        # Mostrar estatísticas básicas
        if "S_val" in result and "I_val" in result and "R_val" in result:
            S = np.array(result.get("S_val", []))
            I = np.array(result.get("I_val", []))
            R = np.array(result.get("R_val", []))
            time_steps = np.array(result.get("time", []))
            
            console.print(format_title("Estatísticas"))
            console.print(f"  {format_param('Pontos de tempo', len(S))}")
            
            max_infected = np.max(I)
            max_infected_time = time_steps[np.argmax(I)]
            console.print(f"  {format_param('Máximo de infectados', f'{max_infected:.4f} (em t={max_infected_time:.2f})')}")
            console.print(f"  {format_param('Recuperados finais', f'{R[-1]:.4f}')}")
    
    # Verificar se há dados de erro
    has_error = "S_err" in result and "I_err" in result and "R_err" in result
    if has_error:
        console.print(format_info("O arquivo contém dados de erro (use --with-error ao plotar)."))
    
    # Exportar em formato adicional, se especificado
    if export:
        if not output:
            output = result_file.replace(".json", f".{export}")
        
        exported_file = ExportManager.export_results(result, output, export)
        log_success(f"Informações exportadas em formato {export.upper()}: {exported_file}")


@cli.command(help="Comparar resultados de múltiplas simulações")
@click.option("--simple", is_flag=True, help="Gerar arquivo de resultado simplificado em CSV (tempo, infectados, erro)")
@click.argument("result_files", nargs=-1, type=str, required=True)
@click.option("--labels", "-l", multiple=True,
              help="Rótulos para cada arquivo (opcional)")
@click.option("--output", "-o", type=str,
              callback=validate_output_file,
              help="Salvar o gráfico em um arquivo (opcional)")
@click.option("--format", "-f", type=click.Choice(['png', 'pdf', 'svg', 'jpg']), default='png',
              help="Formato do gráfico (quando usado com --output)")
@click.option("--dpi", type=int, default=300,
              help="Resolução do gráfico em DPI (quando usado com --output)")
@click.option("--export", "-e", type=click.Choice(['json', 'csv', 'excel', 'md', 'html']),
              help="Exportar resultados comparativos em formato adicional")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Mostrar informações detalhadas")
def compare(simple, result_files, labels, output, format, dpi, export, verbose):
    # Verificar se o parâmetro --simple foi passado globalmente ou localmente
    ctx = click.get_current_context()
    use_simple = simple or ctx.parent.params.get('simple', False)
    """Compara os resultados de múltiplas simulações."""
    # Configurar o modo verboso
    if verbose:
        os.environ["SPKMC_VERBOSE"] = "1"
    
    if not result_files:
        log_error("Especifique pelo menos um arquivo de resultados ou diretório.")
        return
    
    # Processar argumentos - expandir diretórios para arquivos JSON
    expanded_files = []
    for arg in result_files:
        if not os.path.exists(arg):
            log_error(f"O caminho '{arg}' não existe.")
            return
        
        if os.path.isdir(arg):
            # Se for um diretório, buscar todos os arquivos JSON
            json_files = sorted([str(f) for f in Path(arg).glob("*.json")])
            if not json_files:
                log_error(f"Nenhum arquivo JSON encontrado no diretório: {arg}")
                return
            log_info(f"Encontrados {len(json_files)} arquivos JSON em {arg}")
            expanded_files.extend(json_files)
        elif os.path.isfile(arg):
            # Se for um arquivo, verificar se é JSON
            if not arg.endswith('.json'):
                log_warning(f"O arquivo '{arg}' não é um arquivo JSON, ignorando...")
                continue
            expanded_files.append(arg)
        else:
            log_error(f"O caminho '{arg}' não é um arquivo nem diretório válido.")
            return
    
    if not expanded_files:
        log_error("Nenhum arquivo JSON válido encontrado para comparar.")
        return
    
    # Atualizar result_files com a lista expandida
    result_files = expanded_files
    
    log_info(f"Comparando {len(result_files)} arquivo(s) de resultados...")
    
    # Usar nomes de arquivos como rótulos padrão se não forem fornecidos
    if not labels:
        labels = [os.path.basename(f) for f in result_files]
    elif len(labels) < len(result_files):
        # Completar com nomes de arquivos se não houver rótulos suficientes
        labels = list(labels) + [os.path.basename(f) for f in result_files[len(labels):]]
    
    # Carregar resultados
    results = []
    metadata_list = []
    
    with create_progress_bar("Carregando resultados", len(result_files), verbose) as progress:
        task = progress.add_task("Carregando arquivos...", total=len(result_files))
        
        for i, file in enumerate(result_files):
            try:
                result = ResultManager.load_result(file)
                results.append(result)
                
                # Extrair metadados para exibição
                metadata = result.get("metadata", {})
                metadata_list.append({
                    "Arquivo": os.path.basename(file),
                    "Rede": metadata.get("network_type", "").upper(),
                    "Distribuição": metadata.get("distribution", "").capitalize(),
                    "Nós": metadata.get("N", ""),
                    "Rótulo": labels[i]
                })
                
                progress.update(task, advance=1)
            except Exception as e:
                log_error(f"Erro ao carregar {file}: {e}")
                return
    
    # Exibir informações sobre os arquivos carregados
    console.print(format_title("Arquivos para Comparação"))
    print_rich_table(metadata_list, "Resultados a Comparar")
    
    # Comparar resultados
    try:
        log_info("Gerando visualização comparativa...")
        
        # Configurar o título com mais informações
        title = "Comparação de Simulações SPKMC"
        if len(result_files) <= 3:  # Adicionar detalhes apenas se houver poucos arquivos
            title += f" ({', '.join(labels)})"
        
        Visualizer.compare_results(results, labels, title, output)
        
        if output:
            log_success(f"Gráfico comparativo salvo em: {output}")
            
        # Exportar em formato adicional, se especificado
        if export:
            export_path = output.replace(f".{format}" if output else "", f".{export}")
            
            # Criar um resultado combinado para exportação
            combined_result = {
                "results": results,
                "labels": labels,
                "metadata": {
                    "comparison_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "files": list(result_files)
                }
            }
            exported_file = ExportManager.export_results(combined_result, export_path, export)
            log_success(f"Resultados comparativos exportados em formato {export.upper()}: {exported_file}")
        
        # Gerar arquivo CSV simplificado para cada resultado se o parâmetro --simple estiver ativado
        if use_simple:
            for i, result_file in enumerate(result_files):
                try:
                    result = results[i]
                    time_steps = np.array(result.get("time", []))
                    I = np.array(result.get("I_val", []))
                    has_error = "I_err" in result
                    I_err = np.array(result.get("I_err", [])) if has_error else None
                    
                    csv_path = result_file.replace(".json", "_simple.csv")
                    with open(csv_path, 'w') as f:
                        # Dados sem cabeçalho (tempo, infectados, erro)
                        for j, t in enumerate(time_steps):
                            erro = I_err[j] if has_error else 0.0
                            f.write(f"{t},{I[j]},{erro}\n")
                    
                    log_success(f"Resultados simplificados para '{labels[i]}' salvos em CSV: {csv_path}")
                except Exception as e:
                    log_error(f"Erro ao gerar arquivo CSV simplificado para '{labels[i]}': {e}")
            
            
    except Exception as e:
        log_error(f"Erro ao gerar visualização comparativa: {e}")


@cli.command(help="Executar múltiplos cenários de simulação a partir de um arquivo JSON ou experimento")
@click.option("--simple", is_flag=True, help="Gerar arquivo de resultado simplificado em CSV (tempo, infectados, erro)")
@click.argument("scenarios_file", type=str, default=None, required=False)
@click.option("--experiments-dir", "-e", type=str, default=None,
              help="Diretório base para experimentos (padrão: experiments)")
@click.option("--output-dir", "-o", type=str, default="./results",
              help="Diretório para salvar os resultados (padrão: ./results)")
@click.option("--prefix", "-p", type=str, default="",
              help="Prefixo para os nomes dos arquivos de saída")
@click.option("--compare", "-c", is_flag=True, default=False,
              help="Gerar visualização comparativa dos resultados")
@click.option("--no-plot", is_flag=True, default=False,
              help="Desativar a geração de gráficos individuais")
@click.option("--save-plot", is_flag=True, default=False,
              help="Salvar os gráficos em arquivos")
@click.option("--zip", is_flag=True, default=False,
              help="Criar um arquivo zip com os resultados de cada cenário")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Mostrar informações detalhadas durante a execução")
def batch(simple, scenarios_file, experiments_dir, output_dir, prefix, compare, no_plot, save_plot, zip, verbose):
    """
    Executa múltiplos cenários de simulação a partir de um arquivo JSON ou experimento.

    Se nenhum arquivo for especificado, exibe um menu interativo com os experimentos
    disponíveis no diretório 'experiments/' (ou no diretório especificado com --experiments-dir).

    O arquivo JSON deve conter uma lista de objetos, cada um representando um cenário
    com parâmetros para a simulação. Cada cenário será executado sequencialmente e
    os resultados serão salvos em arquivos separados no diretório especificado.
    """
    # Verificar se o parâmetro --simple foi passado globalmente ou localmente
    ctx = click.get_current_context()
    use_simple = simple or ctx.parent.params.get('simple', False)

    # Expandir ~ para caminho absoluto no output_dir
    output_dir = os.path.expanduser(output_dir)

    # Configurar o modo verboso
    if verbose:
        os.environ["SPKMC_VERBOSE"] = "1"

    # ============================================================
    # EXPERIMENT MODE: Se nenhum arquivo foi especificado
    # ============================================================
    if scenarios_file is None:
        # Criar gerenciador de experimentos
        exp_manager = ExperimentManager(experiments_dir)
        experiments = exp_manager.list_experiments()

        if not experiments:
            log_error("Nenhum experimento encontrado no diretório 'experiments/'.")
            log_info("Crie um experimento em experiments/<nome>/data.json ou especifique um arquivo de cenários.")
            return

        # Exibir menu de experimentos
        selected = display_experiments_menu(experiments)

        if selected is None:
            log_info("Operação cancelada pelo usuário.")
            return

        experiment = experiments[selected - 1]
        log_info(f"Experimento selecionado: {experiment.name}")

        # Verificar se há resultados existentes
        force_rerun = False
        if experiment.has_results:
            log_warning(f"O experimento '{experiment.name}' já possui {experiment.result_count} resultado(s).")
            if click.confirm("Deseja limpar os resultados existentes e re-executar?", default=False):
                experiment.clean_results()
                force_rerun = True  # Force re-execution of all scenarios
                log_success("Resultados anteriores removidos. Forçando re-execução completa.")
            else:
                log_info("Mantendo resultados existentes. Cenários já executados serão ignorados.")

        # Executar o experimento
        start_time = time.time()
        log_debug(f"Iniciando execução do experimento em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", verbose_only=False)

        result_files = run_experiment_scenarios(
            experiment=experiment,
            verbose=verbose,
            use_simple=use_simple,
            create_zip=zip,
            no_plot=no_plot,
            save_plot=save_plot,
            force_rerun=force_rerun
        )

        # Resumo final
        end_time = time.time()
        total_execution_time = end_time - start_time

        console.print(format_title("Resumo da Execução do Experimento"))
        console.print(f"  {format_param('Experimento', experiment.name)}")
        console.print(f"  {format_param('Total de cenários', len(experiment.scenarios))}")
        console.print(f"  {format_param('Cenários processados', len(result_files))}")
        console.print(f"  {format_param('Tempo total de execução', f'{total_execution_time:.2f} segundos')}")
        console.print(f"  {format_param('Diretório de resultados', str(experiment.results_dir))}")

        log_success(f"Experimento concluído. {len(result_files)} cenário(s) processado(s).")
        return

    # ============================================================
    # FILE MODE: Execução tradicional com arquivo de cenários
    # ============================================================

    # Registrar início da execução em lote
    start_time = time.time()
    log_debug(f"Iniciando execução em lote em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", verbose_only=False)

    # Verificar se o arquivo de cenários existe
    if not os.path.exists(scenarios_file):
        log_error(f"O arquivo de cenários '{scenarios_file}' não existe.")
        return
    
    # Carregar o arquivo de cenários
    from pathlib import Path

    try:
        log_info(f"Carregando cenários de: {scenarios_file}")
        with open(scenarios_file, 'r') as f:
            data = json.load(f)

        # Support both formats:
        # 1. Direct list of scenarios: [{...}, {...}]
        # 2. Experiment format: {"name": "...", "scenarios": [...], "plot": {...}}
        if isinstance(data, list):
            scenarios = data
            experiment_name = os.path.basename(scenarios_file).replace('.json', '')
            plot_config = None
            description = None
        elif isinstance(data, dict) and "scenarios" in data:
            scenarios = data["scenarios"]
            experiment_name = data.get("name", os.path.basename(scenarios_file).replace('.json', ''))
            plot_config = data.get("plot")
            description = data.get("description")
        else:
            log_error("O arquivo deve conter uma lista de cenários ou um objeto com chave 'scenarios'.")
            return

        if not isinstance(scenarios, list) or len(scenarios) == 0:
            log_error("Nenhum cenário encontrado no arquivo.")
            return

        log_info(f"Experimento: {experiment_name}")
        log_success(f"Carregados {len(scenarios)} cenários para execução.")

        # Auto-detect experiment directory and use its results/ folder
        experiment_dir = Path(os.path.dirname(os.path.abspath(scenarios_file)))
        if output_dir == "./results":
            output_dir = str(experiment_dir / "results")
            log_info(f"Usando diretório de resultados do experimento: {output_dir}")

    except Exception as e:
        log_error(f"Erro ao carregar o arquivo de cenários: {e}")
        return

    # Create PlotConfig from dict
    config = PlotConfig.from_dict(plot_config) if plot_config else PlotConfig()

    # Create Experiment object
    experiment = Experiment(
        name=experiment_name,
        path=experiment_dir,
        description=description,
        plot_config=config,
        scenarios=scenarios
    )

    # Use parallelized execution
    result_files = run_experiment_scenarios(
        experiment=experiment,
        verbose=verbose,
        use_simple=use_simple,
        create_zip=zip,
        no_plot=no_plot,
        save_plot=save_plot
    )

    # Summary
    end_time = time.time()
    total_execution_time = end_time - start_time

    console.print(format_title("Resumo da Execução"))
    console.print(f"  {format_param('Experimento', experiment_name)}")
    console.print(f"  {format_param('Total de cenários', len(scenarios))}")
    console.print(f"  {format_param('Cenários processados', len(result_files))}")
    console.print(f"  {format_param('Tempo total', f'{total_execution_time:.2f} segundos')}")
    console.print(f"  {format_param('Diretório de resultados', output_dir)}")

    log_success(f"Execução concluída. {len(result_files)} cenário(s) processado(s).")