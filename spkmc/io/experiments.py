"""
Gerenciamento de experimentos para o algoritmo SPKMC.

Este módulo contém funções e classes para o gerenciamento de experimentos,
incluindo descoberta, carregamento, validação e execução de experimentos.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class PlotConfig:
    """Configuração para plotagem de resultados."""
    title: Optional[str] = None
    xlabel: str = "Tempo"
    ylabel: str = "Proporção de Indivíduos"
    legend_position: str = "best"
    figsize: Tuple[float, float] = (10, 6)
    colors: Dict[str, str] = field(default_factory=lambda: {"S": "blue", "I": "red", "R": "green"})
    states_to_plot: List[str] = field(default_factory=lambda: ["S", "I", "R"])
    dpi: int = 300
    grid: bool = True
    grid_alpha: float = 0.3

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlotConfig':
        """
        Cria PlotConfig a partir de um dicionário.

        Args:
            data: Dicionário com configurações de plot

        Returns:
            Instância de PlotConfig
        """
        figsize = data.get("figsize", [10, 6])
        return cls(
            title=data.get("title"),
            xlabel=data.get("xlabel", "Tempo"),
            ylabel=data.get("ylabel", "Proporção de Indivíduos"),
            legend_position=data.get("legend_position", "best"),
            figsize=tuple(figsize) if isinstance(figsize, list) else figsize,
            colors=data.get("colors", {"S": "blue", "I": "red", "R": "green"}),
            states_to_plot=data.get("states_to_plot", ["S", "I", "R"]),
            dpi=data.get("dpi", 300),
            grid=data.get("grid", True),
            grid_alpha=data.get("grid_alpha", 0.3)
        )


@dataclass
class Experiment:
    """Representa um experimento SPKMC."""
    name: str
    path: Path
    description: Optional[str] = None
    plot_config: PlotConfig = field(default_factory=PlotConfig)
    scenarios: List[Dict[str, Any]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)

    @property
    def results_dir(self) -> Path:
        """Retorna o caminho do diretório de resultados."""
        return self.path / "results"

    @property
    def has_results(self) -> bool:
        """Verifica se o experimento tem resultados."""
        return self.results_dir.exists() and any(self.results_dir.glob("*.json"))

    @property
    def result_count(self) -> int:
        """Retorna o número de arquivos de resultado."""
        if not self.results_dir.exists():
            return 0
        # Count all JSON files except comparison metadata
        return len([f for f in self.results_dir.glob("*.json") if not f.name.startswith("comparison")])

    def clean_results(self) -> None:
        """Remove todos os resultados do experimento."""
        if self.results_dir.exists():
            shutil.rmtree(self.results_dir)

    def ensure_results_dir(self) -> Path:
        """Garante que o diretório de resultados exista."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        return self.results_dir


class ExperimentManager:
    """Gerencia experimentos SPKMC."""

    DEFAULT_EXPERIMENTS_DIR = "experiments"
    DATA_FILE_NAME = "data.json"

    def __init__(self, experiments_dir: Optional[str] = None):
        """
        Inicializa o gerenciador de experimentos.

        Args:
            experiments_dir: Diretório base para experimentos (opcional)
        """
        self.experiments_dir = Path(
            experiments_dir or
            os.environ.get("SPKMC_EXPERIMENTS_DIR") or
            self.DEFAULT_EXPERIMENTS_DIR
        )

    def list_experiments(self) -> List[Experiment]:
        """
        Lista todos os experimentos disponíveis.

        Returns:
            Lista de objetos Experiment
        """
        experiments = []

        if not self.experiments_dir.exists():
            return experiments

        for exp_dir in sorted(self.experiments_dir.iterdir()):
            if exp_dir.is_dir():
                data_file = exp_dir / self.DATA_FILE_NAME
                if data_file.exists():
                    try:
                        experiment = self.load_experiment(exp_dir.name)
                        experiments.append(experiment)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        # Skip invalid experiments
                        continue

        return experiments

    def load_experiment(self, experiment_name: str) -> Experiment:
        """
        Carrega um experimento pelo nome.

        Args:
            experiment_name: Nome do diretório do experimento

        Returns:
            Objeto Experiment

        Raises:
            FileNotFoundError: Se o experimento não existir
            ValueError: Se o data.json for inválido
        """
        exp_path = self.experiments_dir / experiment_name
        data_file = exp_path / self.DATA_FILE_NAME

        if not data_file.exists():
            raise FileNotFoundError(f"Experimento não encontrado: {experiment_name}")

        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Validate required fields
        if "name" not in data:
            raise ValueError(f"Campo 'name' obrigatório em {data_file}")
        if "scenarios" not in data or not data["scenarios"]:
            raise ValueError(f"Campo 'scenarios' obrigatório e não pode estar vazio em {data_file}")

        # Filter out comment objects from scenarios
        scenarios = [s for s in data["scenarios"] if not s.get("_comment")]

        # Parse plot config
        plot_config = PlotConfig.from_dict(data.get("plot", {}))

        # Extract global parameters (used as defaults for scenarios)
        global_params = data.get("parameters", {})

        # Normalize parameter key names (data.json format -> internal format)
        key_mapping = {
            "time_max": "t_max",
            "time_points": "steps",
        }

        def normalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
            """Normalize parameter keys to internal format."""
            normalized = {}
            for key, value in params.items():
                normalized_key = key_mapping.get(key, key)
                normalized[normalized_key] = value
            return normalized

        normalized_global = normalize_params(global_params)

        # Merge global parameters into each scenario (scenario values override global)
        merged_scenarios = []
        for scenario in scenarios:
            normalized_scenario = normalize_params(scenario)
            merged = {**normalized_global, **normalized_scenario}
            merged_scenarios.append(merged)

        return Experiment(
            name=data["name"],
            path=exp_path,
            description=data.get("description"),
            plot_config=plot_config,
            scenarios=merged_scenarios,
            parameters=global_params
        )

    def get_experiment_by_index(self, index: int) -> Optional[Experiment]:
        """
        Obtém um experimento pelo índice na lista.

        Args:
            index: Índice do experimento (1-based)

        Returns:
            Objeto Experiment ou None se não encontrado
        """
        experiments = self.list_experiments()
        if 1 <= index <= len(experiments):
            return experiments[index - 1]
        return None

    def experiment_exists(self, experiment_name: str) -> bool:
        """
        Verifica se um experimento existe.

        Args:
            experiment_name: Nome do diretório do experimento

        Returns:
            True se o experimento existir
        """
        exp_path = self.experiments_dir / experiment_name
        data_file = exp_path / self.DATA_FILE_NAME
        return data_file.exists()
