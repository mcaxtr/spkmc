"""
Configuration models for SPKMC.

This module contains configuration classes for plotting and other settings.
"""

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class PlotConfig(BaseModel):
    """Configuration for plotting results."""

    title: Optional[str] = None
    xlabel: str = "Time"
    ylabel: str = "Proportion of Individuals"
    legend_position: str = "best"
    figsize: Tuple[float, float] = (10, 6)
    colors: Dict[str, str] = Field(default_factory=lambda: {"S": "blue", "I": "red", "R": "green"})
    states_to_plot: List[str] = Field(default_factory=lambda: ["S", "I", "R"])
    dpi: int = 300
    grid: bool = True
    grid_alpha: float = 0.3

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlotConfig":
        """
        Create PlotConfig from a dictionary.

        Args:
            data: Dictionary with plot settings

        Returns:
            PlotConfig instance
        """
        figsize = data.get("figsize", [10, 6])
        return cls(
            title=data.get("title"),
            xlabel=data.get("xlabel", "Time"),
            ylabel=data.get("ylabel", "Proportion of Individuals"),
            legend_position=data.get("legend_position", "best"),
            figsize=tuple(figsize) if isinstance(figsize, list) else figsize,
            colors=data.get("colors", {"S": "blue", "I": "red", "R": "green"}),
            states_to_plot=data.get("states_to_plot", ["S", "I", "R"]),
            dpi=data.get("dpi", 300),
            grid=data.get("grid", True),
            grid_alpha=data.get("grid_alpha", 0.3),
        )
