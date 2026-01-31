"""
AI-powered analysis module for SPKMC experiments.

This module provides optional AI analysis capabilities using OpenAI's API.
Analysis is only generated when the OPENAI_API_KEY environment variable is set.

Example usage:
    from spkmc.analysis import try_generate_analysis, AIAnalyzer

    # Check if AI analysis is available
    if AIAnalyzer.is_available():
        analysis_path = try_generate_analysis(
            experiment_name="Network Comparison",
            experiment_description="Does network type affect epidemic size?",
            results=loaded_results,
            results_dir=Path("experiments/network_comparison/results")
        )
        if analysis_path:
            print(f"Analysis saved to: {analysis_path}")
"""

from spkmc.analysis.ai_analyzer import AIAnalyzer, try_generate_analysis
from spkmc.analysis.metrics import (
    ExperimentMetrics,
    ScenarioMetrics,
    extract_experiment_metrics,
    extract_scenario_metrics,
)

__all__ = [
    "AIAnalyzer",
    "try_generate_analysis",
    "ScenarioMetrics",
    "ExperimentMetrics",
    "extract_scenario_metrics",
    "extract_experiment_metrics",
]
