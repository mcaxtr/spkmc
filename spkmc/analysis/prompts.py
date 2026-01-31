"""
Prompt templates for AI-powered analysis of SPKMC experiments.

This module contains the system prompt and functions to build user prompts
for generating academic-style analysis of epidemic simulation results.
"""

from typing import List

from spkmc.analysis.metrics import ExperimentMetrics, ScenarioMetrics

SYSTEM_PROMPT = """You are a computational epidemiologist analyzing SIR model simulations \
on complex networks. Your task is to provide rigorous scientific analysis of results.

Writing Style:
- Use formal academic/scientific language
- Be precise and quantitative - always cite specific numbers
- Use proper epidemiological terminology (basic reproduction number, epidemic threshold, \
attack rate, herd immunity threshold, network topology, degree distribution, etc.)
- Focus on findings that address the research question

Structure your analysis with these sections:
1. **Introduction** - Brief context relating findings to the research question (2-3 sentences)
2. **Results** - Key quantitative findings with specific numbers and comparisons
3. **Discussion** - Epidemiological interpretation of the patterns observed
4. **Conclusion** - Direct answer to the research question with main takeaways

Keep the analysis focused and concise (approximately 400-600 words)."""


def _format_network_info(scenario: ScenarioMetrics) -> str:
    """Format network information for a scenario."""
    info = f"N={scenario.nodes:,}"
    if scenario.k_avg is not None:
        info += f", k_avg={scenario.k_avg}"
    if scenario.exponent is not None:
        info += f", gamma={scenario.exponent}"
    return info


def _format_distribution_info(scenario: ScenarioMetrics) -> str:
    """Format distribution information for a scenario."""
    dist = scenario.distribution.capitalize()
    params = []
    if scenario.shape is not None:
        params.append(f"shape={scenario.shape}")
    if scenario.scale is not None:
        params.append(f"scale={scenario.scale}")
    if scenario.mu is not None:
        params.append(f"mu={scenario.mu}")
    if scenario.lambda_param is not None:
        params.append(f"lambda={scenario.lambda_param}")

    if params:
        return f"{dist} ({', '.join(params)})"
    return dist


def build_experiment_prompt(metrics: ExperimentMetrics) -> str:
    """
    Build the prompt for single experiment analysis.

    Args:
        metrics: Extracted experiment metrics

    Returns:
        Formatted prompt string for the LLM
    """
    prompt = f"""Analyze the following epidemic simulation experiment results:

## Experiment: {metrics.name}

## Research Question
{metrics.description}

## Simulation Details
- Model: SIR (Susceptible-Infected-Recovered)
- Method: Shortest Path Kinetic Monte Carlo (SPKMC)
- Number of scenarios: {len(metrics.scenarios)}

## Scenario Results

"""
    for scenario in metrics.scenarios:
        network_type = scenario.network_type.upper()
        if network_type == "ER":
            network_name = "Erdos-Renyi"
        elif network_type == "CN":
            network_name = "Scale-free (Power-law)"
        elif network_type == "RRN":
            network_name = "Random Regular"
        elif network_type == "CG":
            network_name = "Complete Graph"
        else:
            network_name = network_type

        prompt += f"""### {scenario.label}
- **Network**: {network_name} ({_format_network_info(scenario)})
- **Distribution**: {_format_distribution_info(scenario)}
- **Peak Infection**: {scenario.peak_infection:.4f} (at t = {scenario.peak_infection_time:.2f})
- **Final Outbreak Size**: {scenario.final_outbreak_size:.4f}
- **Attack Rate**: {scenario.attack_rate:.2%}
- **Epidemic Duration**: {scenario.epidemic_duration:.2f} time units
- **Simulation**: {scenario.samples} samples, {scenario.num_runs} runs, \
initial infected = {scenario.initial_perc:.1%}

"""

    prompt += f"""## Cross-Scenario Comparison
- Highest peak infection: **{metrics.max_peak_scenario}**
- Lowest peak infection: **{metrics.min_peak_scenario}**
- Peak infection variation: {metrics.peak_variation_range:.4f}
- Final outbreak size variation: {metrics.final_size_variation_range:.4f}

Please provide a scientific analysis that directly addresses the research question."""

    return prompt


def build_collection_prompt(all_experiment_metrics: List[ExperimentMetrics]) -> str:
    """
    Build prompt for collection-level summary across all experiments.

    Used when running with --all flag.

    Args:
        all_experiment_metrics: Metrics from all completed experiments

    Returns:
        Formatted prompt for collection-level synthesis
    """
    prompt = """Synthesize the findings from the following epidemic modeling experiments. \
Identify overarching patterns, common themes, and provide a unified scientific summary.

## Experiments Analyzed

"""
    for exp in all_experiment_metrics:
        prompt += f"""### {exp.name}
- **Research Question**: {exp.description}
- **Scenarios**: {len(exp.scenarios)}
- **Key Finding**: Peak infection ranged from {exp.min_peak_scenario} \
to {exp.max_peak_scenario}
- **Peak Variation**: {exp.peak_variation_range:.4f}
- **Final Size Variation**: {exp.final_size_variation_range:.4f}

"""

    prompt += """## Task
Provide a 2-3 paragraph synthesis that:
1. Identifies the major conclusions about epidemic dynamics on networks
2. Highlights any unexpected or particularly significant findings
3. Suggests implications for understanding real-world epidemic spread

Keep the summary focused and accessible while maintaining scientific rigor."""

    return prompt
