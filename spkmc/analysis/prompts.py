"""
Prompt templates for AI-powered analysis of SPKMC experiments.

This module contains the system prompt and functions to build user prompts
for generating academic-style analysis of epidemic simulation results.
"""

from typing import Dict, List

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
        elif network_type == "SF":
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


CROSS_EXPERIMENT_SYSTEM_PROMPT = """You are a computational epidemiologist synthesizing findings \
from multiple epidemic modeling experiments. You are given the individual AI-generated analyses \
for each experiment. Your task is to create a comprehensive meta-analysis that identifies \
overarching patterns and unified conclusions.

Writing Style:
- Use formal academic/scientific language
- Be precise and quantitative - cite specific findings from the individual analyses
- Use proper epidemiological terminology
- Identify patterns that span multiple experiments
- Highlight convergent and divergent findings across experiments

Structure your meta-analysis with these sections:
1. **Executive Summary** - 2-3 sentence overview of the key cross-cutting findings
2. **Cross-Experiment Patterns** - Common themes and consistent findings across experiments
3. **Notable Findings** - Unique or surprising results from individual experiments
4. **Unified Conclusions** - What the collection of experiments tells us as a whole
5. **Implications** - Practical implications for understanding epidemic dynamics on networks

Keep the analysis focused and insightful (approximately 800-1200 words)."""


def build_cross_experiment_prompt(experiment_analyses: List[Dict[str, str]]) -> str:
    """
    Build prompt for cross-experiment meta-analysis from individual analysis.md contents.

    This creates a synthesis prompt that takes the text of individual experiment
    analyses and asks for a meta-analysis identifying overarching patterns.

    Args:
        experiment_analyses: List of {"name": str, "analysis": str} dictionaries
            where "analysis" is the content of each experiment's analysis.md file

    Returns:
        Formatted prompt string for the LLM
    """
    prompt = """Synthesize the following individual experiment analyses into a comprehensive \
meta-analysis. Identify overarching patterns, common themes, and provide unified conclusions.

## Individual Experiment Analyses

"""
    for exp in experiment_analyses:
        prompt += f"""### {exp["name"]}

{exp["analysis"]}

---

"""

    prompt += """## Task

Based on the individual analyses above, provide a meta-analysis that:
1. Identifies patterns that appear consistently across multiple experiments
2. Highlights any contradictions or tensions between experiment findings
3. Synthesizes the individual conclusions into unified insights about epidemic dynamics
4. Notes any methodological observations about the simulation approach

Focus on what we learn from the collection of experiments as a whole, beyond what any \
single experiment could tell us."""

    return prompt
