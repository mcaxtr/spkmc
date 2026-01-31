# SPKMC Usage Guide

This document provides detailed information on how to use the SPKMC package for epidemic spread simulations on complex networks.

## Installation

### Requirements

- Python 3.8 or newer
- Dependencies listed in `requirements.txt`:
  - NumPy: Efficient numerical operations
  - SciPy: Scientific and mathematical algorithms
  - NetworkX: Creation and manipulation of complex networks
  - Matplotlib: Result visualization
  - Numba: Python code acceleration
  - tqdm: Progress bars
  - Click: Command-line interface
  - Pandas: Data manipulation and export
  - Colorama: Colored terminal output
  - Rich: Advanced terminal formatting
  - openpyxl: Excel export
  - joblib: Task parallelization

### Install via pip

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Optional Dependencies

Some SPKMC features depend on specific packages:

- **Data export**: Requires `pandas` and `openpyxl` to export to CSV and Excel
- **Advanced CLI**: Requires `colorama` and `rich` for formatted/colorized output
- **Parallelization**: Requires `joblib` for parallel simulation runs

If any of these dependencies are missing, SPKMC will still run with reduced functionality. Warning messages will be shown when a feature is unavailable due to a missing dependency.

### Dependency Troubleshooting

If you see errors related to missing dependencies, such as `ModuleNotFoundError: No module named 'pandas'`, ensure all dependencies are installed:

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install a specific dependency
pip install pandas
```

## Command-Line Interface (CLI)

The SPKMC CLI provides a complete interface to run simulations, visualize results, and retrieve information about previous simulations.

### Global Options

The following options are available for all CLI commands:

| Option | Description |
|--------|-------------|
| `--verbose`, `-v` | Enable verbose mode for debugging |
| `--no-color` | Disable colors in output |
| `--simple` | Generate a simplified CSV result file (time, infected, error). It can be used in two ways:
|            | - As a global option before the command: `spkmc --simple batch ...`
|            | - As a command-specific option after the command: `spkmc batch ... --simple`
|            | Both forms have the same effect. |

### Available Commands

The SPKMC CLI provides the following main commands:

- `run`: Run an SPKMC simulation
- `plot`: Visualize results from previous simulations
- `info`: Show information about saved simulations
- `compare`: Compare results from multiple simulations

### `run` Command

The `run` command executes an SPKMC simulation with the specified parameters.

#### Syntax

```bash
python spkmc_cli.py run [OPTIONS]
```

#### Options

| Option | Abbrev | Type | Default | Description |
|--------|--------|------|---------|-------------|
| `--network-type` | `-n` | string | `er` | Network type: Erdos-Renyi (er), Complex Network (cn), Complete Graph (cg), Random Regular Network (rrn) |
| `--dist-type` | `-d` | string | `gamma` | Distribution type: Gamma or Exponential |
| `--shape` | | float | `2.0` | Shape parameter for Gamma distribution |
| `--scale` | | float | `1.0` | Scale parameter for Gamma distribution |
| `--mu` | | float | `1.0` | Mu parameter for Exponential distribution (recovery) |
| `--lambda` | | float | `1.0` | Lambda parameter for infection times |
| `--exponent` | | float | `2.5` | Exponent for complex networks (CN) |
| `--nodes` | `-N` | int | `1000` | Number of nodes in the network |
| `--k-avg` | | float | `10` | Average degree of the network |
| `--samples` | `-s` | int | `50` | Number of samples per run |
| `--num-runs` | `-r` | int | `2` | Number of runs (for averages) |
| `--initial-perc` | `-i` | float | `0.01` | Initial percentage of infected |
| `--t-max` | | float | `10.0` | Maximum simulation time |
| `--steps` | | int | `100` | Number of time steps |
| `--output` | `-o` | string | | Path to save results (optional) |
| `--no-plot` | | flag | `False` | Do not display the results plot |
| `--overwrite` | | flag | `False` | Overwrite existing results |
| `--zip` | | flag | `False` | Create a zip file with results |

#### Examples

```bash
# Basic simulation with Erdos-Renyi network and Gamma distribution
python spkmc_cli.py run -n er -d gamma --shape 2.0 --scale 1.0 -N 1000 --k-avg 10 -s 50

# Simulation with complex network and Exponential distribution
python spkmc_cli.py run -n cn -d exponential --mu 1.0 --lambda 1.0 --exponent 2.5 -N 1000 --k-avg 10 -s 50

# Simulation with complete graph, saving results
python spkmc_cli.py run -n cg -d gamma --shape 2.0 --scale 1.0 -N 500 -s 50 -o results/cg_gamma.json

# Simulation with simplified CSV (as a global option before the command)
python spkmc_cli.py --simple run -n er -d gamma --shape 2.0 --scale 1.0 -N 1000 -o results/er_gamma_simple.json

# Simulation with simplified CSV (as an option after the command)
python spkmc_cli.py run -n er -d gamma --shape 2.0 --scale 1.0 -N 1000 -o results/er_gamma_simple.json --simple

# Simulation with zip file generation
python spkmc_cli.py run -n er -d gamma --shape 2.0 --scale 1.0 -N 1000 -o results/er_gamma.json --zip

# Simulation with simplified CSV and zip file
python spkmc_cli.py run -n er -d gamma --shape 2.0 --scale 1.0 -N 1000 -o results/er_gamma.json --simple --zip

# Both forms above generate a simplified CSV with time, infected, and error
```

### `plot` Command

The `plot` command visualizes results from a previous simulation.

#### Syntax

```bash
python spkmc_cli.py plot RESULT_FILE [OPTIONS]
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `RESULT_FILE` | Path to the results file |

#### Options

| Option | Abbrev | Type | Default | Description |
|--------|--------|------|---------|-------------|
| `--with-error` | `-e` | flag | `False` | Show error bars (if available) |
| `--output` | `-o` | string | | Save the plot to a file (optional) |

#### Examples

```bash
# Visualize basic results
python spkmc_cli.py plot data/spkmc/gamma/ER/results_1000_50_2.0.json

# Visualize with error bars
python spkmc_cli.py plot data/spkmc/gamma/ER/results_1000_50_2.0.json --with-error

# Save the plot to a file
python spkmc_cli.py plot data/spkmc/gamma/ER/results_1000_50_2.0.json -o plots/er_gamma.png
```

### `info` Command

The `info` command shows information about saved simulations.

#### Syntax

```bash
python spkmc_cli.py info [OPTIONS]
```

#### Options

| Option | Abbrev | Type | Default | Description |
|--------|--------|------|---------|-------------|
| `--result-file` | `-f` | string | | Specific results file (optional) |
| `--list` | `-l` | flag | `False` | List all available results files |

#### Examples

```bash
# List all available result files
python spkmc_cli.py info --list

# Show information for a specific file
python spkmc_cli.py info -f data/spkmc/gamma/ER/results_1000_50_2.0.json
```

### `compare` Command

The `compare` command compares results from multiple simulations.

#### Syntax

```bash
python spkmc_cli.py compare RESULT_FILES... [OPTIONS]
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `RESULT_FILES` | Paths to results files (at least one) |

#### Options

| Option | Abbrev | Type | Default | Description |
|--------|--------|------|---------|-------------|
| `--labels` | `-l` | string (multiple) | | Labels for each file (optional) |
| `--output` | `-o` | string | | Save the plot to a file (optional) |

#### Examples

```bash
# Compare two results files
python spkmc_cli.py compare data/spkmc/gamma/ER/results_1000_50_2.0.json data/spkmc/exponential/ER/results_1000_50_.json

# Compare with custom labels
python spkmc_cli.py compare data/spkmc/gamma/ER/results_1000_50_2.0.json data/spkmc/exponential/ER/results_1000_50_.json -l "Gamma" "Exponential"

# Save the comparison plot
python spkmc_cli.py compare data/spkmc/gamma/ER/results_1000_50_2.0.json data/spkmc/exponential/ER/results_1000_50_.json -o plots/comparison.png
```

## Programmatic Usage

In addition to the CLI, SPKMC can be used programmatically in your own Python scripts.

### Imports

```python
from spkmc import SPKMC, GammaDistribution, ExponentialDistribution, NetworkFactory
import numpy as np
```

### Creating Distributions

#### Gamma Distribution

```python
# Parameters: shape, scale, lambda
gamma_dist = GammaDistribution(shape=2.0, scale=1.0, lmbd=1.0)
```

#### Exponential Distribution

```python
# Parameters: mu, lambda
exp_dist = ExponentialDistribution(mu=1.0, lmbd=1.0)
```

### Creating Networks

#### Erdos-Renyi Network

```python
# Parameters: number of nodes, average degree
G = NetworkFactory.create_erdos_renyi(N=1000, k_avg=10)
```

#### Complex Network

```python
# Parameters: number of nodes, exponent, average degree
G = NetworkFactory.create_complex_network(N=1000, exponent=2.5, k_avg=10)
```

#### Complete Graph

```python
# Parameters: number of nodes
G = NetworkFactory.create_complete_graph(N=100)
```

### Running Simulations

#### Simulator Initialization

```python
# Initialize the simulator with a distribution
simulator = SPKMC(distribution=gamma_dist)
```

#### Parameter Configuration

```python
# Number of nodes
N = 1000

# Average degree
k_avg = 10

# Number of samples
samples = 50

# Initial percentage of infected
initial_perc = 0.01

# Time configuration
t_max = 10.0
steps = 100
time_steps = np.linspace(0, t_max, steps)
```

#### Erdos-Renyi Simulation

```python
# Run the simulation
S, I, R, S_err, I_err, R_err = simulator.simulate_erdos_renyi(
    num_runs=2,
    time_steps=time_steps,
    N=N,
    k_avg=k_avg,
    samples=samples,
    initial_perc=initial_perc
)
```

#### Complex Network Simulation

```python
# Run the simulation
S, I, R, S_err, I_err, R_err = simulator.simulate_complex_network(
    num_runs=2,
    exponent=2.5,
    time_steps=time_steps,
    N=N,
    k_avg=k_avg,
    samples=samples,
    initial_perc=initial_perc
)
```

#### Complete Graph Simulation

```python
# Run the simulation
S, I, R = simulator.simulate_complete_graph(
    time_steps=time_steps,
    N=N,
    samples=samples,
    initial_perc=initial_perc
)
```

#### Generic Simulation

```python
# Run the simulation based on the network type
result = simulator.run_simulation(
    network_type="er",  # or "cn", "cg"
    time_steps=time_steps,
    N=N,
    k_avg=k_avg,
    samples=samples,
    initial_perc=initial_perc,
    num_runs=2,
    overwrite=False
)

# Extract results
S = result["S_val"]
I = result["I_val"]
R = result["R_val"]
has_error = result.get("has_error", False)

if has_error:
    S_err = result["S_err"]
    I_err = result["I_err"]
    R_err = result["R_err"]
```

### Result Visualization

#### Basic Visualization

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(time_steps, S, 'b-', label='Susceptible')
plt.plot(time_steps, I, 'r-', label='Infected')
plt.plot(time_steps, R, 'g-', label='Recovered')

plt.xlabel('Time')
plt.ylabel('Proportion of Individuals')
plt.title('SIR Model Dynamics Over Time')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

#### Visualization with Error Bars

```python
from spkmc.visualization.plots import Visualizer

Visualizer.plot_result_with_error(S, I, R, S_err, I_err, R_err, time_steps,
                                 title="SPKMC Simulation with Error Bars")
```

#### Network Visualization

```python
Visualizer.plot_network(G, title="Network Visualization")
```

#### Results Comparison

```python
# Results from multiple simulations
results = [result1, result2]
labels = ["Simulation 1", "Simulation 2"]

Visualizer.compare_results(results, labels, title="SPKMC Simulation Comparison")
```

### Results Management

#### Saving Results

```python
from spkmc.io.results import ResultManager

# Create a results dictionary
result = {
    "S_val": list(S),
    "I_val": list(I),
    "R_val": list(R),
    "time": list(time_steps),
    "metadata": {
        "network_type": "ER",
        "distribution": "gamma",
        "N": N,
        "k_avg": k_avg,
        "samples": samples
    }
}

# Save results
ResultManager.save_result("results/my_simulation.json", result)
```

#### Loading Results

```python
# Load results
result = ResultManager.load_result("results/my_simulation.json")

# Extract data
S = np.array(result["S_val"])
I = np.array(result["I_val"])
R = np.array(result["R_val"])
time_steps = np.array(result["time"])
```

#### Listing Results

```python
# List all available results
result_files = ResultManager.list_results()

# List results with filter
gamma_results = ResultManager.list_results(filter_by={"metadata.distribution": "gamma"})
```

#### Metadata Extraction

```python
# Extract metadata from file path
metadata = ResultManager.get_metadata_from_path("data/spkmc/gamma/ER/results_1000_50_2.0.json")

# Format result for CLI display
formatted = ResultManager.format_result_for_cli(result)
```

## Complete Examples

For complete usage examples, see the following files:

- [CLI usage examples](../examples/cli_examples.md)
- [Basic simulation example](../examples/basic_example.py)
- [Programmatic CLI example](../examples/basic_simulation.py)

## Tips and Best Practices

### Choosing Network Type

- **Erdos-Renyi (ER)**: Good for initial simulations and tests, as it is a simple random network.
- **Complex Network (CN)**: More realistic for social and biological networks, as it follows a power-law distribution.
- **Complete Graph (CG)**: Useful for extreme cases where all nodes are connected.

### Choosing Distribution

- **Gamma**: More flexible for modeling recovery times, allowing shape and scale adjustments.
- **Exponential**: Simpler, assumes events occur at a constant rate.

### Performance Optimization

- For large networks (N > 10000), consider reducing the number of samples and runs.
- The `--k-avg` parameter (average degree) significantly affects runtime; higher values result in more connections and more expensive calculations.
- Use `--no-plot` to avoid generating plots during execution, which can save time and resources.

### Results Analysis

- Compare different network types with the same parameters to understand how network structure affects spread.
- Vary distribution parameters to observe how recovery and infection times affect epidemic dynamics.
- Use the `compare` command to visualize multiple simulations in a single plot.

### `batch` Command

The `batch` command runs multiple simulation scenarios from a JSON file, making it easier to execute experiments in bulk.

#### Syntax

```bash
python spkmc_cli.py batch [SCENARIOS_FILE] [OPTIONS]
```

#### Arguments

Argument | Description |
|--------|-------------|
`SCENARIOS_FILE` | Path to the JSON file containing scenarios (optional - if omitted, shows the experiment menu) |

#### Options

Option | Abbrev | Type | Default | Description |
|-------|--------|------|---------|-------------|
`--output-dir` | `-o` | string | `./results` | Directory to save results |
`--prefix` | `-p` | string | | Prefix for output filenames |
`--compare` | `-c` | flag | `False` | Generate comparative visualization of results |
`--no-plot` | | flag | `False` | Disable individual plot generation |
`--save-plot` | | flag | `False` | Save plots to files |
`--zip` | | flag | `False` | Create a zip file with results for each scenario |
`--verbose` | `-v` | flag | `False` | Show detailed information during execution |

#### Scenario JSON File Format

The JSON file must contain a list of objects, each representing a simulation scenario. Each scenario runs sequentially and results are saved to separate files in the specified directory.

Example JSON file:

```json
[
  {
    "network_type": "er",
    "dist_type": "gamma",
    "nodes": 1000,
    "k_avg": 10,
    "shape": 2.0,
    "scale": 1.0,
    "lambda_val": 0.5,
    "samples": 50,
    "num_runs": 2,
    "initial_perc": 0.01,
    "t_max": 10.0,
    "steps": 100,
    "label": "er_gamma_shape2"
  },
  {
    "network_type": "cn",
    "dist_type": "exponential",
    "nodes": 2000,
    "k_avg": 8,
    "exponent": 2.5,
    "mu": 1.0,
    "lambda_val": 0.5,
    "samples": 50,
    "num_runs": 2,
    "initial_perc": 0.01,
    "t_max": 10.0,
    "steps": 100,
    "label": "cn_exp_exponent2.5"
  }
]
```

Available parameters for each scenario:

Parameter | Type | Default | Description |
|----------|------|---------|-------------|
`network_type` | string | `er` | Network type: Erdos-Renyi (er), Complex Network (cn), Complete Graph (cg) |
`dist_type` | string | `gamma` | Distribution type: Gamma or Exponential |
`nodes` | int | `1000` | Number of nodes in the network |
`k_avg` | float | `10` | Average degree (for er and cn) |
`shape` | float | `2.0` | Shape parameter for Gamma distribution |
`scale` | float | `1.0` | Scale parameter for Gamma distribution |
`mu` | float | `1.0` | Mu parameter for Exponential distribution |
`lambda_val` | float | `1.0` | Lambda parameter for infection times |
`exponent` | float | `2.5` | Exponent for complex networks (CN) |
`samples` | int | `50` | Number of samples per run |
`num_runs` | int | `2` | Number of runs (for averages) |
`initial_perc` | float | `0.01` | Initial percentage of infected |
`t_max` | float | `10.0` | Maximum simulation time |
`steps` | int | `100` | Number of time steps |
`label` | string | | Optional label to identify the scenario (used in filenames) |

#### Examples

```bash
# Run experiments (shows interactive menu)
spkmc batch

# Run scenarios from a specific file
python spkmc_cli.py batch experiments/test_scenarios.json

# Run scenarios and save results to a specific directory
python spkmc_cli.py batch --output-dir results/experiment1

# Run scenarios with an output prefix
python spkmc_cli.py batch --prefix "exp1_"

# Run scenarios and generate a comparison visualization
python spkmc_cli.py batch --compare

# Run scenarios, save plots, and do not show them on screen
python spkmc_cli.py batch --save-plot --no-plot

# Run scenarios with verbose output
python spkmc_cli.py batch --verbose

# Run scenarios and generate simplified CSVs (global option)
python spkmc_cli.py --simple batch --output-dir results/simple_csv

# Run scenarios and generate simplified CSVs (after the command)
python spkmc_cli.py batch --output-dir results/simple_csv --simple

# Run scenarios and create zip files
python spkmc_cli.py batch --output-dir results/zip_results --zip

# Run scenarios, generate simplified CSVs, and create zip files
python spkmc_cli.py batch --output-dir results/complete --simple --zip

# Both forms above generate simplified CSVs with time, infected, and error for each scenario
```

#### Tips and Best Practices for `batch`

- **Scenario organization**: Group related scenarios in the same JSON file for easier comparison and analysis.
- **Label usage**: Use the `label` parameter to clearly identify each scenario in output files.
- **Automatic comparison**: Use the `--compare` option to automatically generate a comparative plot of all scenarios.
- **Batch runs**: For large experiments, consider splitting scenarios across multiple JSON files and running them separately.
- **Meaningful prefixes**: Use prefixes that indicate the experiment purpose (e.g., `gamma_vs_exp_`, `network_size_test_`).
- **Silent mode**: For long runs, use `--no-plot` to avoid displaying plots and improve performance.
- **Experiment documentation**: Save scenario JSON files alongside results to fully document the experiment.
- **Simplified CSV files**: Use `--simple` to generate simplified CSVs with three columns (time, infected, error). This option can be used in two ways:
  - As a global option before the command: `spkmc --simple batch ...`
  - As a command-specific option after the command: `spkmc batch ... --simple`
  Both forms have the same effect. These files are useful for quick analyses or importing into visualization tools.
- **Zipped files**: Use `--zip` to create zip files containing each scenario's results. This is useful for:
  - Sharing results more neatly
  - Saving disk space when storing multiple results
  - Simplifying download and transfer of results
  When used with `batch`, in addition to creating a zip for each scenario, a single zip containing all batch results is also created.
