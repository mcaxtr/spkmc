# SPKMC - Shortest Path Kinetic Monte Carlo

A high-performance simulation tool for modeling epidemic spread on complex networks using the SIR (Susceptible-Infected-Recovered) model.

## What is SPKMC?

SPKMC implements the Shortest Path Kinetic Monte Carlo algorithm, a method for simulating how diseases spread through networks of connected individuals based on the framework introduced by [Tolić, Kleineberg & Antulov-Fantulin (2018)](https://doi.org/10.1038/s41598-018-24648-w). Instead of simulating each infection event one at a time (which can be slow), SPKMC uses graph theory to compute when each person in the network will become infected, based on the shortest weighted paths from initially infected individuals.

This approach maps SIR dynamics to edge weights representing propagation times, then uses Dijkstra's algorithm to efficiently determine infection arrival times. This makes SPKMC significantly faster than traditional Monte Carlo methods, especially for large networks with thousands or millions of nodes.

### The SIR Model

In the SIR model, each individual (node) in the network can be in one of three states:

- **Susceptible (S)**: Healthy individuals who can become infected
- **Infected (I)**: Currently infected individuals who can spread the disease
- **Recovered (R)**: Individuals who have recovered and are now immune

The simulation tracks how the proportions of S, I, and R change over time as the epidemic progresses through the network.

### Key Features

- **Experiment-driven workflow**: Define multi-scenario experiments in simple JSON files, run them with one command, and automatically generate comparison plots
- **Multiple network types**: Simulate epidemics on different network structures (random, scale-free, regular)
- **Flexible timing distributions**: Model realistic recovery and infection times using Gamma or Exponential distributions
- **High performance**: Uses Numba JIT compilation for speed, with optional GPU acceleration
- **Publication-quality plots**: Generate professional visualizations of epidemic dynamics
- **Multiple export formats**: Save results as JSON, CSV, Excel, Markdown, or HTML
- **Web interface**: Interactive Streamlit dashboard for managing experiments, viewing results, and running AI analysis in the browser

## Web Interface

SPKMC includes a full-featured web dashboard built with Streamlit for managing experiments, viewing results, and running AI analysis directly in the browser.

### Quick Start

```bash
# Launch the web interface (opens browser at http://localhost:8501)
spkmc web

# Custom port
spkmc web --port 8080

# Headless mode (no browser auto-open)
spkmc web --no-browser

# Bind to all interfaces (for remote access)
spkmc web --host 0.0.0.0
```

### Features

- **Experiment Dashboard** -- Browse all experiments with summary stats (total experiments, scenarios, completion rates). Create new experiments through a guided modal with network, distribution, and simulation parameter configuration. Click any experiment card to drill into details.

- **Scenario Management** -- View scenario cards with live status badges (Pending / Running / Completed / Failed). Add new scenarios with parameter overrides, edit existing ones, or run simulations directly from the browser. Parameters that differ from global defaults are visually highlighted.

- **Interactive Charts** -- Plotly-based SIR curve visualization with toggleable S/I/R traces, chart type switching (Line / Area / Scatter), error bands for multi-run results, and multi-scenario comparison overlays. All charts support zoom, pan, and image export.

- **AI Analysis** -- Generate academic-style analysis reports for experiments and individual scenarios using OpenAI models. Reports include epidemic dynamics interpretation, key findings, and actionable insights. Requires an OpenAI API key configured in Preferences.

- **Export** -- Download scenario results in JSON, CSV, Excel, Markdown, or HTML format directly from the scenario detail modal.

- **Preferences** -- Configure chart colors, height, and template; set default simulation parameters; manage directory paths; select AI model; and store API keys. All settings auto-save on change.

### Architecture

```
spkmc/web/
├── app.py                    # Streamlit entry point, sidebar navigation, CSS injection
├── config.py                 # Configuration management (JSON prefs + Streamlit secrets)
├── state.py                  # Typed session state accessors (prevents st.session_state spaghetti)
├── plotting.py               # Core Plotly figure builders (SIR curves, comparisons)
├── components.py             # Reusable UI components (forms, metric cards, badges)
├── styles.py                 # Design system (CSS, card renderers, color tokens)
├── runner.py                 # Subprocess-based simulation runner
├── analysis_runner.py        # AI analysis subprocess runner
└── pages/
    ├── dashboard.py          # Experiments list, stats cards, create modal
    ├── experiment_detail.py  # Single experiment view, scenario cards, detail modal
    └── settings.py           # Preferences page (AI, chart, simulation, storage)
```

### Design Decisions

**Everything is an experiment.** Even a single simulation run is treated as an experiment with one scenario. This unified model simplifies the codebase and provides consistent storage patterns.

**Subprocess execution.** Simulations run in background subprocesses that survive browser refresh, page navigation, and UI interactions. Progress is tracked via filesystem-based IPC (`.spkmc_web/status/*.json`), not in-memory state.

**Filesystem-first storage.** No database required. Experiments are stored as `experiments/<name>/data.json`, results as `experiments/<name>/<scenario>.json`, and status as `.spkmc_web/status/<run_id>.json`. All files are portable JSON.

**Parameter inheritance.** Global parameters are defined at the experiment level. Each scenario only specifies what differs from the defaults, keeping configurations DRY and making overrides instantly visible in the UI.

### Configuration

**User preferences** are stored at `~/.spkmc/web_config.json`:
- Directory paths (data, experiments)
- Default simulation parameters
- Chart styling (height, colors, template)
- Export format preference

Override the config file location with an environment variable:

```bash
SPKMC_WEB_CONFIG_FILE=/path/to/config.json spkmc web
```

**API keys** are stored in `.streamlit/secrets.toml` (managed through the Preferences page):

```toml
OPENAI_API_KEY = "sk-your-key-here"
```

### Workflow

1. Open the Dashboard and create a new experiment with global parameters
2. Add scenarios -- each can override any parameter from the global defaults
3. Run individual scenarios or all at once from the experiment detail page
4. View interactive Plotly charts with toggleable S/I/R traces
5. Compare multiple scenarios with overlaid charts
6. Generate AI analysis reports (optional, requires API key)
7. Export results in your preferred format

### Troubleshooting

**Simulations not starting** -- Check `.spkmc_web/status/*.json` files for error messages. Common issues include Numba compilation errors, missing parameters, or invalid network/distribution combinations.

**Browser doesn't open** -- Use `spkmc web --no-browser` and navigate to the URL shown in terminal output.

**Charts not displaying** -- Ensure `plotly>=5.18.0` is installed: `pip install --upgrade plotly`.

### Extending the Web Interface

To add a new page:

1. Create `spkmc/web/pages/my_page.py` with a `render()` function
2. Register the page in `spkmc/web/pages/__init__.py`
3. Add sidebar navigation in `spkmc/web/app.py`
4. Add routing logic in the `main()` function

## Installation

```bash
pip install spkmc
```

That's it. This installs SPKMC and all required dependencies.

### GPU Acceleration

SPKMC automatically detects if you have an NVIDIA GPU. If GPU hardware is found but GPU packages aren't installed, you'll see a suggestion to enable acceleration:

```
NVIDIA GPU detected but GPU acceleration packages are not installed.
For significantly faster simulations on large networks (10,000+ nodes),
install GPU support with:

    pip install spkmc[gpu]
```

To install GPU support manually:

```bash
pip install spkmc[gpu]
```

This requires an NVIDIA GPU with CUDA drivers installed.

## Quick Start

After installation, the `spkmc` command will be available in your terminal.

### Running Experiments

The most powerful way to use SPKMC is through **experiments**. An experiment is a JSON configuration file that defines multiple **scenarios** (simulations with different parameters) to run and compare automatically.

**Why experiments?**
- **Reproducibility**: Configuration is saved as JSON documenting exactly what you ran
- **Automation**: Run dozens of scenarios with one command
- **Comparison**: Automatic generation of comparison plots
- **Organization**: Results saved in `data/<experiment_name>/` directories

#### Creating Experiments

**Option 1: Interactive Wizard**

```bash
spkmc experiments
```

Select **"[+] Create New Experiment"** from the menu. The wizard guides you through:
1. Experiment name and description
2. Base parameters (network type, distribution, nodes, etc.)
3. Which parameter to vary across scenarios
4. The values for that parameter

The wizard saves the configuration to `experiments/<name>/data.json` and can run it immediately.

**Option 2: Manual Configuration**

Create `experiments/my_experiment/data.json`:

```json
{
  "name": "My Experiment",
  "description": "How does infection rate affect epidemic size?",
  "parameters": {
    "network": "er",
    "distribution": "gamma",
    "nodes": 1000,
    "k_avg": 10,
    "lambda": 0.5,
    "samples": 100,
    "num_runs": 3
  },
  "scenarios": [
    { "label": "Baseline" },
    { "label": "High Infection", "lambda": 0.8 },
    { "label": "Low Infection", "lambda": 0.3 }
  ]
}
```

The `parameters` field defines defaults; each scenario only specifies what differs.

#### Running Experiments

```bash
# Interactive menu to select and run
spkmc experiments

# Run all experiments at in sequence
spkmc experiments --all

# Re-run from scratch (clears previous results)
spkmc experiments --override
```

Results are saved to `data/<experiment_name>/` with one JSON file per scenario plus a `comparison.png` plot.

#### Experiment Options

| Option | Description |
|--------|-------------|
| `-a, --all` | Run all experiments (no menu) |
| `--override` | Clear results and re-run |
| `--no-plot` | Disable plot generation |
| `-x, --export` | Output format: `json`, `csv`, `excel`, `md`, `html` (default: `json`) |
| `--debug` | Show detailed debug info |
| `--clear-cache` | Clear Numba compilation cache |

Results are always saved to `data/experiments/<experiment_name>/`.

### Running a Single Simulation

For quick tests or one-off simulations:

```bash
spkmc run -n er -d gamma --nodes 1000 --samples 50
```

This creates an Erdos-Renyi network with 1000 nodes, uses Gamma-distributed recovery times, runs 50 samples, and displays a plot. Use `-o results.json` to save results.

## Understanding the Parameters

### Network Types

The network structure dramatically affects how epidemics spread. SPKMC supports four types:

**Erdos-Renyi (`er`)** - The simplest random network model. Each pair of nodes has an equal probability of being connected. Good for modeling well-mixed populations where everyone has roughly the same number of contacts.

**Scale-Free Network (`sf`)** - A scale-free network where some nodes (hubs) have many more connections than others, following a power-law distribution. This better represents real social networks where some people are much more connected than others.

**Complete Graph (`cg`)** - Every node is connected to every other node. Useful as a theoretical baseline but not realistic for most applications.

**Random Regular (`rrn`)** - Every node has exactly the same number of connections. Useful for studying how epidemics spread when everyone has equal contact rates.

### Distribution Types

The timing of infection and recovery events follows probability distributions:

**Gamma Distribution** - Recovery times follow a Gamma distribution controlled by `shape` and `scale` parameters. When shape > 1, there's a characteristic delay before most recoveries occur, which is realistic for many diseases. Infection times use the `lambda` parameter.

**Exponential Distribution** - Recovery times follow an Exponential distribution controlled by the `mu` parameter. Recovery events are "memoryless" - the probability of recovering doesn't depend on how long you've been infected. Infection times use the `lambda` parameter.

### Parameter Reference

The following parameters apply to both `spkmc run` and experiment scenarios.

#### Network Parameters

| Parameter | Description | Default | Applies to |
|-----------|-------------|---------|------------|
| `--nodes`, `-N` | Number of individuals in the network. Larger networks are more accurate but slower to simulate. | 1000 | All networks |
| `--k-avg` | Average number of connections per node. Higher values mean faster epidemic spread. | 10 | `er`, `sf`, `rrn` |
| `--exponent` | Power-law exponent γ for degree distribution. Lower values (e.g., 2.1) create more hub nodes; higher values (e.g., 3.0) are more uniform. Valid range: > 2.0. | 2.5 | `sf` only |

#### Distribution Parameters

**For Gamma distribution (`-d gamma`):**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--shape` | Shape parameter (k) of the Gamma distribution. Controls the "peakedness" of recovery times. | 2.0 |
| `--scale` | Scale parameter (θ) of the Gamma distribution. Mean recovery time = shape × scale. | 1.0 |
| `--lambda` | Infection rate (β). Higher values mean faster transmission along edges. | 1.0 |

**For Exponential distribution (`-d exponential`):**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--mu` | Recovery rate (γ). Mean recovery time = 1/mu. | 1.0 |
| `--lambda` | Infection rate (β). Higher values mean faster transmission along edges. | 1.0 |

#### Simulation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--samples`, `-s` | Number of Monte Carlo samples per run. More samples = smoother curves and better statistics. | 50 |
| `--num-runs`, `-r` | Number of independent runs to average. Provides error estimates when > 1. | 2 |
| `--initial-perc`, `-i` | Fraction of population initially infected (0.0 to 1.0). | 0.01 |
| `--t-max` | Maximum simulation time in arbitrary units. | 10.0 |
| `--steps` | Number of time points to record in output. | 100 |

## Command Reference

### Running Single Simulations: `spkmc run`

The `run` command executes a single simulation with the parameters you specify.

**Basic usage:**
```bash
spkmc run -n <network_type> -d <distribution> [options]
```

**Common examples:**

```bash
# Simple simulation with default parameters
spkmc run -n er -d gamma

# Larger network with more samples for publication-quality results
spkmc run -n er -d gamma --nodes 10000 --samples 200 --num-runs 5

# Scale-free network with specific power-law exponent
spkmc run -n sf -d gamma --nodes 5000 --exponent 2.5 --k-avg 8

# Using exponential distribution for recovery times
spkmc run -n er -d exponential --mu 0.5 --lambda 0.8

# Save results to a JSON file (default format)
spkmc run -n er -d gamma -o my_results.json

# Save results as CSV instead of JSON
spkmc run -n er -d gamma -o my_results --export csv

# Run without displaying the plot (useful for automated processing or servers)
spkmc run -n er -d gamma -o results.json --no-plot
```

**Note:** Results are only saved when you specify `-o, --output`. Without it, the simulation runs and displays a plot but nothing is saved to disk.

**All options for `spkmc run`:**

| Option | Default | Description |
|--------|---------|-------------|
| `-n, --network-type` | `er` | Network type: `er`, `sf`, `cg`, or `rrn` |
| `-d, --dist-type` | `gamma` | Distribution: `gamma` or `exponential` |
| `-N, --nodes` | `1000` | Number of nodes in the network |
| `--k-avg` | `10` | Average degree (connections per node) |
| `--exponent` | `2.5` | Power-law exponent for scale-free networks |
| `--shape` | `2.0` | Gamma distribution shape parameter |
| `--scale` | `1.0` | Gamma distribution scale parameter |
| `--mu` | `1.0` | Exponential distribution rate parameter |
| `--lambda` | `1.0` | Infection transmission rate |
| `-s, --samples` | `50` | Number of Monte Carlo samples |
| `-r, --num-runs` | `2` | Number of independent runs (for error bars) |
| `-i, --initial-perc` | `0.01` | Initial fraction infected (0.01 = 1%) |
| `--t-max` | `10.0` | Maximum simulation time |
| `--steps` | `100` | Number of time points to record |
| `-o, --output` | None | Path to save results (required to save anything) |
| `-e, --export` | `json` | Output format: `json`, `csv`, `excel`, `md`, `html` |
| `--no-plot` | False | Don't display the plot |
| `--override` | False | Overwrite existing output file |

---

### Visualizing Results: `spkmc plot`

The `plot` command creates visualizations from saved result files. It handles both single files and comparisons of multiple results.

**Supported file formats:** JSON (`.json`), CSV (`.csv`), Excel (`.xlsx`, `.xls`)

**Plotting a single result:**
```bash
spkmc plot results.json
spkmc plot results.csv      # Also works with CSV exports
spkmc plot results.xlsx     # Also works with Excel exports
```

**Comparing multiple results:**

Pass multiple files or directories to create a comparison plot:
```bash
# Compare specific files
spkmc plot result1.json result2.json

# Compare all scenarios from an experiment
spkmc plot data/average_degree_effect/

# Compare scenarios from multiple experiments
spkmc plot data/experiment1/ data/experiment2/

# Add custom labels
spkmc plot data/exp/k_avg_4.json data/exp/k_avg_10.json \
    -l "Low connectivity" -l "High connectivity"
```

**Customizing the plot:**
```bash
# Show only the infected curve
spkmc plot results.json -s I

# Show infected and recovered curves with error bars
spkmc plot results.json -s I -s R --with-error

# Save as a high-resolution PDF for publication
spkmc plot results.json -o figure.pdf --dpi 600

# Plot each scenario separately instead of comparing
spkmc plot data/my_experiment/ --separate
```

**All options for `spkmc plot`:**

| Option | Description |
|--------|-------------|
| `-e, --with-error` | Display error bars (requires multiple runs) |
| `-o, --output` | Save plot to file instead of displaying |
| `-f, --format` | Output format: `png`, `pdf`, `svg`, `jpg` |
| `--dpi` | Image resolution (default: 300) |
| `-s, --states` | Which states to plot (can use multiple times) |
| `--separate` | Create separate plots instead of comparison |
| `-l, --labels` | Custom labels for comparison (use multiple times) |
| `-x, --export` | Export the underlying data |

---

### Inspecting Results: `spkmc info`

The `info` command helps you explore saved results without loading them into Python.

**List all available result files:**
```bash
spkmc info --list
```

**Show details of a specific result:**
```bash
spkmc info -f results.json
```

This displays the simulation parameters, network configuration, and summary statistics.

**Export result information:**
```bash
spkmc info -f results.json --export md -o summary.md
```

**All options for `spkmc info`:**

| Option | Description |
|--------|-------------|
| `-f, --result-file` | Path to a specific result file to inspect |
| `-l, --list` | List all available result files |
| `-e, --export` | Export format: `json`, `csv`, `excel`, `md`, `html` |
| `-o, --output` | Path to save the exported file |

---

### AI-Powered Analysis: `spkmc analyze`

SPKMC can automatically generate academic-style analysis reports for your simulation results using OpenAI's language models. This feature helps interpret simulation results by providing structured scientific analysis including introduction, results interpretation, discussion, and conclusions.

#### Requirements

1. **OpenAI API Key**: Set it as an environment variable:
   ```bash
   export OPENAI_API_KEY="sk-your-api-key-here"
   ```
   You can also add this to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.) to make it permanent.

2. **OpenAI Package**: The `openai` Python package must be installed:
   ```bash
   pip install openai
   ```

#### How It Works

When you run analysis, SPKMC:

1. Loads the simulation results and metadata
2. Sends the data to OpenAI's `gpt-4o-mini` model along with experiment context
3. Generates a structured analysis in Markdown format
4. Saves the analysis as `analysis.md` in the results directory

#### Option 1: Automatic analysis with `--analyze` flag

Add the global `--analyze` flag to automatically generate analysis after running simulations:

```bash
# Run simulation and analyze results automatically
spkmc --analyze run -n er -d gamma -o result.json

# Run experiments and analyze results automatically
spkmc --analyze experiments

# Run all experiments with analysis
spkmc --analyze experiments --all
```

#### Option 2: Analyze existing results with `spkmc analyze`

**Analyze a single result file:**
```bash
spkmc analyze result.json
```

**Analyze multiple result files:**
```bash
spkmc analyze result1.json result2.json result3.json
```

**Analyze results in a directory:**
```bash
spkmc analyze experiments/network_comparison/
```

**Analyze all experiments at once:**
```bash
spkmc analyze --all
```

**Force regeneration (even if analysis already exists):**
```bash
spkmc analyze experiments/my_experiment/ --force
```

**Use a different model:**
```bash
spkmc analyze data/results/ --model gpt-4o
```

**Custom output path (single file/directory only):**
```bash
spkmc analyze data/results/ -o my_analysis.md
```

#### Output Location

The AI analysis is saved as `analysis.md` in the same directory as your results:

```
data/
├── my_experiment/
│   ├── scenario_1.json
│   ├── scenario_2.json
│   ├── comparison.png
│   └── analysis.md      ← AI-generated analysis
```

#### Analysis Structure

The generated `analysis.md` file follows an academic paper structure:

1. **Introduction**: Context about the simulation parameters and what was being tested
2. **Results**: Quantitative summary of the simulation outcomes (peak infection rates, final epidemic sizes, timing)
3. **Discussion**: Interpretation of the results and their implications
4. **Conclusion**: Key takeaways and potential next steps

#### Important Notes

- **Experiment description matters**: The AI uses your experiment's `description` field to understand what hypothesis you're testing. More detailed descriptions produce better analysis.
- **Skips existing analysis**: If `analysis.md` already exists, SPKMC won't regenerate it. Use `--force` to regenerate.
- **Requires results**: Analysis only runs after successful simulation completion.
- **API costs**: Each analysis uses the OpenAI API. The `gpt-4o-mini` model is cost-effective but check OpenAI's pricing for current rates.

#### All options for `spkmc analyze`

| Option | Description |
|--------|-------------|
| `PATHS...` | One or more files or directories to analyze |
| `-a, --all` | Analyze all experiments |
| `-m, --model` | OpenAI model to use (default: `gpt-4o-mini`) |
| `-f, --force` | Regenerate analysis even if it exists |
| `-o, --output` | Custom output path (only for single path) |

**Global `--analyze` flag:**

| Command | Description |
|---------|-------------|
| `spkmc --analyze run ...` | Run simulation and generate analysis |
| `spkmc --analyze experiments` | Run experiments and generate analysis |

---

### Cleaning Up: `spkmc clean`

The `clean` command removes result files and optionally clears the Numba compilation cache.

**Clean results for a specific experiment:**
```bash
spkmc clean network_type_comparison
```

**Clean all results (with confirmation):**
```bash
spkmc clean
```

You'll be asked to confirm before anything is deleted.

**Clean without confirmation:**
```bash
spkmc clean -y
```

**Also clear Numba's compilation cache:**
```bash
spkmc clean --numba-cache
```

This is useful if you're experiencing strange behavior after updating SPKMC.

**All options for `spkmc clean`:**

| Option | Description |
|--------|-------------|
| `-y, --yes` | Skip confirmation prompt |
| `--numba-cache` | Also clear the Numba compilation cache |

Results are always stored in `data/experiments/<experiment_name>/`.

---

## Experiment Configuration Reference

This section provides detailed documentation for the `data.json` configuration file format.

### Full Example

```json
{
  "name": "Network Type Comparison",
  "description": "How does network structure affect epidemic dynamics?",

  "parameters": {
    "distribution": "gamma",
    "nodes": 10000,
    "k_avg": 10,
    "shape": 2.0,
    "scale": 0.5,
    "lambda": 0.5,
    "samples": 100,
    "num_runs": 5,
    "t_max": 20,
    "steps": 200,
    "initial_perc": 0.01
  },

  "plot": {
    "title": "Epidemic Spread Across Network Types",
    "xlabel": "Time",
    "ylabel": "Proportion of Population",
    "states_to_plot": ["I", "R"],
    "figsize": [12, 8],
    "dpi": 300,
    "grid": true
  },

  "scenarios": [
    {
      "label": "Random Network",
      "network": "er"
    },
    {
      "label": "Scale-Free Network",
      "network": "sf",
      "exponent": 2.5
    }
  ]
}
```

### Top-Level Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Human-readable name displayed in the experiment menu |
| `description` | No | Brief description of what the experiment tests |
| `parameters` | No | Default parameters inherited by all scenarios |
| `plot` | No | Plot configuration (see below) |
| `scenarios` | Yes | Array of scenario configurations |

### Plot Configuration

| Field | Default | Description |
|-------|---------|-------------|
| `title` | Auto-generated | Title shown on the comparison plot |
| `xlabel` | `"Time"` | X-axis label |
| `ylabel` | `"Proportion of Population"` | Y-axis label |
| `states_to_plot` | `["S", "I", "R"]` | Which curves to show |
| `figsize` | `[10, 6]` | Plot dimensions in inches `[width, height]` |
| `dpi` | `300` | Resolution for saved images |
| `grid` | `true` | Whether to show grid lines |

### Scenario Parameters

Parameters not specified in a scenario inherit from the top-level `parameters` field.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `label` | `scenario_001` | Name shown in plot legend |
| `network` | `er` | `er`, `sf`, `cg`, or `rrn` |
| `distribution` | `gamma` | `gamma` or `exponential` |
| `network_size` | `1000` | Number of nodes |
| `k_avg` | `10` | Average connections per node |
| `exponent` | `2.5` | Power-law exponent (for `sf` only) |
| `shape` | `2.0` | Gamma distribution shape |
| `scale` | `1.0` | Gamma distribution scale |
| `mu` | `1.0` | Exponential distribution rate |
| `lambda` | `1.0` | Infection rate |
| `samples` | `50` | Monte Carlo samples per run |
| `num_runs` | `2` | Independent runs (for error bars) |
| `t_max` | `10.0` | Simulation duration |
| `steps` | `100` | Number of time points |
| `initial_perc` | `0.01` | Initial infected fraction |

### Tips

**Focused comparisons**: Good experiments vary only one parameter at a time. This makes it clear what's causing differences in the results.

**Recommended sample sizes**:
- Quick tests: `samples: 50`, `num_runs: 2`
- Standard analysis: `samples: 100`, `num_runs: 5`
- Publication quality: `samples: 200+`, `num_runs: 10`

---

## Using SPKMC from Python

While the CLI is convenient for quick analyses, you can also use SPKMC directly from Python for more control:

```python
from spkmc import SPKMC, GammaDistribution, NetworkFactory
import numpy as np

# Step 1: Create a probability distribution for recovery times
# shape=2.0 means there's a characteristic delay before recovery
# scale=1.0 controls the time scale
# lmbd=0.5 is the infection rate
distribution = GammaDistribution(shape=2.0, scale=1.0, lmbd=0.5)

# Step 2: Create the simulator
simulator = SPKMC(distribution)

# Step 3: Generate a network
# N=1000 nodes, average of 10 connections per node
network = NetworkFactory.create_erdos_renyi(N=1000, k_avg=10)

# Step 4: Define the time points where we want measurements
time_points = np.linspace(0, 20.0, 200)  # 200 points from t=0 to t=20

# Step 5: Run the simulation
# sources: which nodes are initially infected (node 0 in this case)
# samples: how many independent runs to average over
S, I, R = simulator.run_multiple_simulations(
    network,
    sources=np.array([0]),
    time_steps=time_points,
    samples=100
)

# S, I, R are numpy arrays with the proportion in each state at each time point
print(f"Peak infection: {I.max():.1%} at t={time_points[I.argmax()]:.1f}")
print(f"Final epidemic size: {R[-1]:.1%} of population")
```

### Available Python Classes

```python
from spkmc import (
    # Core simulation
    SPKMC,                    # Main simulator class

    # Probability distributions
    GammaDistribution,        # Gamma-distributed recovery times
    ExponentialDistribution,  # Exponential recovery times

    # Network generation
    NetworkFactory,           # Create different network types

    # Visualization
    Visualizer,               # Create plots programmatically

    # Data management
    ResultManager,            # Save and load results
)
```

---

## Where Results Are Stored

When you run experiments, results are saved in the `data/` directory:

```
data/
├── network_type_comparison/
│   ├── random_network.json
│   ├── scale_free_network.json
│   └── comparison.png
├── another_experiment/
│   └── ...
```

Each JSON file contains:

```json
{
  "S_val": [0.99, 0.95, 0.85, ...],
  "I_val": [0.01, 0.04, 0.10, ...],
  "R_val": [0.00, 0.01, 0.05, ...],
  "S_err": [0.001, 0.002, ...],
  "I_err": [0.001, 0.003, ...],
  "R_err": [0.001, 0.002, ...],
  "time": [0.0, 0.1, 0.2, ...],
  "metadata": {
    "network": "er",
    "distribution": "gamma",
    "N": 1000,
    "k_avg": 10,
    "shape": 2.0,
    "scale": 1.0,
    "lambda": 0.5,
    "samples": 100,
    "num_runs": 5
  }
}
```

The `_err` fields contain standard errors and are only present when `num_runs > 1`.

---

## Project Structure

```
spkmc/
├── analysis/                 # AI-powered analysis
│   ├── ai_analyzer.py        # OpenAI integration for experiment and scenario analysis
│   ├── metrics.py            # Metric extraction from simulation results
│   └── prompts.py            # LLM prompt templates
├── cli/                      # Command-line interface (Click-based)
│   ├── commands.py           # CLI commands: run, plot, info, compare, experiments, web
│   ├── validators.py         # Parameter validation callbacks
│   └── formatting.py         # Rich terminal output formatting
├── core/                     # Core algorithm implementation
│   ├── simulation.py         # SPKMC class - main simulation algorithm
│   ├── distributions.py      # Gamma & Exponential distribution classes
│   └── networks.py           # NetworkFactory for graph creation
├── io/                       # Input/output operations
│   ├── export.py             # Multi-format export (CSV, JSON, Excel, MD, HTML)
│   ├── data_manager.py       # Result persistence, loading, and report generation
│   └── results.py            # Result file discovery and metadata
├── models/                   # Data models
│   ├── experiment.py         # Experiment and Scenario Pydantic models
│   └── scenario.py           # Scenario configuration model
├── visualization/            # Plotting (Plotly-based)
│   └── plots.py              # SIR curve visualization for CLI and programmatic use
├── web/                      # Streamlit web interface
│   ├── app.py                # Entry point, sidebar navigation, CSS injection
│   ├── config.py             # Configuration management (JSON prefs + secrets)
│   ├── state.py              # Typed session state accessors
│   ├── plotting.py           # Core Plotly figure builders
│   ├── components.py         # Reusable UI components (forms, badges, cards)
│   ├── styles.py             # Design system (CSS, card renderers, color tokens)
│   ├── runner.py             # Subprocess-based simulation runner
│   ├── analysis_runner.py    # AI analysis subprocess runner
│   └── pages/
│       ├── dashboard.py      # Experiments list, stats cards, create modal
│       ├── experiment_detail.py  # Experiment view, scenario cards, detail modal
│       └── settings.py       # Preferences page
└── utils/                    # Utilities
    └── numba_utils.py        # Numba JIT-optimized functions

tests/
├── test_web/                 # Unit tests for web modules
│   ├── test_state.py         # Session state management tests
│   ├── test_config.py        # Configuration management tests
│   ├── test_runner.py        # Simulation runner tests
│   ├── test_analysis_runner.py  # Analysis runner tests
│   ├── test_plotting.py      # Plotly figure builder tests
│   └── test_experiment_detail.py  # Experiment detail logic tests
└── e2e/                      # Playwright end-to-end tests
    ├── conftest.py           # Server lifecycle, page helpers, fixture seeding
    ├── fixtures/             # Pre-seeded experiment data
    ├── test_navigation.py    # Sidebar nav, page routing, title
    ├── test_dashboard.py     # Stats cards, create modal, experiment cards
    ├── test_experiment_detail.py  # Params, scenario cards, modal, charts
    └── test_settings.py      # Preference sections, inputs, reset
```

---

## Development

### Running Tests

```bash
# Run all unit tests with coverage
pytest

# Run web module tests only
pytest tests/test_web/ -v

# Run E2E tests (requires playwright browsers installed)
pip install -e ".[e2e]"
playwright install chromium
pytest tests/e2e/ -v --browser chromium
```

### Dependencies

**Core:**
- `numpy`, `scipy`, `networkx` -- Numerical computation and graph algorithms
- `numba` -- JIT compilation for performance-critical loops
- `plotly` -- Interactive visualization (CLI and web)
- `streamlit` -- Web interface framework
- `pydantic` -- Data validation and models

**CLI:**
- `click` -- CLI framework
- `rich` -- Terminal formatting
- `tqdm` -- Progress bars

**Data:**
- `pandas`, `openpyxl` -- DataFrame operations and Excel export
- `joblib` -- Parallel experiment execution
- `humanize` -- Human-readable formatting
- `psutil` -- Process management
- `openai` -- AI analysis integration

---

## Performance Tips

### Choosing Sample Sizes

- **Quick exploration**: 20-50 samples
- **Reliable results**: 100-200 samples
- **Publication quality**: 500+ samples with 5-10 runs

More samples give smoother curves and better statistics, but take longer to compute.

### Network Size Considerations

- **< 1,000 nodes**: Very fast, good for testing
- **1,000-10,000 nodes**: Typical research use
- **> 10,000 nodes**: Consider using GPU acceleration

### First Run is Slow

The first time you run SPKMC, Numba compiles the performance-critical functions to machine code. This takes a few seconds. Subsequent runs will be much faster.

If you want to clear this cache (for example, after updating SPKMC):
```bash
spkmc clean --numba-cache
```

### Parallel Execution

When running experiments with multiple scenarios, SPKMC automatically runs them in parallel using all available CPU cores. You don't need to configure this manually.

---

## Global Options

These options work with any command:

| Option | Description |
|--------|-------------|
| `--version` | Display the SPKMC version |
| `-v, --verbose` | Show detailed progress and debug information |
| `--analyze` | Run AI-powered analysis of results (requires `OPENAI_API_KEY` environment variable) |
| `--help` | Show help for any command |

**Examples:**

```bash
# Check your installed version
spkmc --version

# Run with verbose output for debugging
spkmc -v run -n er -d gamma

# Get help for a specific command
spkmc run --help
```

---

## Troubleshooting

### "Command not found: spkmc"

Make sure you installed SPKMC with `pip install spkmc` and that your Python environment is activated.

### Plots don't display

If you're running over SSH or in an environment without a display, use `--no-plot` to skip visualization. You can later use `spkmc plot results.json -o figure.png` to save plots to files.

### Out of memory errors

Try reducing the network size (`--nodes`) or number of samples (`--samples`).

### Slow first run

This is normal - Numba is compiling optimized code. Subsequent runs will be faster.

---

## License

MIT License - feel free to use SPKMC in your research and applications.

## Citation

If you use SPKMC in your research, please cite the original algorithm paper:

```bibtex
@article{tolic2018simulating,
  title = {Simulating SIR processes on networks using weighted shortest paths},
  author = {Toli{\'c}, Dijana and Kleineberg, Kaj-Kolja and Antulov-Fantulin, Nino},
  journal = {Scientific Reports},
  volume = {8},
  number = {1},
  pages = {6562},
  year = {2018},
  publisher = {Nature Publishing Group},
  doi = {10.1038/s41598-018-24648-w}
}
```

And optionally, this software implementation:

```bibtex
@software{spkmc,
  title = {SPKMC: Shortest Path Kinetic Monte Carlo for Epidemic Simulation},
  author = {Castro, Marcus},
  url = {https://github.com/mcaxtr/spkmc}
}
```

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
