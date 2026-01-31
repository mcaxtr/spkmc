# SPKMC - Shortest Path Kinetic Monte Carlo

An efficient implementation of the Shortest Path Kinetic Monte Carlo (SPKMC) algorithm to simulate epidemic spread on complex networks using the SIR (Susceptible-Infected-Recovered) model.

## Description

SPKMC is a stochastic algorithm for simulating epidemic spread on complex networks. This implementation supports multiple network types (Erdos-Renyi, Complex Networks with power-law degree distribution, Complete Graphs, Random Regular Networks) and different probability distributions (Gamma, Exponential) for recovery and infection times.

SPKMC uses shortest-path concepts to compute epidemic propagation times between nodes, making it more computationally efficient than traditional Monte Carlo methods on large networks.

### Key Features

- Epidemic simulations on multiple network types
- Support for multiple probability distributions
- Efficient shortest-path computation using optimized algorithms
- Full-featured command-line interface (CLI)
- Result visualization with informative plots
- Comparison across multiple simulations
- Result export in JSON, CSV, Excel, Markdown, and HTML

## Installation

### Requirements

- Python 3.8 or newer
- Basic command-line knowledge

### Quick Install

```bash
pip install -e .
```

### Optional GPU Support

To enable GPU acceleration with CUDA, install optional GPU dependencies:

```bash
pip install -e ".[gpu]"
```

**GPU requirements:**
- NVIDIA driver installed
- Compatible CUDA Toolkit (12.x)

If you use CUDA 11.x, install a compatible CuPy version manually.

## CLI Usage

The `spkmc` CLI provides commands to run simulations and visualize results.

```bash
spkmc --help
```

Example:

```bash
spkmc run -n er -d gamma --shape 2.0 --scale 1.0 -N 1000 --k-avg 10 -s 50
```

For detailed CLI examples, see `examples/cli_examples.md`.

## Programmatic Usage

```python
from spkmc import SPKMC, GammaDistribution, NetworkFactory
import numpy as np

# Distribution and simulator
gamma_dist = GammaDistribution(shape=2.0, scale=1.0, lmbd=1.0)
simulator = SPKMC(gamma_dist)

# Network and time steps
G = NetworkFactory.create_erdos_renyi(N=1000, k_avg=10)
time_steps = np.linspace(0, 10.0, 100)

# Run simulation
S, I, R = simulator.run_multiple_simulations(G, sources=np.array([0]), time_steps=time_steps, samples=50)
```

## Documentation

- `docs/usage.md` - Full usage guide
- `docs/architecture.md` - Architecture overview

## Examples

- `examples/basic_example.py`
- `examples/basic_simulation.py`
- `examples/cli_examples.md`

## License

MIT License
