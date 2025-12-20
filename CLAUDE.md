# CLAUDE.md - SPKMC Project Guide

## Project Overview

**SPKMC** (Shortest Path Kinetic Monte Carlo) is a Python implementation for simulating epidemic propagation in complex networks using the SIR (Susceptible-Infected-Recovered) model. The algorithm uses shortest path calculations on weighted graphs to efficiently model disease spread dynamics.

**Version:** 1.0.0
**Python:** 3.8+
**License:** MIT

## Quick Commands

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=spkmc --cov-report=term-missing

# Run a simulation
python -m spkmc.cli run -n er -d gamma --nodes 1000 --samples 50

# Run batch scenarios (interactive experiment menu)
python -m spkmc.cli batch

# List saved results
python -m spkmc.cli info --list
```

## Project Architecture

```
spkmc/
├── spkmc/                    # Main package
│   ├── cli/                  # Command-line interface (Click-based)
│   │   ├── commands.py       # CLI commands: run, plot, info, compare, batch
│   │   ├── validators.py     # Parameter validation callbacks
│   │   └── formatting.py     # Rich terminal output formatting
│   ├── core/                 # Core algorithm implementation
│   │   ├── simulation.py     # SPKMC class - main algorithm
│   │   ├── distributions.py  # Gamma & Exponential distributions
│   │   └── networks.py       # NetworkFactory for graph creation
│   ├── io/                   # Input/output operations
│   │   ├── export.py         # Multi-format export (CSV, JSON, Excel, MD, HTML)
│   │   └── results.py        # Result persistence and loading
│   ├── visualization/        # Plotting
│   │   └── plots.py          # Matplotlib-based SIR curve visualization
│   └── utils/                # Utilities
│       └── numba_utils.py    # Numba JIT-optimized functions
├── tests/                    # Comprehensive test suite
├── docs/                     # Documentation
├── examples/                 # Usage examples
└── experiments/              # Structured experiments with data.json configs
```

## Core Concepts

### Network Types
- `er` - Erdos-Renyi (random networks)
- `cn` - Complex Networks (power-law degree distribution)
- `cg` - Complete Graph (fully connected)
- `rrn` - Random Regular Network (uniform degree)

### Distributions
- `gamma` - Gamma distribution for recovery times (parameters: shape, scale, lambda)
- `exponential` - Exponential distribution (parameters: mu, lambda)

### Algorithm Flow
1. Create network topology via `NetworkFactory`
2. Sample recovery times from distribution
3. Compute infection transmission times for edges
4. Run Dijkstra's algorithm on sparse weighted graph
5. Classify node states (S, I, R) at each timestep
6. Aggregate statistics across samples

## Key Patterns

### Factory Pattern
```python
# Network creation
network = NetworkFactory.create_network(network_type, nodes, k_avg, exponent)

# Distribution creation
dist = create_distribution(dist_type, shape=shape, scale=scale, mu=mu, lambda_param=lambda_param)
```

### Abstract Base Class for Distributions
```python
class Distribution(ABC):
    @abstractmethod
    def get_recovery_weights(self, num_nodes: int) -> np.ndarray: ...

    @abstractmethod
    def get_infection_times(self, weights: np.ndarray) -> np.ndarray: ...
```

### Numba JIT for Performance
Critical loops use `@njit(parallel=True)` decorators for speed:
```python
@njit(parallel=True)
def compute_infection_times_gamma(weights: np.ndarray, shape: float, scale: float) -> np.ndarray:
    ...
```

## Coding Conventions

### Style
- Python 3.8+ with type hints throughout
- Line length: 100 characters (Black formatter)
- snake_case for functions and variables
- PascalCase for classes
- Comprehensive docstrings on public methods

### Import Organization (isort)
1. Standard library
2. Third-party packages
3. Local imports

### Error Handling
- Use Click parameter callbacks for CLI validation
- Raise informative exceptions with context
- Graceful degradation for optional dependencies (openpyxl, pandas)

### Testing
- pytest framework with fixtures
- Test files mirror source structure: `test_<module>.py`
- Integration tests in `test_integration.py`
- Run with: `pytest -v`

## Dependencies

### Core
- `numpy>=1.20.0` - Numerical arrays
- `scipy>=1.7.0` - Sparse graphs, Dijkstra algorithm
- `networkx>=2.6.0` - Graph creation
- `matplotlib>=3.4.0` - Plotting
- `numba>=0.54.0` - JIT compilation

### CLI
- `click>=8.0.0` - CLI framework
- `rich>=10.0.0` - Terminal formatting
- `tqdm>=4.60.0` - Progress bars

### Data
- `pandas>=1.3.0` - DataFrame operations
- `openpyxl>=3.0.7` - Excel export (optional)
- `joblib>=1.0.1` - Parallel batch execution

## CLI Commands

### `run` - Execute simulation
```bash
spkmc run -n <network_type> -d <distribution> [OPTIONS]
  -n, --network-type    Network type (er|cn|cg|rrn)
  -d, --dist-type       Distribution (gamma|exponential)
  -N, --nodes           Number of nodes (default: 1000)
  -s, --samples         Samples per run (default: 50)
  --shape, --scale      Gamma parameters
  --mu, --lambda        Exponential/infection parameters
  -o, --output          Save results to file
  --no-plot             Skip visualization
```

### `plot` - Visualize results
```bash
spkmc plot <result_file> [--save <output>] [--states S,I,R]
```

### `info` - List/inspect results
```bash
spkmc info --list                    # List all results
spkmc info --result-file <path>      # Show specific result
```

### `compare` - Compare multiple runs
```bash
spkmc compare <file1> <file2> ... [-o output]
```

### `batch` - Run multiple scenarios
```bash
spkmc batch                                    # Interactive experiment menu
spkmc batch scenarios.json --output-dir out/   # File mode
```

## Result Storage

Results are stored in hierarchical JSON format:
```
data/spkmc/
  <distribution>/
    <network_type>/
      results_<nodes>_<samples>_<params>.json
```

## Important Files

| File | Purpose |
|------|---------|
| `spkmc/core/simulation.py` | Main SPKMC algorithm implementation |
| `spkmc/core/distributions.py` | Probability distribution classes |
| `spkmc/core/networks.py` | Network topology factory |
| `spkmc/cli/commands.py` | All CLI command definitions |
| `spkmc/utils/numba_utils.py` | Performance-critical JIT functions |
| `experiments/*/data.json` | Experiment configurations |

## Development Guidelines

1. **Never disable lint rules** - Fix the underlying issue instead
2. **Use established libraries** - Rely on NumPy, SciPy, NetworkX patterns
3. **Maintain type hints** - All public functions should be typed
4. **Write tests** - Add tests for new functionality
5. **Keep modules focused** - Separation of concerns between core/cli/io/visualization
6. **Preserve Numba compatibility** - JIT functions have restrictions on Python features

## Performance Considerations

- Use sparse matrices (SciPy CSR) for large networks
- Leverage Numba `@njit(parallel=True)` for loops over nodes/edges
- Avoid Python loops in hot paths - use NumPy vectorization
- NetworkX graphs convert to sparse adjacency matrices for computation

## Validation Callbacks

CLI validators in `spkmc/cli/validators.py`:
- `validate_percentage()` - Ensures 0 <= value <= 1
- `validate_positive()` - Ensures value > 0
- `validate_positive_int()` - Ensures positive integer
- `validate_network_type()` - Validates network type string
- `validate_distribution_type()` - Validates distribution string
