# SPKMC Architecture

This document describes the architecture of the SPKMC package, explaining the main components and their interactions.

## Overview

SPKMC (Shortest Path Kinetic Monte Carlo) is an algorithm for simulating epidemic spread on networks using the SIR (Susceptible-Infected-Recovered) model. The implementation is based on classes and interfaces that enable simulation across different network types and probability distributions.

## Project Structure

```
spkmc/
├── cli/                # CLI module
│   ├── __init__.py
│   ├── __main__.py     # Entry point for python -m spkmc.cli
│   ├── commands.py     # CLI commands
│   └── validators.py   # CLI parameter validators
├── core/               # Core module
│   ├── __init__.py
│   ├── distributions.py # Distribution classes
│   ├── networks.py     # Network classes
│   └── simulation.py   # SPKMC class
├── io/                 # Input/output module
│   ├── __init__.py
│   └── results.py      # Results management
├── utils/              # Utilities
│   ├── __init__.py
│   └── numba_utils.py  # Numba-optimized functions
└── visualization/      # Visualization module
    ├── __init__.py
    └── plots.py        # Visualization functions
```

## Main Components

### `core` Module

The `core` module contains the main classes for the SPKMC algorithm.

#### `distributions.py`

This file contains probability distribution classes used in SPKMC:

- `Distribution`: Abstract class defining the distribution interface.
- `GammaDistribution`: Gamma distribution implementation.
- `ExponentialDistribution`: Exponential distribution implementation.
- `create_distribution`: Factory function to create a distribution instance based on type and parameters.

#### `networks.py`

This file contains classes for creating and manipulating networks:

- `NetworkFactory`: Creates different network types (Erdos-Renyi, Complex, Complete Graph).

#### `simulation.py`

This file contains the main SPKMC algorithm implementation:

- `SPKMC`: Core class that implements the SPKMC algorithm.

### `utils` Module

The `utils` module contains helper functions for the SPKMC algorithm.

#### `numba_utils.py`

This file contains Numba-optimized functions to improve simulation performance:

- `gamma_sampling`: Gamma distribution sampling.
- `get_weight_exponential`: Exponential distribution sampling.
- `compute_infection_times_gamma`: Infection time calculation using Gamma distribution.
- `compute_infection_times_exponential`: Infection time calculation using Exponential distribution.
- `get_states`: Compute states (S, I, R) for each node at a given time.
- `calculate`: Compute the proportion of individuals in each state for each time step.

### `io` Module

The `io` module contains classes for input/output data management.

#### `results.py`

This file contains classes for managing simulation results:

- `ResultManager`: Save and load simulation results.

### `visualization` Module

The `visualization` module contains classes for visualizing results.

#### `plots.py`

This file contains visualization helpers:

- `Visualizer`: Result visualization utilities.

### `cli` Module

The `cli` module implements the command-line interface.

#### `commands.py`

This file contains CLI commands:

- `cli`: Main command group.
- `run`: Run a simulation.
- `plot`: Visualize results.
- `info`: Show information about saved simulations.
- `compare`: Compare results from multiple simulations.

#### `validators.py`

This file contains CLI parameter validators:

- `validate_percentage`: Validate a percentage value.
- `validate_positive`: Validate a positive value.
- `validate_positive_int`: Validate a positive integer.
- `validate_network_type`: Validate the network type.
- `validate_distribution_type`: Validate the distribution type.

## Execution Flow

1. The user creates a distribution instance (`GammaDistribution` or `ExponentialDistribution`).
2. The user creates an `SPKMC` instance with the distribution.
3. The user creates a network using `NetworkFactory`.
4. The user runs the simulation using methods on `SPKMC`.
5. The user visualizes results using `Visualizer`.
6. The user saves results using `ResultManager`.

## Optimizations

SPKMC uses Numba to optimize critical functions. Optimized functions live in `numba_utils.py` and are decorated with `@njit` or `@njit(parallel=True)` for parallelization.

## Command-Line Interface

The SPKMC CLI allows running simulations, visualizing results, and retrieving information directly from the terminal. The CLI is implemented using Click in the `cli` module.

## Extensibility

SPKMC is designed to be extensible. New distributions can be added by implementing the abstract `Distribution` class. New network types can be added by implementing new methods in `NetworkFactory`.
