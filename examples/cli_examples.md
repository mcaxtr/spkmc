# SPKMC CLI Usage Examples

This document contains detailed examples of how to use the SPKMC command-line interface (CLI) for different simulation scenarios.

## Installation

Before using the CLI, install the dependencies:

```bash
pip install -r requirements.txt
```

Or install the package in development mode:

```bash
pip install -e .
```

## Basic Simulation with an Erdos-Renyi Network and Gamma Distribution

An Erdos-Renyi network is a random network model where each pair of nodes has the same probability of being connected. The Gamma distribution is often used to model recovery times in epidemics.

### Basic Example

```bash
python spkmc_cli.py run -n er -d gamma --shape 2.0 --scale 1.0 --lambda 1.0 -N 1000 --k-avg 10 -s 50
```

This command runs a simulation with:
- Erdos-Renyi network (`-n er`)
- Gamma distribution (`-d gamma`) with shape (`--shape 2.0`) and scale (`--scale 1.0`) parameters
- 1000 nodes (`-N 1000`)
- Average degree of 10 (`--k-avg 10`)
- 50 samples per run (`-s 50`)
- Lambda parameter for infection times (`--lambda 1.0`)

### Parameter Variations

#### Varying Network Size

```bash
# Small network (500 nodes)
python spkmc_cli.py run -n er -d gamma --shape 2.0 --scale 1.0 -N 500 --k-avg 10 -s 50

# Medium network (2000 nodes)
python spkmc_cli.py run -n er -d gamma --shape 2.0 --scale 1.0 -N 2000 --k-avg 10 -s 50

# Large network (5000 nodes)
python spkmc_cli.py run -n er -d gamma --shape 2.0 --scale 1.0 -N 5000 --k-avg 10 -s 50
```

#### Varying Average Degree

```bash
# Low average degree (5)
python spkmc_cli.py run -n er -d gamma --shape 2.0 --scale 1.0 -N 1000 --k-avg 5 -s 50

# Medium average degree (15)
python spkmc_cli.py run -n er -d gamma --shape 2.0 --scale 1.0 -N 1000 --k-avg 15 -s 50

# High average degree (30)
python spkmc_cli.py run -n er -d gamma --shape 2.0 --scale 1.0 -N 1000 --k-avg 30 -s 50
```

#### Varying Gamma Distribution Parameters

```bash
# Low shape (1.0)
python spkmc_cli.py run -n er -d gamma --shape 1.0 --scale 1.0 -N 1000 --k-avg 10 -s 50

# High shape (3.0)
python spkmc_cli.py run -n er -d gamma --shape 3.0 --scale 1.0 -N 1000 --k-avg 10 -s 50

# Low scale (0.5)
python spkmc_cli.py run -n er -d gamma --shape 2.0 --scale 0.5 -N 1000 --k-avg 10 -s 50

# High scale (2.0)
python spkmc_cli.py run -n er -d gamma --shape 2.0 --scale 2.0 -N 1000 --k-avg 10 -s 50
```

### Saving Results

```bash
python spkmc_cli.py run -n er -d gamma --shape 2.0 --scale 1.0 -N 1000 --k-avg 10 -s 50 -o results/er_gamma.json
```

## Simulation with a Complex Network and Exponential Distribution

Complex networks follow a power-law degree distribution, which makes them more realistic for modeling many real-world networks. The exponential distribution is often used to model times of random events.

### Basic Example

```bash
python spkmc_cli.py run -n cn -d exponential --mu 1.0 --lambda 1.0 --exponent 2.5 -N 1000 --k-avg 10 -s 50
```

This command runs a simulation with:
- Complex network (`-n cn`)
- Exponential distribution (`-d exponential`) with mu (`--mu 1.0`) and lambda (`--lambda 1.0`) parameters
- Power-law exponent (`--exponent 2.5`)
- 1000 nodes (`-N 1000`)
- Average degree of 10 (`--k-avg 10`)
- 50 samples per run (`-s 50`)

### Parameter Variations

#### Varying the Power-Law Exponent

```bash
# Low exponent (2.1) - more hubs, longer tail
python spkmc_cli.py run -n cn -d exponential --mu 1.0 --lambda 1.0 --exponent 2.1 -N 1000 --k-avg 10 -s 50

# Medium exponent (2.5) - typical for many real networks
python spkmc_cli.py run -n cn -d exponential --mu 1.0 --lambda 1.0 --exponent 2.5 -N 1000 --k-avg 10 -s 50

# High exponent (3.0) - fewer hubs, more homogeneous
python spkmc_cli.py run -n cn -d exponential --mu 1.0 --lambda 1.0 --exponent 3.0 -N 1000 --k-avg 10 -s 50
```

#### Varying Exponential Distribution Parameters

```bash
# Low mu (0.5) - slower recovery
python spkmc_cli.py run -n cn -d exponential --mu 0.5 --lambda 1.0 --exponent 2.5 -N 1000 --k-avg 10 -s 50

# High mu (2.0) - faster recovery
python spkmc_cli.py run -n cn -d exponential --mu 2.0 --lambda 1.0 --exponent 2.5 -N 1000 --k-avg 10 -s 50

# Low lambda (0.5) - slower infection
python spkmc_cli.py run -n cn -d exponential --mu 1.0 --lambda 0.5 --exponent 2.5 -N 1000 --k-avg 10 -s 50

# High lambda (2.0) - faster infection
python spkmc_cli.py run -n cn -d exponential --mu 1.0 --lambda 2.0 --exponent 2.5 -N 1000 --k-avg 10 -s 50
```

### Saving Results

```bash
python spkmc_cli.py run -n cn -d exponential --mu 1.0 --lambda 1.0 --exponent 2.5 -N 1000 --k-avg 10 -s 50 -o results/cn_exponential.json
```

## Simulation with a Random Regular Network (RRN)

A random regular network is one where all nodes have exactly the same number of connections (degree), but those connections are distributed randomly. This creates a degree-homogeneous network with random structure.

### Basic Example

```bash
python spkmc_cli.py run -n rrn -d gamma --shape 2.0 --scale 1.0 --lambda 1.0 -N 1000 --k-avg 10 -s 50
```

This command runs a simulation with:
- Random regular network (`-n rrn`)
- Gamma distribution (`-d gamma`) with shape (`--shape 2.0`) and scale (`--scale 1.0`) parameters
- 1000 nodes (`-N 1000`)
- Regular degree of 10 (`--k-avg 10`) - must be an even number
- 50 samples per run (`-s 50`)
- Lambda parameter for infection times (`--lambda 1.0`)

### Parameter Variations

#### Varying Regular Degree

```bash
# Low regular degree (4)
python spkmc_cli.py run -n rrn -d gamma --shape 2.0 --scale 1.0 -N 1000 --k-avg 4 -s 50

# Medium regular degree (10)
python spkmc_cli.py run -n rrn -d gamma --shape 2.0 --scale 1.0 -N 1000 --k-avg 10 -s 50

# High regular degree (20)
python spkmc_cli.py run -n rrn -d gamma --shape 2.0 --scale 1.0 -N 1000 --k-avg 20 -s 50
```

### Saving Results

```bash
python spkmc_cli.py run -n rrn -d gamma --shape 2.0 --scale 1.0 -N 1000 --k-avg 10 -s 50 -o results/rrn_gamma.json
```

## Comparison Between Different Network Types

To compare different network types, run simulations for each type and then use the `compare` command.

### Running Simulations for Different Networks

```bash
# Erdos-Renyi network
python spkmc_cli.py run -n er -d gamma --shape 2.0 --scale 1.0 -N 1000 --k-avg 10 -s 50 -o results/er_gamma.json

# Complex network
python spkmc_cli.py run -n cn -d gamma --shape 2.0 --scale 1.0 --exponent 2.5 -N 1000 --k-avg 10 -s 50 -o results/cn_gamma.json

# Complete graph
python spkmc_cli.py run -n cg -d gamma --shape 2.0 --scale 1.0 -N 500 -s 50 -o results/cg_gamma.json

# Random regular network
python spkmc_cli.py run -n rrn -d gamma --shape 2.0 --scale 1.0 -N 1000 --k-avg 10 -s 50 -o results/rrn_gamma.json
```
