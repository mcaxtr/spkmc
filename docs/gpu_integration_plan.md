# SPKMC GPU Integration Plan

This document outlines the plan to integrate GPU functionality into the SPKMC project while staying consistent with the existing object-oriented architecture.

## 1. Overview

### General Approach

- Integrate GPU functionality directly into the existing `SPKMC` class
- Add a `use_gpu` parameter to relevant methods
- Make GPU dependencies optional
- Add a global `--gpu` flag in the CLI

### Files to Modify or Create

#### New Files
1. `spkmc/utils/gpu_utils.py` - GPU utility functions

#### Files to Modify
1. `spkmc/core/simulation.py` - Add GPU support to `SPKMC`
2. `spkmc/cli/commands.py` - Add global `--gpu` option
3. `setup.py` - Add GPU dependencies as optional extras
4. `docs/usage.md` - Document GPU acceleration usage

## 2. Detailed Implementation

### 2.1 GPU Utilities Module

Create a new file `spkmc/utils/gpu_utils.py` with GPU dependency checks, availability checks, and GPU-accelerated helpers. Key functions:

- `check_gpu_dependencies()`
- `is_gpu_available()`
- `get_dist_gpu(...)`
- `get_states_gpu(...)`
- `calculate_gpu(...)`

These should be guarded with conditional imports so the package works without GPU dependencies.

### 2.2 SPKMC Class Changes

Update `spkmc/core/simulation.py` to:

- Accept a `use_gpu` parameter
- Detect GPU availability and auto-select GPU based on problem size
- Fall back to CPU if GPU is unavailable or errors occur

### 2.3 CLI Changes

Update `spkmc/cli/commands.py` to add a global `--gpu` flag that enables GPU mode.

### 2.4 Packaging

Expose GPU dependencies as extras (e.g., `pip install spkmc[gpu]`).

### 2.5 Documentation

Document GPU usage, requirements, and optional dependencies in `docs/usage.md`.

## 3. Notes

- GPU acceleration should be optional and never required for core functionality.
- Provide clear user messaging when GPU dependencies are missing or GPU is unavailable.
- Maintain CPU behavior as the default for reliability and compatibility.
