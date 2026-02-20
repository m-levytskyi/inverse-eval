# Reflectorch Evaluation Pipeline

A research evaluation pipeline for neutron/X-ray reflectometry parameter estimation using normalizing flow (NF) neural networks. This repository forms part of a master's thesis on probabilistic machine learning methods for thin-film analysis.

The pipeline evaluates the custom **nflows_reflectorch** package — an extension of [reflectorch](https://github.com/schreiber-lab/reflectorch) that adds normalizing flows support and several advanced input transformations — against the MARIA instrument dataset from the MLZ (Maier-Leibnitz Zentrum) neutron source.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Notebooks (Quick Start)](#notebooks-quick-start)
- [Dataset Setup](#dataset-setup)
- [Configuration](#configuration)
- [Running Experiments](#running-experiments)
- [Reproducing Thesis Figures](#reproducing-thesis-figures)
- [Module Reference](#module-reference)

---

## Overview

### What This Pipeline Does

1. **Loads** experimental neutron reflectometry curves (Q vs. R with error bars dR)
2. **Preprocesses** the data (filters high-error points that cause tensor errors)
3. **Constructs constraint-based prior bounds** from the true structure parameters
4. **Runs inference** using an NF model trained via the `nflows_reflectorch` package
5. **Evaluates** predictions against true parameters using Constraint MAPE
6. **Generates publication-quality plots** matching the thesis figure style

### Key Physical Quantities

| Symbol | Quantity | Unit |
|--------|----------|------|
| Q | Momentum transfer | Å⁻¹ |
| R | Reflectivity | dimensionless |
| dR | Reflectivity uncertainty | dimensionless |
| SLD | Scattering length density | 10⁻⁶ Å⁻² |
| d | Layer thickness | Å |
| σ | Interfacial roughness | Å |

---

## Repository Structure

```
evaluation_pipeline/
├── README.md                     # This file
├── Pipfile / Pipfile.lock        # Pipenv environment specification
├── requirements.txt              # pip-compatible dependency list
├── config.py                     # Centralized path and parameter configuration
├── model_constraints.json        # Physical constraint bounds (per parameter type)
├── paper.mplstyle                # Matplotlib style matching thesis typography
│
├── Core Pipeline
│   ├── simple_pipeline.py        # Single-experiment inference workflow
│   ├── batch_pipeline.py         # Batch orchestration with CLI interface
│   └── batch_sweep_runner.py     # Automated parameter sweep over prior configs
│
├── Analysis & Evaluation
│   ├── batch_analysis.py         # Summary statistics and distribution analysis
│   ├── error_calculation.py      # MAPE, constraint MAPE, metric utilities
│   ├── evaluate_random_guessing.py # Baseline comparison vs. random predictions
│   ├── evaluate_pickle_predictions.py # Load and re-evaluate saved predictions
│   ├── nf_statistics.py          # NF sample statistics (mean, std, coverage)
│   └── compare_model_versions.py # Side-by-side comparison of two model variants
│
├── Data & Preprocessing
│   ├── data_preprocessing.py     # Error-bar filtering, data cleaning
│   ├── parameter_discovery.py    # Experiment file discovery and true-param parsing
│   ├── parameter_constraints.py  # Physical constraint application
│   ├── constraints_utils.py      # Constraint range and width helpers
│   └── find_prominent_peaks.py   # Peak detection for prominent-feature subsets
│
├── Plotting
│   ├── plotting_utils.py         # All publication-quality plot functions
│   ├── plot_mape_vs_std.py       # MAPE vs. posterior std scatter and coverage
│   ├── replot_batch_results.py   # Regenerate plots from saved batch JSON
│   ├── compare_sld_profiles.py   # SLD profile comparison utilities
│   └── sld_profile_utils.py      # SLD profile generation helpers
│
├── Utilities
│   └── utils.py                  # JSON serialization, directory helpers
│
├── Configuration
│   └── sweep_configs/            # YAML configs for automated sweeps
│       ├── baseline.yaml         # NF baseline model sweep
│       ├── qweighted.yaml        # Q-weighted input transformation sweep
│       ├── calibration.yaml      # Calibration analysis sweep
│       └── ...
│
├── Results
│   ├── batch_inference_results/  # Timestamped batch output directories
│   └── paper_batches/            # Curated batches used in thesis figures
│
└── notebooks/
    ├── 01_single_experiment.ipynb   # Single-experiment walkthrough
    └── 02_batch_inference.ipynb     # Batch processing demonstration
```

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10-3.12 |
| make | any |
| Git | any |
| CUDA (optional) | 11.8+ (for GPU acceleration) |
| LaTeX (optional) | for `text.usetex = True` in plots |

---

## Installation

### 1. Clone this repository

```bash
git clone <this-repo-url>
cd evaluation_pipeline
```

### 2. Set up the environment

```bash
make setup && make check-torch
```

This creates a local `.venv`, clones and installs `nflows_reflectorch` (editable), installs PyTorch (CUDA 11.8 by default), and installs all remaining dependencies.

If `check-torch` reports a CUDA error, rerun with a different wheel:

```bash
# CUDA 12.1
make setup TORCH_WHEEL=cu121 && make check-torch

# CPU only
make setup TORCH_WHEEL=cpu && make check-torch
```

### 3. Activate the environment

```bash
source .venv/bin/activate
```

---

## Notebooks (Quick Start)

The notebooks are the recommended starting point for exploring this pipeline:

| Notebook | Description |
|----------|-------------|
| [`notebooks/01_single_experiment.ipynb`](notebooks/01_single_experiment.ipynb) | Step-by-step walkthrough of a single inference run — load data, build priors, run the NF model, visualise the fit and SLD profile |
| [`notebooks/02_batch_inference.ipynb`](notebooks/02_batch_inference.ipynb) | Batch processing demonstration — run inference over many experiments and analyse the results |

Launch with:

```bash
jupyter lab notebooks/
```

---

## Dataset Setup

The dataset is included in the repository root under `dataset/`. By default, `config.py` points to:

```python
DATA_DIRECTORY = "dataset/test"
```

### Dataset Layout

Experiment IDs use the format `sXXXXXX` (zero-padded six-digit integer), e.g. `s000780`.
Each experiment consists of three files stored flat inside the split directory:

```
dataset/
├── test/
│   ├── s000004_experimental_curve.dat   # Q, R, dR columns (experimental)
│   ├── s000004_theoretical_curve.dat    # Q, R columns (simulated ground truth)
│   ├── s000004_model.txt                # True structural parameters
│   ├── s000014_experimental_curve.dat
│   ├── s000014_model.txt
│   └── ...
└── train/
    ├── s000001_experimental_curve.dat
    └── ...
```

---

## Configuration

All default paths and hyperparameters live in `config.py`. Override specific values per-run via CLI flags.

### Key configuration values

| Setting | Default | Description |
|---------|---------|-------------|
| `DEFAULT_NF_CONFIG` | `example_nf_config_reflectorch.yaml` | NF model YAML config |
| `DEFAULT_NF_SAMPLES` | `1000` | Posterior samples per experiment |
| `DEFAULT_PRIORS_TYPE` | `constraint_based` | Prior construction method |
| `DEFAULT_PRIORS_DEVIATION` | `0.30` | Constraint half-width (30% of true value) |
| `DEFAULT_SLD_MODE` | `none` | SLD fixing: `none`, `backing`, or `all` |
| `DEFAULT_LAYER_COUNT` | `1` | Target film structure |

### Sweep configuration files

YAML files in `sweep_configs/` define automated multi-run parameter sweeps:

| File | Model | Description |
|------|-------|-------------|
| `baseline.yaml` | NF baseline | Default generated-data training |
| `qweighted.yaml` | NF + Q-weighted + dR | Mixed data, Q-weighted inputs |
| `q_exp1_2.yaml` | NF + Q-weighted α=2,β=2 | Standard Fresnel weighting |
| `q_exp2.yaml` | NF + Q-weighted α=4,β=4 | Strong Fresnel weighting |
| `mean.yaml` | NF + mean conditioned | Mean-centered curve scaling |
| `calibration.yaml` | Calibration analysis | MAPE vs. posterior std |

---

## Running Experiments

### Single experiment

```python
from simple_pipeline import run_single_experiment

results = run_single_experiment(
    experiment_id="s000004",
    enable_preprocessing=True,
)
print(results["param_metrics"]["overall"]["constraint_mape"])
```

### Single batch (CLI)

```bash
python batch_pipeline.py \
  --layer-count 1 \
  --num-experiments 100 \
  --priors-type constraint_based \
  --priors-deviation 30 \
  --fix-sld-mode none \
  --inference-backend nf \
  --config-name example_nf_config_reflectorch.yaml \
  --nf-num-samples 1000
```

Key flags:

| Flag | Values | Description |
|------|--------|-------------|
| `--layer-count` | `0`, `1`, `2` | Film layer count |
| `--num-experiments` | integer or omit | Experiments to process (omit = all) |
| `--priors-deviation` | `5`, `30`, `99` | Constraint width as % of true value |
| `--fix-sld-mode` | `none`, `backing`, `all` | Fix SLD parameters from ground truth |
| `--use-prominent-features` | flag | Restrict to high-contrast curve subset |
| `--inference-backend` | `nf` | Inference backend |
| `--config-name` | filename | NF model YAML config file |
| `--use-sigmas-input` | flag | Pass dR as additional model input |

### Full parameter sweep

```bash
python batch_sweep_runner.py --config sweep_configs/baseline.yaml
```

This runs `len(prior_deviations) × len(sld_modes) × len(prominent_features)` pipeline instances automatically (18 runs with default settings). Results are saved to `sweep_results_nf_baseline/sweep_<timestamp>/`.

To run the Q-weighted model sweep:

```bash
python batch_sweep_runner.py --config sweep_configs/qweighted.yaml
```

### Re-generate plots from saved results

```bash
python replot_batch_results.py --batch-id 075 --base-dir batch_inference_results
```

---

## Reproducing Thesis Figures

All figures are produced using `plotting_utils.py` with `paper.mplstyle` applied globally. Figures are exported as PDF.

### Prerequisites

Completed batch inference runs are required. The relevant batch groups used in the thesis are stored in `paper_batches/`.

---

### Figure 1 — MAPE Histograms (per-model)

**What it shows**: Distribution of Constraint MAPE across all test experiments for each parameter type (thickness, roughness, SLD). One histogram per model variant.

**Configuration used**: 30% constraint prior, no SLD fixing, no prominent-feature filtering.

> **Note**: Each run processes the full test split (~900 experiments) and takes approximately **15 minutes**.

**How to generate** (run once per model):

```bash
# NF Baseline
python batch_pipeline.py \
  --data-directory dataset/test \
  --priors-type constraint_based \
  --priors-deviation 30 \
  --fix-sld-mode none \
  --inference-backend nf \
  --config-name example_nf_config_reflectorch.yaml \
  --nf-num-samples 1000

# NF + Q-Weighted + dR
python batch_pipeline.py \
  --data-directory dataset/test \
  --priors-type constraint_based \
  --priors-deviation 30 \
  --fix-sld-mode none \
  --inference-backend nf \
  --config-name nf_config_mixed_sigmas_qweighted.yaml \
  --nf-num-samples 1000 \
  --use-sigmas-input

# NF + Mean Conditioned
python batch_pipeline.py \
  --data-directory dataset/test \
  --priors-type constraint_based \
  --priors-deviation 30 \
  --fix-sld-mode none \
  --inference-backend nf \
  --config-name nf_config_mixed_mean_conditioned.yaml \
  --nf-num-samples 1000
```

After each run, regenerate the plots:

```bash
python replot_batch_results.py --results-dir batch_inference_results/<batch_dir>
```

Or programmatically:

```python
import json
from plotting_utils import create_batch_analysis_plots

with open("batch_inference_results/<batch_dir>/batch_results.json") as f:
    batch_results = json.load(f)

successful = {k: v for k, v in batch_results.items() if v["success"]}
create_batch_analysis_plots(successful, layer_count=1, output_dir="figures/", save=True)
```

This produces `mape_distribution_1layer.pdf` and `parameter_breakdown_1layer.pdf`.

---

### Figure 2 — Coverage Plot

**What it shows**: Fraction of experiments where the true parameter falls within the posterior credible interval at each confidence level (ideal = identity line).

**How to generate**:

```bash
python plot_mape_vs_std.py --batch-dir batch_inference_results/<batch_dir>
```

Or from Python:

```python
from plot_mape_vs_std import plot_coverage, compute_coverage_data

coverage = compute_coverage_data("batch_inference_results/<batch_dir>/batch_results.json")
plot_coverage(coverage, output_dir="figures/")
```

---

### Figure 3 — Single Reflectivity Curve Fit

**What it shows**: Experimental reflectivity curve overlaid with the model prediction and uncertainty band, plus the inferred SLD depth profile. To reproduce the thesis figure, choose an experiment whose curve displays a visually prominent oscillation peak.

**How to generate**:

```python
from simple_pipeline import run_single_experiment
from plotting_utils import plot_simple_comparison

# Use an experiment ID with a prominent oscillation peak, e.g. s000780
results = run_single_experiment(
    experiment_id="s000780",
    enable_preprocessing=True,
)

plot_simple_comparison(
    results,
    output_path="figures/single_curve_s000780.pdf",
    save=True,
)
```

---

### Figure 4 — Evaluation Against Random Guessing

**What it shows**: MAPE distribution of the NF model overlaid with a random-guessing baseline that samples uniformly from the prior bounds.

**How to generate**:

```bash
python evaluate_random_guessing.py \
  --batch-dir batch_inference_results/<batch_dir> \
  --num-random-samples 1000 \
  --output-dir figures/
```

Or from Python:

```python
from evaluate_random_guessing import run_random_guessing_evaluation
from plotting_utils import plot_random_guessing_comparison

model_results, random_results = run_random_guessing_evaluation(
    batch_dir="batch_inference_results/<batch_dir>",
    num_random_samples=1000,
)

plot_random_guessing_comparison(
    model_results=model_results,
    random_results=random_results,
    output_path="figures/random_guessing.pdf",
)
```

---

## Module Reference

### `simple_pipeline.py`

Core single-experiment workflow.

| Function | Description |
|----------|-------------|
| `run_single_experiment(experiment_id, layer_count, ...)` | Full pipeline for one experiment; returns metrics dict |
| `load_experimental_data(path, enable_preprocessing, ...)` | Load Q/R/dR from `.dat` file with optional filtering |
| `run_inference(model, q, curve, prior_bounds, ...)` | Execute NF inference, return prediction dict |

### `batch_pipeline.py`

```bash
python batch_pipeline.py [OPTIONS]
```

| Class / Function | Description |
|-----------------|-------------|
| `BatchInferencePipeline` | Orchestrates multi-experiment processing |
| `.run_batch_inference()` | Main entry point; discovers experiments, runs pipeline, saves JSON |

### `batch_sweep_runner.py`

```bash
python batch_sweep_runner.py --config sweep_configs/baseline.yaml
```

| Class / Function | Description |
|-----------------|-------------|
| `BatchPipelineSweep` | Iterates all (prior_deviation × sld_mode × prominent) combinations |
| `.run_full_sweep()` | Runs all combinations, saves summary JSON |

### `plotting_utils.py`

All plotting is publication-ready (PDF output, LaTeX labels, `paper.mplstyle`).

| Function | Output File | Description |
|----------|-------------|-------------|
| `plot_simple_comparison(results, ...)` | `comparison_*.pdf` | Single experiment: curve + SLD profile |
| `plot_batch_mape_distribution(batch_results, ...)` | `mape_distribution_*.pdf` | MAPE histogram over all experiments |
| `plot_batch_parameter_breakdown(batch_results, ...)` | `parameter_breakdown_*.pdf` | Per-parameter MAPE breakdown |
| `plot_model_comparison_histogram(...)` | user-specified | Overlay two model MAPE distributions |
| `plot_random_guessing_comparison(...)` | user-specified | NF model vs. random baseline |
| `plot_parameter_comparison_grid(...)` | user-specified | Per-parameter grid comparison |
| `create_batch_analysis_plots(batch_results, ...)` | multiple | Convenience wrapper for all batch plots |

### `data_preprocessing.py`

Filters experimental data to prevent tensor dimension mismatches during inference.

Steps applied:
1. Remove negative R values
2. Truncate at first consecutive high-error run (configurable threshold and window)
3. Optionally remove isolated high-error singleton points

### `parameter_discovery.py`

Discovers experiment files, parses true structural parameters from `model.dat`, and builds constraint-based prior bounds.

### `error_calculation.py`

Calculates MAPE variants:
- **Standard MAPE**: `|pred - true| / |true| × 100`
- **Constraint MAPE**: `|pred - true| / constraint_width × 100` — normalises by prior half-width rather than true value, making it interpretable across different parameter scales

### `find_prominent_peaks.py`

Identifies experiments with prominent oscillation features using peak detection on the reflectivity curve. Used to create higher-difficulty evaluation subsets.


