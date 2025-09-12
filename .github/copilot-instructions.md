# Reflectorch API Playground - AI Agent Instructions

## Project Overview
This is a research playground for neutron/X-ray reflectometry analysis using the **Reflectorch** machine learning package. The project focuses on analyzing experimental reflectivity data to determine scattering length density (SLD) profiles and structural parameters of thin films.

## Environment & Dependencies
- **Package Management**: Use Pipenv (`pipenv install`, `pipenv shell`)
- **Python Version**: 3.12 
- **Key Dependencies**: reflectorch, torch, numpy, scipy, pandas, matplotlib, seaborn
- **Reflectorch Package**: Core ML package for reflectometry analysis - https://github.com/schreiber-lab/reflectorch

## Core Architecture & Data Flow

### Pipeline Structure
1. **Data Input**: Experimental reflectivity curves (Q vs intensity with error bars)
2. **Preprocessing**: Filter high-error data points to prevent tensor concatenation errors
3. **Model Configuration**: YAML configs define neural network architecture and prior bounds
4. **Inference**: Use EasyInferenceModel for parameter estimation
5. **Output**: SLD profiles, fitted parameters, and comparison plots

### Key Components
- `simple_pipeline.py`: Core single-experiment processing workflow
- `batch_pipeline.py`: Orchestrates multiple experiments with BatchInferencePipeline class
- `parameter_discovery.py`: Experiment file discovery and true parameter parsing
- `data_preprocessing.py`: Error bar filtering (critical for tensor operations)
- `plotting_utils.py`: Visualization utilities for reflectivity curves and SLD profiles

## Critical Development Patterns

### Data Structure Convention
```python
# Standard reflectivity data format
q_exp = data[..., 0]      # Momentum transfer (Å⁻¹)
curve_exp = data[..., 1]  # Reflectivity intensity
sigmas_exp = data[..., 2] # Error bars/uncertainties
```

### Output Organization
Results automatically organize into timestamped directories:
```
batch_inference_results/
  {N}experiments_{layers}_layer_{timestamp}/
    plots/
    batch_results.json
```

## MARIA Dataset Handling
**CRITICAL**: Never attempt to read the MARIA_VIPR_dataset folder directly (10,001 experiments).
- Use existing analysis results: `maria_dataset_summary.json`, `maria_dataset_*.json`
- For new analysis, write scripts or use terminal commands
- Layer counts: 0-2 layers (3340 with 2 layers, 3492 with 0 layers, 3169 with 1 layer)

## Common Workflows

### Single Experiment Analysis
```python
from simple_pipeline import run_single_experiment
results = run_single_experiment(experiment_id, layer_count, enable_preprocessing=True)
```

### Batch Processing
```python
from batch_pipeline import BatchInferencePipeline
pipeline = BatchInferencePipeline(num_experiments=10, layer_count=1)
pipeline.run_batch_inference()
```

### Custom Analysis
Use existing JSON analysis results rather than parsing raw data files directly.

## Code Guidelines

### Development Principles
- **Minimal Changes**: Make as few modifications as possible - prefer reusing existing scripts
- **Script Reuse**: Always check for existing utilities before creating new ones
- **No Emojis**: Never use emoji characters in code, comments, or outputs
- **Leverage Existing Pipeline**: Use `simple_pipeline.py` and `batch_pipeline.py` functions rather than reimplementing
- **Modular Approach**: Utilize existing modules (`parameter_discovery.py`, `data_preprocessing.py`, `plotting_utils.py`) for their specific functions