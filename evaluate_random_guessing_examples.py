#!/usr/bin/env python3
"""
Quick reference guide for random guessing evaluation.

This script provides example usage patterns for the random guessing evaluation
functionality.
"""

# ============================================================================
# BASIC USAGE
# ============================================================================

# Run evaluation for batches 102-119 (default)
# python evaluate_random_guessing.py

# ============================================================================
# PROGRAMMATIC USAGE
# ============================================================================

from evaluate_random_guessing import (
    evaluate_batch_against_random,
    plot_comparison_histogram,
    process_batches,
)
from pathlib import Path

# Example 1: Evaluate a single batch
# ------------------------------------
batch_dir = Path(
    "batch_inference_results/102_3169exps_1layers_5constraint_29october2025_12_25"
)
model_mapes, random_mapes = evaluate_batch_against_random(
    batch_dir, num_random_samples=100
)
print(f"Model mean MAPE: {sum(model_mapes) / len(model_mapes):.2f}%")
print(f"Random mean MAPE: {sum(random_mapes) / len(random_mapes):.2f}%")

# Example 2: Process specific batches
# ------------------------------------
# Process only batches 102, 105, and 110
process_batches([102, 105, 110], num_random_samples=100)

# Example 3: Process with more random samples for better statistics
# ------------------------------------------------------------------
process_batches(range(102, 120), num_random_samples=500)

# ============================================================================
# OUTPUT STRUCTURE
# ============================================================================

# Generated plots are saved to: random_guessing_evaluation/
# - batch_102_comparison.png
# - batch_103_comparison.png
# - ...
# - batch_119_comparison.png

# Each plot contains:
# 1. Histogram of model MAPE values (colored bars)
# 2. Overlay histogram of random MAPE values (semi-transparent grey)
# 3. Statistics table showing mean, median, std dev for both
# 4. Legend with summary information

# ============================================================================
# CUSTOMIZATION
# ============================================================================

# To change output directory:
# Edit the 'output_dir' variable in process_batches() function

# To adjust histogram bins:
# Edit 'mape_ranges' in plot_comparison_histogram() function

# To change number of random samples per experiment:
# Pass num_random_samples parameter to process_batches()

# ============================================================================
# UNDERSTANDING THE METRICS
# ============================================================================

# MAPE (Mean Absolute Percentage Error):
# - Standard MAPE: |predicted - true| / |true| * 100
# - Constraint-based MAPE: |predicted - true| / constraint_width * 100
#
# For constraint-based priors, the constraint-based MAPE is used to
# provide a bounded error metric that scales with the prior width.

# Random predictions:
# - Sampled uniformly from prior bounds for each parameter
# - Respect physical constraints defined in model_constraints.json
# - Provide a baseline for "uninformed guessing"

# ============================================================================
# INTERPRETATION
# ============================================================================

# Good model performance:
# - Model MAPE << Random MAPE
# - Model MAPE concentrated in low ranges (0-10%)
# - Clear separation between model and random distributions

# Poor model performance:
# - Model MAPE ≈ Random MAPE
# - Overlapping distributions
# - Model not learning beyond random guessing
