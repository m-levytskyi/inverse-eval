#!/usr/bin/env python3
"""
Bridge script to evaluate pickle predictions using the reflectorch pipeline.

This script:
1. Loads pickle predictions (5 parameters, excluding sld_fronting)
2. Converts them to reflectorch batch_results format
3. Uses constraint-based prior bounds with proper clipping
4. Calculates constraint-based MAPE using reflectorch error_calculation
5. Generates overall + parameter-specific MAPE histograms

Pickle file: results_exp_1L_fitconstraints0_width0.3_simple.pkl
- 1 layer experiments
- 30% constraint-based priors (width=0.3)
- No SLD fixing (fitconstraints=0, all parameters used)
- Should match batch_results_143.json setup
"""

import pickle
import logging
import numpy as np
from pathlib import Path

from error_calculation import calculate_parameter_metrics
from plotting_utils import plot_batch_mape_distribution, plot_batch_parameter_breakdown


logger = logging.getLogger(__name__)


def load_pickle_data(pickle_file="results_exp_1L_fitconstraints0_width0.3_simple.pkl"):
    """Load pickle prediction data."""
    logger.info(f"Loading pickle file: {pickle_file}")
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    targets = data[0]  # True values (3169, 6)
    predictions = data[1]  # Predictions (3169, 6)
    indices = data[2]  # Manifest indices
    fitconstraints = data[3]  # 0 = all parameters
    width = data[4]  # 0.3 = 30% constraint-based
    bounds_flags = data[5]  # Out of bounds indicator

    logger.info(f"  Loaded {len(indices)} experiments")
    logger.info(f"  fitconstraints: {fitconstraints} (0=all params)")
    logger.info(f"  width: {width} (constraint deviation)")
    logger.info(f"  Out of bounds: {np.sum(bounds_flags)} experiments")

    return targets, predictions, indices, bounds_flags, width


def load_manifest(manifest_file="manifest_exp_1L.pkl"):
    """Load manifest to get experiment IDs."""
    logger.info(f"Loading manifest: {manifest_file}")
    with open(manifest_file, "rb") as f:
        manifest = pickle.load(f)

    samples = manifest["samples"]
    logger.info(f"  Loaded {len(samples)} manifest entries")

    return samples


def convert_pickle_to_batch_results(
    targets, predictions, indices, bounds_flags, manifest_samples, width=0.3
):
    """
    Convert pickle predictions to reflectorch batch_results format.

    Note: We only use 5 parameters (excluding sld_fronting) to match reflectorch.
    """
    logger.info("\nConverting pickle predictions to batch_results format...")

    batch_results = {}
    outlier_count = 0

    # Reflectorch parameter names for 5 parameters (lowercase as used in error_calculation.py)
    # Order matches batch_results_143.json: thickness, amb_rough, sub_rough, layer_sld, sub_sld
    reflectorch_param_names = [
        "thickness",
        "amb_rough",
        "sub_rough",
        "layer_sld",
        "sub_sld",
    ]

    # Pickle order: sld_fronting, roughness_fronting, sld_1, thickness_1, roughness_1, sld_backing
    # Map to reflectorch order: thickness_1, roughness_fronting, roughness_1, sld_1, sld_backing
    pickle_to_reflectorch_indices = [3, 1, 4, 2, 5]  # Skip index 0 (sld_fronting)

    for pkl_pos in range(len(indices)):
        manifest_idx = indices[pkl_pos]

        if manifest_idx >= len(manifest_samples):
            continue

        experiment_id = manifest_samples[manifest_idx]["base_id"]

        # Check if out of bounds - mark as outlier
        is_outlier = bool(bounds_flags[pkl_pos] == 1)

        if is_outlier:
            outlier_count += 1
            # Skip outliers - don't add to batch_results
            continue

        # Extract 5 parameters (skip sld_fronting)
        true_vals_5 = [targets[pkl_pos][i] for i in pickle_to_reflectorch_indices]
        pred_vals_5 = [predictions[pkl_pos][i] for i in pickle_to_reflectorch_indices]

        # Convert SLD values from Å⁻² to 10⁻⁶ Å⁻² (reflectometry standard unit)
        # Indices 3 and 4 in reflectorch order are layer_sld and sub_sld
        true_vals_5[3] *= 1e6  # layer_sld
        true_vals_5[4] *= 1e6  # sub_sld
        pred_vals_5[3] *= 1e6  # layer_sld
        pred_vals_5[4] *= 1e6  # sub_sld

        # Calculate regular and constraint-based parameter MAPE.
        param_metrics = calculate_parameter_metrics(
            pred_params=pred_vals_5,
            true_params=true_vals_5,
            param_names=reflectorch_param_names,
        )

        # Build result entry in reflectorch format
        batch_results[experiment_id] = {
            "experiment_id": experiment_id,
            "success": True,
            "param_metrics": param_metrics,
            "priors_config": {
                "priors_type": "constraint_based",
                "priors_deviation": width,
                "fix_sld_mode": "none",
            },
            "layer_count": 1,
            # Add pickle-specific metadata
            "_pickle_metadata": {
                "manifest_index": int(manifest_idx),
                "pickle_position": int(pkl_pos),
                "was_outlier": is_outlier,
            },
        }

        if (pkl_pos + 1) % 500 == 0:
            logger.info(f"  Processed {pkl_pos + 1}/{len(indices)} experiments...")

    logger.info("\nConversion complete:")
    logger.info(f"  Total experiments in pickle: {len(indices)}")
    logger.info(f"  Outliers (excluded): {outlier_count}")
    logger.info(f"  Valid experiments: {len(batch_results)}")

    return batch_results, outlier_count


def main():
    """Main execution function."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("=" * 80)
    logger.info("PICKLE PREDICTIONS EVALUATION")
    logger.info("Using Reflectorch Pipeline with Constraint-Based Priors")
    logger.info("=" * 80)

    # Load data
    targets, predictions, indices, bounds_flags, width = load_pickle_data()
    manifest_samples = load_manifest()

    # Convert to batch_results format
    batch_results, outlier_count = convert_pickle_to_batch_results(
        targets, predictions, indices, bounds_flags, manifest_samples, width
    )

    if not batch_results:
        logger.error("No valid experiments to evaluate")
        return

    # Generate plots using reflectorch plotting utilities
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING MAPE DISTRIBUTION PLOTS")
    logger.info("=" * 80)

    output_dir = Path(".")

    # 1. Overall MAPE histogram
    logger.info("\n1. Overall MAPE Distribution...")
    plot_path_overall = plot_batch_mape_distribution(
        batch_results=batch_results,
        layer_count=1,
        output_dir=str(output_dir),
        save=True,
        narrow_priors_deviation=width,
        use_prominent_features=False,
        failed_count=0,
        outlier_count=outlier_count,
    )

    if plot_path_overall:
        logger.info(f"   Saved to: {plot_path_overall}")

    # 2. Parameter-specific MAPE breakdown
    logger.info("\n2. Parameter-Specific MAPE Breakdown...")
    plot_path_params = plot_batch_parameter_breakdown(
        batch_results=batch_results,
        layer_count=1,
        output_dir=str(output_dir),
        save=True,
        narrow_priors_deviation=width,
        use_prominent_features=False,
        failed_count=0,
        outlier_count=outlier_count,
    )

    if plot_path_params:
        logger.info(f"   Saved to: {plot_path_params}")

    # Print summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 80)

    constraint_mapes = []
    param_stats = {"thickness": [], "roughness": [], "sld": []}

    for result in batch_results.values():
        if "param_metrics" in result and result["param_metrics"]:
            pm = result["param_metrics"]

            # Overall constraint MAPE
            if "overall" in pm and "constraint_mape" in pm["overall"]:
                constraint_mapes.append(pm["overall"]["constraint_mape"])

            # Parameter-specific constraint MAPEs
            if "by_type" in pm:
                for param_type in ["thickness", "roughness", "sld"]:
                    if (
                        param_type in pm["by_type"]
                        and "constraint_mape" in pm["by_type"][param_type]
                    ):
                        param_stats[param_type].append(
                            pm["by_type"][param_type]["constraint_mape"]
                        )

    if constraint_mapes:
        logger.info("\nOverall Constraint-Based MAPE Statistics:")
        logger.info(f"  Total experiments: {len(constraint_mapes)}")
        logger.info(f"  Mean: {np.mean(constraint_mapes):.2f}%")
        logger.info(f"  Median: {np.median(constraint_mapes):.2f}%")
        logger.info(f"  Std Dev: {np.std(constraint_mapes):.2f}%")
        logger.info(f"  Min: {np.min(constraint_mapes):.2f}%")
        logger.info(f"  Max: {np.max(constraint_mapes):.2f}%")

        # Distribution
        excellent = sum(1 for m in constraint_mapes if m < 5)
        good = sum(1 for m in constraint_mapes if 5 <= m < 10)
        acceptable = sum(1 for m in constraint_mapes if 10 <= m < 20)
        poor = sum(1 for m in constraint_mapes if m >= 20)

        total = len(constraint_mapes)
        logger.info("\n  Distribution:")
        logger.info(
            f"    Excellent (< 5%):     {excellent:4d} ({excellent / total * 100:5.1f}%)"
        )
        logger.info(f"    Good (5-10%):         {good:4d} ({good / total * 100:5.1f}%)")
        logger.info(
            f"    Acceptable (10-20%):  {acceptable:4d} ({acceptable / total * 100:5.1f}%)"
        )
        logger.info(f"    Poor (≥ 20%):         {poor:4d} ({poor / total * 100:5.1f}%)")

    # Parameter-specific statistics
    logger.info("\nParameter-Specific Constraint-Based MAPE:")
    for param_type, values in param_stats.items():
        if values:
            logger.info(f"\n  {param_type.upper()}:")
            logger.info(f"    Count: {len(values)}")
            logger.info(f"    Mean: {np.mean(values):.2f}%")
            logger.info(f"    Median: {np.median(values):.2f}%")
            logger.info(f"    Std: {np.std(values):.2f}%")
            logger.info(f"    Range: [{np.min(values):.2f}%, {np.max(values):.2f}%]")

    logger.info("\n" + "=" * 80)
    logger.info("Evaluation complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
