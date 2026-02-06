#!/usr/bin/env python3
"""
Evaluate model performance against random guessing baseline.

This script loads batch inference results and compares model predictions
against random predictions that are consistent with constraints and prior bounds.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from constraints_utils import get_constraint_range


def generate_random_prediction(
    param_names: List[str], prior_bounds: List[Tuple[float, float]]
) -> np.ndarray:
    """
    Generate random parameter predictions within prior bounds.

    Args:
        param_names: List of parameter names
        prior_bounds: List of (min, max) tuples for each parameter

    Returns:
        Array of random parameter values
    """
    random_params = []
    for i, (param_name, (min_val, max_val)) in enumerate(
        zip(param_names, prior_bounds)
    ):
        # Generate uniform random value within bounds
        random_val = np.random.uniform(min_val, max_val)
        random_params.append(random_val)

    return np.array(random_params)


def calculate_mape_for_prediction(
    pred_params: np.ndarray,
    true_params: np.ndarray,
    param_names: List[str],
    priors_type: str = None,
) -> float:
    """
    Calculate MAPE for a single prediction.

    Args:
        pred_params: Predicted parameter values
        true_params: True parameter values
        param_names: List of parameter names
        priors_type: Type of priors used ("constraint_based" or other)

    Returns:
        MAPE value as percentage
    """
    if len(pred_params) != len(true_params):
        return -1.0

    errors = pred_params - true_params

    # Calculate percentage errors, handling true zeros
    zero_mask = np.abs(true_params) < 1e-10
    percentage_errors = np.zeros_like(errors)
    nonzero_mask = ~zero_mask

    if np.any(nonzero_mask):
        percentage_errors[nonzero_mask] = (
            np.abs(errors[nonzero_mask] / true_params[nonzero_mask]) * 100
        )
    if np.any(zero_mask):
        percentage_errors[zero_mask] = np.abs(errors[zero_mask])

    # For constraint-based priors, calculate constraint-based MAPE
    if priors_type == "constraint_based":
        from constraints_utils import get_constraint_width

        constraint_based_percentage_errors = np.zeros_like(errors)

        for i in range(len(errors)):
            param_name = param_names[i]
            # Standardize parameter name to match constraint definitions
            standardized_name = standardize_param_name(param_name)
            try:
                constraint_width = get_constraint_width(standardized_name)
                constraint_based_percentage_errors[i] = (
                    np.abs(errors[i]) / constraint_width * 100
                )
            except KeyError:
                print(
                    f"Warning: Could not find constraint width for {param_name} (standardized: {standardized_name})"
                )
                constraint_based_percentage_errors[i] = percentage_errors[i]

        return float(np.mean(constraint_based_percentage_errors))
    else:
        return float(np.mean(percentage_errors))


def standardize_param_name(param_name: str) -> str:
    """
    Standardize parameter names to match constraint definitions.

    Args:
        param_name: Original parameter name (e.g., "Thickness L1", "SLD sub")

    Returns:
        Standardized parameter name for constraint lookup
    """
    param_lower = param_name.lower()

    # Thickness parameters
    if "thickness" in param_lower:
        if "l1" in param_lower or "layer 1" in param_lower:
            return "thickness1" if "l2" in param_lower else "thickness"
        elif "l2" in param_lower or "layer 2" in param_lower:
            return "thickness2"
        else:
            return "thickness"

    # Roughness parameters
    if "rough" in param_lower:
        if "amb" in param_lower or "front" in param_lower:
            return "amb_rough"
        elif "sub" in param_lower or "back" in param_lower:
            return "sub_rough"
        elif "int" in param_lower or "interface" in param_lower:
            return "int_rough"
        else:
            return "sub_rough"

    # SLD parameters
    if "sld" in param_lower:
        if "amb" in param_lower or "front" in param_lower:
            return "amb_sld"
        elif "sub" in param_lower or "back" in param_lower:
            return "sub_sld"
        elif "l1" in param_lower or "layer 1" in param_lower:
            return "layer1_sld" if "l2" in param_lower else "layer_sld"
        elif "l2" in param_lower or "layer 2" in param_lower:
            return "layer2_sld"
        else:
            return "layer_sld"

    return param_name


def load_batch_results(batch_dir: Path) -> Dict:
    """
    Load batch results from JSON file.

    Args:
        batch_dir: Path to batch directory

    Returns:
        Dictionary with batch results
    """
    results_file = batch_dir / "batch_results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file, "r") as f:
        return json.load(f)


def evaluate_batch_against_random(batch_dir: Path) -> Tuple[List[float], List[float]]:
    """
    Evaluate a batch against random guessing baseline.

    Args:
        batch_dir: Path to batch directory

    Returns:
        Tuple of (model_mapes, random_mapes)
    """
    print(f"\nProcessing batch: {batch_dir.name}")

    batch_results = load_batch_results(batch_dir)

    # Filter successful experiments
    successful_results = {
        k: v for k, v in batch_results.items() if v.get("success", False)
    }

    if not successful_results:
        print(f"  No successful experiments in batch")
        return [], []

    print(f"  Found {len(successful_results)} successful experiments")

    model_mapes = []
    random_mapes = []

    # Determine priors type from first successful result
    priors_type = None
    for result in successful_results.values():
        if "priors_config" in result:
            priors_type = result["priors_config"].get("priors_type")
            break

    print(f"  Priors type: {priors_type}")

    for exp_id, result in successful_results.items():
        # Get model MAPE
        if "param_metrics" in result and result["param_metrics"]:
            param_metrics = result["param_metrics"]

            if priors_type == "constraint_based":
                if (
                    "overall" in param_metrics
                    and "constraint_mape" in param_metrics["overall"]
                ):
                    model_mape = param_metrics["overall"]["constraint_mape"]
                    model_mapes.append(model_mape)
            else:
                if "overall" in param_metrics and "mape" in param_metrics["overall"]:
                    model_mape = param_metrics["overall"]["mape"]
                    model_mapes.append(model_mape)

        # Get true parameters and param names
        if "true_params_dict" not in result or "prediction_dict" not in result:
            continue

        true_params_dict = result["true_params_dict"]
        param_names = result["prediction_dict"]["param_names"]
        prior_bounds = result.get("prior_bounds", [])

        if not prior_bounds:
            continue

        # Extract true params from nested structure
        layer_count = result.get("layer_count", 1)
        layer_key = f"{layer_count}_layer"

        if layer_key not in true_params_dict:
            continue

        true_params = np.array(true_params_dict[layer_key]["params"])

        # Generate ONE random prediction per experiment
        random_pred = generate_random_prediction(param_names, prior_bounds)
        random_mape = calculate_mape_for_prediction(
            random_pred, true_params, param_names, priors_type
        )
        if random_mape >= 0:
            random_mapes.append(random_mape)

    print(f"  Collected {len(model_mapes)} model MAPEs")
    print(f"  Collected {len(random_mapes)} random MAPEs")

    return model_mapes, random_mapes


def plot_comparison_histogram(
    model_mapes: List[float],
    random_mapes: List[float],
    batch_name: str,
    output_path: Path,
    priors_type: str = None,
    layer_count: int = 1,
    narrow_priors_deviation: float = 0.99,
    use_prominent_features: bool = False,
    fix_sld_mode: str = "none",
    failed_count: int = 0,
    outlier_count: int = 0,
    thesis_mode: bool = False,
):
    """
    Create comparison histogram with model and random-guess MAPEs.

    Args:
        model_mapes: List of model MAPE values
        random_mapes: List of random-guess MAPE values
        batch_name: Name of the batch for plot title
        output_path: Path to save the plot
        priors_type: Type of priors used
        layer_count: Number of layers
        narrow_priors_deviation: Deviation for narrow priors
        use_prominent_features: Whether prominent features filtering was used
        fix_sld_mode: SLD fixing mode
        failed_count: Number of failed experiments
        outlier_count: Number of outlier experiments
        thesis_mode: Use thesis styling if True
    """
    if thesis_mode:
        from thesis_plotting_utils import thesis_style

        with thesis_style():
            fig = _create_comparison_histogram(
                model_mapes,
                random_mapes,
                batch_name,
                priors_type,
                layer_count,
                narrow_priors_deviation,
                use_prominent_features,
                fix_sld_mode,
                failed_count,
                outlier_count,
            )
            output_path = Path(output_path).with_suffix(".pdf")
            plt.savefig(output_path)
            plt.close()
    else:
        fig = _create_comparison_histogram(
            model_mapes,
            random_mapes,
            batch_name,
            priors_type,
            layer_count,
            narrow_priors_deviation,
            use_prominent_features,
            fix_sld_mode,
            failed_count,
            outlier_count,
        )
        plt.savefig(output_path)
        plt.close()

    print(f"Comparison plot saved to: {output_path}")


def _create_comparison_histogram(
    model_mapes,
    random_mapes,
    batch_name,
    priors_type,
    layer_count,
    narrow_priors_deviation,
    use_prominent_features,
    fix_sld_mode,
    failed_count,
    outlier_count,
):
    """Internal function to create comparison histogram."""
    # Determine MAPE type label
    mape_type_label = (
        "Constraint-Based MAPE" if priors_type == "constraint_based" else "MAPE"
    )
    mape_label = "Constraint MAPE" if priors_type == "constraint_based" else "MAPE"

    # Create title with all configuration information (matching original)
    title_parts = []
    if fix_sld_mode != "none":
        title_parts.append(f"SLD fix: {fix_sld_mode}")
    if use_prominent_features:
        title_parts.append("Prominent Features")

    title_suffix = f" ({', '.join(title_parts)})" if title_parts else ""

    # Create distribution plot (matching original layout)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle(
        f"{mape_type_label} Distribution - {len(model_mapes)} {layer_count}-Layer Experiments{title_suffix}\n"
        f"(Narrow Priors ±{int(narrow_priors_deviation * 100)}%)",
        fontsize=16,
        fontweight="bold",
    )

    # Define MAPE ranges - fixed 5% bins from 0-100% (matching original)
    mape_ranges = list(range(0, 105, 5))
    range_labels = [f"{i}-{i + 5}%" for i in range(0, 100, 5)]

    # Count model MAPEs in each range
    model_counts = []
    for i in range(len(mape_ranges) - 1):
        count = sum(
            1 for mape in model_mapes if mape_ranges[i] <= mape < mape_ranges[i + 1]
        )
        model_counts.append(count)

    # Count random MAPEs in each range
    random_counts = []
    for i in range(len(mape_ranges) - 1):
        count = sum(
            1 for mape in random_mapes if mape_ranges[i] <= mape < mape_ranges[i + 1]
        )
        random_counts.append(count)

    # Create bar chart
    bars = ax.bar(range(len(model_counts)), model_counts, alpha=0.8)

    # Overlay random-guess histogram as semi-transparent
    ax.bar(range(len(random_counts)), random_counts, alpha=0.3, linewidth=1.5)

    # Add value labels on model bars (matching original)
    for i, (bar, count) in enumerate(zip(bars, model_counts)):
        if count > 0:
            percentage = (count / len(model_mapes)) * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{count}\n({percentage:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    ax.set_xlabel("MAPE Range")
    ax.set_ylabel("Number of Experiments")
    ax.set_xticks(range(len(range_labels)))
    ax.set_xticklabels(range_labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    # Calculate random guessing mean
    random_mean = np.mean(random_mapes) if random_mapes else 0

    # Add statistics text with outlier and failure counts (matching original layout)
    if model_mapes:
        stats_text = f"Total: {len(model_mapes)} experiments\n"
        stats_text += f"Mean {mape_label}: {np.mean(model_mapes):.1f}%\n"
        stats_text += f"Median {mape_label}: {np.median(model_mapes):.1f}%\n"
        stats_text += f"Std Dev: {np.std(model_mapes):.1f}%\n"
        stats_text += f"Min {mape_label}: {np.min(model_mapes):.1f}%\n"
        stats_text += f"Max {mape_label}: {np.max(model_mapes):.1f}%\n"
        stats_text += f"\nRandom Guessing Mean: {random_mean:.1f}%"

        # Add outlier and failure information if present
        if outlier_count > 0 or failed_count > 0:
            stats_text += f"\n\n--- Excluded ---"
            if outlier_count > 0:
                stats_text += f"\nOutliers: {outlier_count}"
            if failed_count > 0:
                stats_text += f"\nFailed: {failed_count}"

        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", alpha=0.8),
        )

    plt.tight_layout()
    return fig


def process_batches(batch_numbers: List[int]):
    """
    Process multiple batches and generate comparison plots.

    Args:
        batch_numbers: List of batch numbers to process
    """
    base_dir = Path("batch_inference_results")
    output_dir = Path("random_guessing_evaluation")
    output_dir.mkdir(exist_ok=True)

    for batch_num in batch_numbers:
        # Find batch directory
        batch_dirs = list(base_dir.glob(f"{batch_num}_*"))

        if not batch_dirs:
            print(f"Batch {batch_num} not found")
            continue

        if len(batch_dirs) > 1:
            print(
                f"Warning: Multiple directories found for batch {batch_num}, using first"
            )

        batch_dir = batch_dirs[0]

        try:
            # Evaluate batch
            model_mapes, random_mapes = evaluate_batch_against_random(batch_dir)

            if not model_mapes or not random_mapes:
                print(f"  Skipping batch {batch_num} - no valid data")
                continue

            # Load batch results to get configuration details
            batch_results = load_batch_results(batch_dir)

            # Get configuration from first successful result
            priors_type = None
            fix_sld_mode = "none"
            narrow_priors_deviation = 0.99
            use_prominent_features = "PROMINENT" in batch_dir.name
            layer_count = 1

            # Count outliers and failures
            outlier_count = sum(
                1 for v in batch_results.values() if v.get("excluded_as_outlier", False)
            )
            failed_count = sum(
                1
                for v in batch_results.values()
                if not v.get("success", False)
                and not v.get("excluded_as_outlier", False)
            )

            for result in batch_results.values():
                if result.get("success") and "priors_config" in result:
                    priors_type = result["priors_config"].get("priors_type")
                    fix_sld_mode = result["priors_config"].get("fix_sld_mode", "none")
                    layer_count = result.get("layer_count", 1)

                    # Extract narrow_priors_deviation from batch name
                    if "5constraint" in batch_dir.name:
                        narrow_priors_deviation = 0.05
                    elif "30constraint" in batch_dir.name:
                        narrow_priors_deviation = 0.30
                    elif "60constraint" in batch_dir.name:
                        narrow_priors_deviation = 0.60
                    elif "99constraint" in batch_dir.name:
                        narrow_priors_deviation = 0.99

                    break

            # Generate comparison plot with original layout
            output_path = output_dir / f"batch_{batch_num:03d}_comparison.png"
            plot_comparison_histogram(
                model_mapes,
                random_mapes,
                batch_dir.name,
                output_path,
                priors_type,
                layer_count,
                narrow_priors_deviation,
                use_prominent_features,
                fix_sld_mode,
                failed_count,
                outlier_count,
            )

        except Exception as e:
            print(f"  Error processing batch {batch_num}: {e}")
            continue

    print(f"\nEvaluation complete! Plots saved to: {output_dir}")


def generate_synthetic_random_evaluation(
    num_experiments: int = 3000,
    layer_count: int = 1,
    deviation: float = 0.99,
    output_path: Path = None,
):
    """
    Generate synthetic evaluation where BOTH true values and predictions are random.

    This simulates truly uninformed random guessing where we don't even know
    the true parameters - everything is sampled from constraint intervals.

    Args:
        num_experiments: Number of synthetic experiments to generate
        layer_count: Number of layers (1 or 2)
        deviation: Prior deviation from true value (0.99 = 99% constraint-based)
        output_path: Path to save the plot
    """
    print(f"\n{'=' * 80}")
    print(f"SYNTHETIC RANDOM EVALUATION")
    print(f"{'=' * 80}")
    print(f"Generating {num_experiments} synthetic experiments")
    print(f"Layer count: {layer_count}")
    print(f"Prior deviation: ±{int(deviation * 100)}%")
    print()

    # Import prior bounds generation from main pipeline
    from parameter_discovery import (
        get_parameter_names_for_layer_count,
        get_constraint_based_prior_bounds,
    )

    # Get parameter names for this layer count
    param_names = get_parameter_names_for_layer_count(layer_count)

    print(f"Parameters: {param_names}")
    print()

    # Get constraint ranges for reference
    constraint_ranges = []
    constraint_widths = []

    for param_name in param_names:
        standardized = standardize_param_name(param_name)
        min_val, max_val = get_constraint_range(standardized)
        width = max_val - min_val
        constraint_ranges.append((min_val, max_val))
        constraint_widths.append(width)
        print(
            f"  {param_name:20s} -> {standardized:15s}: [{min_val:7.2f}, {max_val:7.2f}] (width: {width:.2f})"
        )

    print()
    print("Generating random true values and predictions...")
    print(
        "Using get_constraint_based_prior_bounds() to ensure physical constraints are respected"
    )
    print()

    mapes = []
    per_param_mapes = {param_name: [] for param_name in param_names}

    for i in range(num_experiments):
        # Generate random TRUE values from full constraint ranges
        true_params = np.array(
            [
                np.random.uniform(min_val, max_val)
                for min_val, max_val in constraint_ranges
            ]
        )

        # Create true_params_dict in the format expected by get_constraint_based_prior_bounds
        layer_key = f"{layer_count}_layer"
        true_params_dict = {
            layer_key: {"params": true_params.tolist(), "param_names": param_names}
        }

        # Use the same prior bounds calculation as the main pipeline
        # This ensures physical constraints are properly applied
        prior_bounds = get_constraint_based_prior_bounds(
            true_params_dict, layer_count, deviation
        )

        # Generate random PREDICTION from prior bounds
        pred_params = generate_random_prediction(param_names, prior_bounds)

        # Calculate constraint-based MAPE (overall)
        mape = calculate_mape_for_prediction(
            pred_params, true_params, param_names, "constraint_based"
        )

        if mape >= 0:
            mapes.append(mape)

            # Calculate per-parameter MAPEs
            for j, param_name in enumerate(param_names):
                standardized = standardize_param_name(param_name)
                constraint_min, constraint_max = get_constraint_range(standardized)
                constraint_width = constraint_max - constraint_min

                param_error = abs(pred_params[j] - true_params[j])
                param_mape = (param_error / constraint_width) * 100
                per_param_mapes[param_name].append(param_mape)

    print(f"Generated {len(mapes)} valid MAPE values")
    print(f"Mean MAPE: {np.mean(mapes):.2f}%")
    print(f"Median MAPE: {np.median(mapes):.2f}%")
    print(f"Std Dev: {np.std(mapes):.2f}%")
    print(f"Min: {np.min(mapes):.2f}%")
    print(f"Max: {np.max(mapes):.2f}%")
    print()

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    title = f"Synthetic Random - {num_experiments} {layer_count}-Layer Experiments\n"
    title += f"(±{int(deviation * 100)}% Constraint-Based Priors, Both True & Predicted Random)"
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Define MAPE ranges
    mape_ranges = list(range(0, 105, 5))
    range_labels = [f"{i}-{i + 5}%" for i in range(0, 100, 5)]

    # Count MAPEs in each range
    counts = []
    for i in range(len(mape_ranges) - 1):
        count = sum(1 for mape in mapes if mape_ranges[i] <= mape < mape_ranges[i + 1])
        counts.append(count)

    # Create bar chart
    bars = ax.bar(range(len(counts)), counts, alpha=0.8)

    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        if count > 0:
            percentage = (count / len(mapes)) * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{count}\n({percentage:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    ax.set_xlabel("Constraint-Based MAPE Range", fontsize=14)
    ax.set_ylabel("Number of Experiments", fontsize=14)
    ax.set_xticks(range(len(range_labels)))
    ax.set_xticklabels(range_labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    # Add statistics text
    stats_text = f"Total: {len(mapes)} experiments\n"
    stats_text += f"Mean Constraint MAPE: {np.mean(mapes):.1f}%\n"
    stats_text += f"Median Constraint MAPE: {np.median(mapes):.1f}%\n"
    stats_text += f"Std Dev: {np.std(mapes):.1f}%\n"
    stats_text += f"Min Constraint MAPE: {np.min(mapes):.1f}%\n"
    stats_text += f"Max Constraint MAPE: {np.max(mapes):.1f}%\n"

    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", alpha=0.8),
    )

    plt.tight_layout()

    # Save plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {output_path}")

    plt.close(fig)

    # Create per-parameter MAPE histograms
    print("\nGenerating per-parameter MAPE distributions...")
    n_params = len(param_names)
    fig_params, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    fig_params.suptitle(
        f"Per-Parameter MAPE Distributions - Synthetic Random Evaluation\n"
        f"{num_experiments} {layer_count}-Layer Experiments (±{int(deviation * 100)}% Priors)",
        fontsize=16,
        fontweight="bold",
    )

    for idx, param_name in enumerate(param_names):
        if idx >= len(axes):
            break

        ax = axes[idx]
        param_mapes = per_param_mapes[param_name]

        # Get constraint info for theoretical expectation
        standardized = standardize_param_name(param_name)
        constraint_min, constraint_max = get_constraint_range(standardized)
        constraint_width = constraint_max - constraint_min
        theoretical_mean = 100 / 3  # For uniform distribution

        # Create histogram
        n_bins = 50
        counts, bins, patches = ax.hist(param_mapes, bins=n_bins, alpha=0.7)

        # Add theoretical mean line
        ax.axvline(
            theoretical_mean,
            linestyle="--",
            linewidth=2,
            label=f"Theoretical: {theoretical_mean:.1f}%",
        )

        # Add actual mean line
        actual_mean = np.mean(param_mapes)
        ax.axvline(
            actual_mean,
            linestyle="--",
            linewidth=2,
            label=f"Actual: {actual_mean:.1f}%",
        )

        # Styling
        ax.set_xlabel("Constraint-Based MAPE (%)", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title(
            f"{param_name}\n"
            f"Constraint: [{constraint_min:.1f}, {constraint_max:.1f}] (width: {constraint_width:.1f})",
            fontsize=11,
            fontweight="bold",
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Add statistics box
        stats = f"Mean: {actual_mean:.1f}%\n"
        stats += f"Median: {np.median(param_mapes):.1f}%\n"
        stats += f"Std: {np.std(param_mapes):.1f}%"
        ax.text(
            0.98,
            0.98,
            stats,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.8),
        )

    # Hide unused subplots
    for idx in range(len(param_names), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    # Save per-parameter plot
    if output_path:
        per_param_path = (
            output_path.parent / f"{output_path.stem}_per_parameter{output_path.suffix}"
        )
        plt.savefig(per_param_path, dpi=300, bbox_inches="tight")
        print(f"Saved per-parameter plot: {per_param_path}")

    plt.close(fig_params)

    # Print per-parameter statistics
    print("\nPer-Parameter MAPE Statistics:")
    print("=" * 80)
    print(
        f"{'Parameter':<20} {'Mean':<10} {'Median':<10} {'Std Dev':<10} {'Theoretical':<12}"
    )
    print("=" * 80)
    for param_name in param_names:
        param_mapes = per_param_mapes[param_name]
        print(
            f"{param_name:<20} {np.mean(param_mapes):>8.2f}%  {np.median(param_mapes):>8.2f}%  "
            f"{np.std(param_mapes):>8.2f}%  {100 / 3:>10.2f}%"
        )
    print("=" * 80)

    print(f"\n{'=' * 80}")
    print("INTERPRETATION:")
    print(f"{'=' * 80}")
    print(f"This represents the WORST CASE scenario - completely uninformed guessing")
    print(f"where we don't even know the true parameter values.")
    print(f"")
    print(f"Expected mean MAPE ≈ {100 / 3:.1f}% for uniform random (theoretical)")
    print(f"Actual overall mean MAPE = {np.mean(mapes):.1f}%")
    print(f"")
    print(f"Each parameter should follow a triangular distribution with mean ≈ 33.33%")
    print(f"when both true and predicted values are uniformly sampled.")
    print(f"")
    print(f"Any model or informed random guessing that performs better than")
    print(f"{np.mean(mapes):.1f}% demonstrates knowledge of the true parameters!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    import sys

    # Check if synthetic random evaluation is requested
    if len(sys.argv) > 1 and sys.argv[1] == "--synthetic":
        print(
            "Running synthetic random evaluation (both true and predicted are random)"
        )
        from pathlib import Path

        # Create output directory
        output_dir = Path("random_guessing_evaluation")
        output_dir.mkdir(exist_ok=True)

        # Generate synthetic random evaluation
        generate_synthetic_random_evaluation(
            num_experiments=3000,
            layer_count=1,
            deviation=0.99,
            output_path=output_dir / "synthetic_random_vs_random.png",
        )
    elif len(sys.argv) > 1:
        # Process specific batch numbers provided as arguments
        batch_numbers = []
        for arg in sys.argv[1:]:
            try:
                batch_num = int(arg)
                batch_numbers.append(batch_num)
            except ValueError:
                print(f"Warning: Ignoring invalid batch number '{arg}'")

        if batch_numbers:
            if len(batch_numbers) == 1:
                print(f"Evaluating batch {batch_numbers[0]} against random guessing")
            else:
                print(f"Evaluating batches {batch_numbers} against random guessing")
            print(f"Generating ONE random prediction per experiment\n")

            process_batches(batch_numbers)
        else:
            print("No valid batch numbers provided!")
    else:
        # Default: Process batches 102 to 119 inclusive
        batch_numbers = list(range(102, 120))

        print(
            f"Evaluating batches {batch_numbers[0]} to {batch_numbers[-1]} against random guessing"
        )
        print(f"Generating ONE random prediction per experiment\n")

        process_batches(batch_numbers)
