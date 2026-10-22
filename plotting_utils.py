#!/usr/bin/env python3
"""
Plotting utilities for reflectometry analysis.

This module contains all plotting functions used in the reflectometry pipeline,
keeping plotting logic separate from the main inference pipeline.

"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


logger = logging.getLogger(__name__)

# ============================================================================
# SHARED HELPERS
# ============================================================================


def _successful_results(batch_results):
    """Return successful batch results."""
    return {key: value for key, value in batch_results.items() if value.get("success")}


def _get_overall_mape(param_metrics):
    """Extract the overall constraint-based MAPE."""
    return param_metrics.get("overall", {}).get("constraint_mape")


def _get_param_mape(param_data):
    """Extract a constraint-based MAPE value from a parameter entry."""
    return param_data.get("constraint_mape")


def _mape_label():
    """Return the constraint-based MAPE label."""
    return "Constraint MAPE"


def _build_filename(prefix, layer_count, use_prominent_features=False):
    """Build a standard plot filename."""
    parts = [f"{prefix}_{layer_count}layer"]
    if use_prominent_features:
        parts.append("prominent")
    return "_".join(parts) + ".pdf"


def _get_mape_ranges():
    """Return standard MAPE ranges and labels."""
    mape_ranges = list(range(0, 105, 5))
    range_labels = [f"{i}-{i + 5}" for i in range(0, 100, 5)]
    return mape_ranges, range_labels


def _count_mapes_in_ranges(mapes, mape_ranges):
    """Count MAPE values in each range bin."""
    return [
        sum(1 for m in mapes if mape_ranges[i] <= m < mape_ranges[i + 1])
        for i in range(len(mape_ranges) - 1)
    ]


def _build_comparison_title(config):
    """Build title for comparison plots with config-based suffix."""
    mape_type = "Constraint-Based MAPE"

    title_parts = []
    if config.get("sld_fix_mode", "none") != "none":
        mode = config["sld_fix_mode"]
        mode_name = "All SLD Fixed" if mode == "all" else "Backing SLD Fixed"
        title_parts.append(mode_name)
    if config.get("prominent", False):
        title_parts.append("Prominent Features")

    title_suffix = f" ({', '.join(title_parts)})" if title_parts else ""
    deviation_pct = int(config.get("deviation", 0.30) * 100)

    return mape_type, title_suffix, deviation_pct


# SINGLE EXPERIMENT PLOT
# ============================================================================


def plot_simple_comparison(
    q_exp,
    curve_exp,
    sigmas_exp,
    q_model,
    predicted_curve,
    polished_curve,
    predicted_sld_x,
    predicted_sld_y,
    polished_sld_y,
    true_sld_x=None,
    true_sld_y=None,
    experiment_name="Analysis",
    show=True,
    priors_config=None,
    hide_title=False,
):
    """
    Simple plot for single model comparison (used by simple_pipeline).

    Args:
        q_exp: Experimental Q values
        curve_exp: Experimental reflectivity values
        sigmas_exp: Experimental uncertainties
        q_model: Model Q values
        predicted_curve: Predicted reflectivity curve
        polished_curve: Polished reflectivity curve
        predicted_sld_x: SLD profile x-axis
        predicted_sld_y: Predicted SLD profile
        polished_sld_y: Polished SLD profile
        true_sld_x: True SLD profile x-axis (optional)
        true_sld_y: True SLD profile values (optional)
        experiment_name: Name for plot title
        show: Whether to show the plot
        priors_config: Configuration dictionary containing SLD fixing mode
        hide_title: Hide plot titles (for publication)
    """
    # Add SLD fixing mode to the experiment name if available
    display_name = experiment_name
    if priors_config and "fix_sld_mode" in priors_config:
        fix_sld_mode = priors_config["fix_sld_mode"]
        if fix_sld_mode != "none":
            display_name = f"{experiment_name} (SLD fixed: {fix_sld_mode})"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_r, ax_sld = axes

    # --- Reflectivity panel ---
    ax_r.set_yscale("log")
    ax_r.set_xlabel("Q [$\\AA^{-1}$]", fontsize=14)
    ax_r.set_ylabel("R(Q)", fontsize=14)
    ax_r.tick_params(axis="both", which="both", labelsize=12, length=0)

    ax_r.errorbar(
        q_exp,
        curve_exp,
        yerr=sigmas_exp,
        xerr=None,
        elinewidth=1,
        marker="o",
        linestyle="none",
        markersize=3,
        label="Experimental",
        zorder=1,
    )
    ax_r.plot(q_model, predicted_curve, lw=2, label="Predicted")
    ax_r.plot(q_model, polished_curve, ls="--", lw=2, label="Polished")
    ax_r.legend(loc="upper right", fontsize=12)
    if not hide_title:
        ax_r.set_title(f"Reflectivity - {display_name}", fontsize=14)

    # --- SLD profile panel ---
    ax_sld.plot(predicted_sld_x, predicted_sld_y, lw=2, label="Predicted")
    ax_sld.plot(predicted_sld_x, polished_sld_y, ls="--", lw=2, label="Polished")
    if true_sld_x is not None and true_sld_y is not None:
        ax_sld.plot(true_sld_x, true_sld_y, lw=1.5, ls=":", label="True")
    ax_sld.set_xlabel("z [$\\AA$]", fontsize=14)
    ax_sld.set_ylabel("SLD [$10^{-6}$ $\\AA^{-2}$]", fontsize=14)
    ax_sld.tick_params(axis="both", which="both", labelsize=12, length=0)
    ax_sld.legend(loc="upper right", fontsize=12)
    if not hide_title:
        ax_sld.set_title(f"SLD Profile - {display_name}", fontsize=14)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


# ============================================================================
# BATCH PLOTS
# ============================================================================


def plot_batch_mape_distribution(
    batch_results,
    layer_count=1,
    output_dir=".",
    save=True,
    use_prominent_features=False,
):
    """
    Create MAPE distribution plot showing how experiments are distributed
    across MAPE ranges.

    Args:
        batch_results: Dictionary of batch results from BatchInferencePipeline
        layer_count: Number of layers
        output_dir: Directory to save plot
        save: Whether to save the plot
        use_prominent_features: Whether prominent features filtering was used

    Returns:
        Figure path if saved, None otherwise
    """
    successful = _successful_results(batch_results)

    if not successful:
        logger.info("No successful results available for MAPE distribution plot")
        return None

    # Collect overall MAPE values
    mapes = []
    for result in successful.values():
        if "param_metrics" in result and result["param_metrics"]:
            mape = _get_overall_mape(result["param_metrics"])
            if mape is not None:
                mapes.append(mape)

    if not mapes:
        logger.info("No MAPE data available for plotting")
        return None

    label = _mape_label()

    # Create distribution plot
    fig, ax = plt.subplots()

    # Fixed 5% bins from 0-100%
    bin_edges = list(range(0, 105, 5))
    range_labels = [f"{i}-{i + 5}" for i in range(0, 100, 5)]

    counts = []
    for i in range(len(bin_edges) - 1):
        counts.append(sum(1 for m in mapes if bin_edges[i] <= m < bin_edges[i + 1]))

    bars = ax.bar(range_labels, counts, alpha=0.7)

    # Value labels on bars
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{count}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("MAPE Range (%)")
    ax.set_ylabel("Number of Experiments")
    ax.set_xticks(range(len(range_labels)))
    ax.set_xticklabels(range_labels, rotation=45, ha="right")

    # Statistics text box
    stats_text = f"Mean {label}: {np.mean(mapes):.1f}%\n"
    stats_text += f"Median {label}: {np.median(mapes):.1f}%"
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="none"),
    )

    if save:
        filename = _build_filename(
            "mape_distribution", layer_count, use_prominent_features
        )
        plot_file = Path(output_dir) / filename
        plt.savefig(plot_file)
        plt.close()
        logger.info(f"MAPE distribution plot saved to: {plot_file}")
        return plot_file
    else:
        plt.show()
        return None


def plot_batch_parameter_breakdown(
    batch_results,
    layer_count=1,
    output_dir=".",
    save=True,
    use_prominent_features=False,
):
    """
    Create parameter-specific MAPE breakdown box plot.

    Args:
        batch_results: Dictionary of batch results from BatchInferencePipeline
        layer_count: Number of layers
        output_dir: Directory to save plot
        save: Whether to save the plot
        use_prominent_features: Whether prominent features filtering was used

    Returns:
        Figure path if saved, None otherwise
    """
    successful = _successful_results(batch_results)

    if not successful:
        logger.info("No successful results available for parameter breakdown plot")
        return None

    # Collect parameter-specific MAPE values
    param_mapes = {"thickness": [], "roughness": [], "sld": [], "overall": []}

    for result in successful.values():
        if "param_metrics" not in result or not result["param_metrics"]:
            continue
        pm = result["param_metrics"]

        # Overall MAPE
        overall = _get_overall_mape(pm)
        if overall is not None:
            param_mapes["overall"].append(overall)

        # Per-type MAPEs
        if "by_type" in pm:
            for param_type in ["thickness", "roughness", "sld"]:
                if param_type in pm["by_type"] and isinstance(
                    pm["by_type"][param_type], dict
                ):
                    val = _get_param_mape(pm["by_type"][param_type])
                    if val is not None:
                        param_mapes[param_type].append(val)

    # Filter out empty types
    param_mapes = {k: v for k, v in param_mapes.items() if v}

    if not param_mapes:
        logger.info("No parameter-specific MAPE data available for plotting")
        return None

    label = _mape_label()
    param_names = list(param_mapes.keys())

    # Separate outliers (>100% MAPE) from regular data
    regular_values = []
    outlier_info = []
    for name in param_names:
        vals = param_mapes[name]
        regular = [v for v in vals if v <= 100]
        outliers = [v for v in vals if v > 100]
        regular_values.append(regular)
        outlier_info.append(
            {"count": len(outliers), "max": max(outliers) if outliers else 0}
        )

    fig, ax = plt.subplots()

    ax.boxplot(
        regular_values, tick_labels=param_names, patch_artist=True, showfliers=False
    )

    ax.set_ylim(0, 100)
    ax.set_xlabel("Parameter Type")
    ax.set_ylabel(f"{label} (%)")

    # Outlier indicators
    has_outliers = False
    for i, (name, info) in enumerate(zip(param_names, outlier_info)):
        if info["count"] > 0:
            has_outliers = True
            ax.text(
                i + 1,
                105,
                f"{info['count']} outliers\n(max: {info['max']:.0f}%)",
                ha="center",
                va="bottom",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="none"),
            )

    if has_outliers:
        ax.set_ylim(0, 115)

    # Statistical annotations
    for i, (name, vals, info) in enumerate(
        zip(param_names, regular_values, outlier_info)
    ):
        if vals:
            y_pos = 95 if not has_outliers else 85
            stats = f"Med: {np.median(vals):.1f}%\nMean: {np.mean(vals):.1f}%"
            if info["count"] > 0:
                stats += f"\n{info['count']} outliers"
            ax.text(
                i + 1,
                y_pos,
                stats,
                ha="center",
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="none"),
            )

    if save:
        filename = _build_filename(
            "parameter_breakdown", layer_count, use_prominent_features
        )
        plot_file = Path(output_dir) / filename
        plot_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_file)
        plt.close()
        logger.info(f"Parameter breakdown plot saved to: {plot_file}")
        return plot_file
    else:
        plt.show()
        return None


def create_batch_analysis_plots(
    batch_results,
    layer_count=1,
    output_dir=".",
    save=True,
    use_prominent_features=False,
):
    """
    Create all batch analysis plots (MAPE distribution and parameter breakdown).

    Returns:
        Dictionary with paths to saved plots
    """
    plot_paths = {}

    if save and output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    plot_paths["mape_distribution"] = plot_batch_mape_distribution(
        batch_results,
        layer_count,
        output_dir,
        save,
        use_prominent_features,
    )

    plot_paths["parameter_breakdown"] = plot_batch_parameter_breakdown(
        batch_results,
        layer_count,
        output_dir,
        save,
        use_prominent_features,
    )

    return plot_paths


# ============================================================================
# MODEL COMPARISON PLOTS
# ============================================================================


def plot_model_comparison_histogram(
    baseline_mapes,
    comparison_mapes,
    config,
    output_dir=".",
    save=True,
    baseline_label="Baseline",
    comparison_label="Comparison",
    baseline_meta=None,
    comparison_meta=None,
):
    """
    Create comparison histogram with baseline and comparison model MAPEs.

    Args:
        baseline_mapes: List of baseline model MAPE values
        comparison_mapes: List of comparison model MAPE values
        config: Configuration dict with deviation, sld_fix_mode, prominent keys
        output_dir: Directory to save plot
        save: Whether to save the plot
        baseline_label: Label for baseline model
        comparison_label: Label for comparison model
        baseline_meta: Metadata dict for baseline (priors_type, failed, outliers)
        comparison_meta: Metadata dict for comparison

    Returns:
        Figure path if saved, None otherwise
    """
    baseline_meta = baseline_meta or {}
    comparison_meta = comparison_meta or {}

    mape_label = _mape_label()
    mape_type, title_suffix, deviation_pct = _build_comparison_title(config)

    # Create plot
    fig, ax = plt.subplots()

    # Get MAPE ranges and count values
    mape_ranges, range_labels = _get_mape_ranges()
    baseline_counts = _count_mapes_in_ranges(baseline_mapes, mape_ranges)
    comparison_counts = _count_mapes_in_ranges(comparison_mapes, mape_ranges)

    # Side-by-side bars
    x = np.arange(len(range_labels))
    width = 0.35

    ax.bar(x - width / 2, baseline_counts, width, alpha=0.8, label=baseline_label)
    ax.bar(x + width / 2, comparison_counts, width, alpha=0.8, label=comparison_label)

    ax.set_xlabel("MAPE Range (%)")
    ax.set_ylabel("Number of Experiments")
    ax.set_xticks(x)
    ax.set_xticklabels(range_labels, rotation=45, ha="right")
    ax.legend()

    # Statistics text: mean MAPE per model
    baseline_mean = np.mean(baseline_mapes)
    comparison_mean = np.mean(comparison_mapes)

    stats_text = f"{baseline_label} Mean {mape_label}: {baseline_mean:.1f}%\n"
    stats_text += f"{comparison_label} Mean {mape_label}: {comparison_mean:.1f}%"

    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="none"),
    )

    if save:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_file = output_dir / "model_comparison_mape.pdf"
        plt.savefig(plot_file)
        plt.close()
        logger.info(f"Model comparison plot saved to: {plot_file}")
        return plot_file
    else:
        plt.show()
        return None


def plot_random_guessing_comparison(
    model_mapes,
    random_mapes,
    output_dir=".",
    save=True,
    layer_count=1,
    use_prominent_features=False,
):
    """
    Create comparison histogram with model and random-guessing baseline.

    Args:
        model_mapes: List of model MAPE values
        random_mapes: List of random-guess MAPE values
        output_dir: Directory to save plot
        save: Whether to save the plot
        layer_count: Number of layers
        use_prominent_features: Whether prominent features filtering was used

    Returns:
        Figure path if saved, None otherwise
    """
    mape_label = _mape_label()

    # Create plot
    fig, ax = plt.subplots()

    # Get MAPE ranges and count values
    mape_ranges, range_labels = _get_mape_ranges()
    model_counts = _count_mapes_in_ranges(model_mapes, mape_ranges)
    random_counts = _count_mapes_in_ranges(random_mapes, mape_ranges)

    # Side-by-side bars
    x = np.arange(len(range_labels))
    width = 0.35

    ax.bar(x - width / 2, model_counts, width, alpha=0.8, label="Model")
    ax.bar(x + width / 2, random_counts, width, alpha=0.8, label="Random Guessing")

    ax.set_xlabel("MAPE Range (%)")
    ax.set_ylabel("Number of Experiments")
    ax.set_xticks(x)
    ax.set_xticklabels(range_labels, rotation=45, ha="right")
    ax.legend()

    # Statistics text
    model_mean = np.mean(model_mapes)
    random_mean = np.mean(random_mapes) if random_mapes else 0
    stats_text = f"Model Mean {mape_label}: {model_mean:.1f}%\n"
    stats_text += f"Random Guessing Mean {mape_label}: {random_mean:.1f}%"

    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="none"),
    )

    ax.tick_params(axis="both", which="both", length=0)

    if save:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = _build_filename(
            "random_comparison", layer_count, use_prominent_features
        )
        plot_file = output_dir / filename
        plt.savefig(plot_file)
        plt.close()
        logger.info(f"Random guessing comparison plot saved to: {plot_file}")
        return plot_file
    else:
        plt.show()
        return None


def plot_parameter_comparison_grid(
    baseline_per_param,
    comparison_per_param,
    config,
    output_dir=".",
    save=True,
    baseline_label="Baseline",
    comparison_label="Comparison",
):
    """
    Create per-parameter MAPE comparison plots in a 2x3 grid.

    Args:
        baseline_per_param: Dict of parameter name -> list of MAPEs for baseline
        comparison_per_param: Dict of parameter name -> list of MAPEs for comparison
        config: Configuration dict with deviation, sld_fix_mode, prominent keys
        output_dir: Directory to save plot
        save: Whether to save the plot
        baseline_label: Label for baseline model
        comparison_label: Label for comparison model
    Returns:
        Figure path if saved, None otherwise
    """
    # Get all parameter names
    param_names = sorted(
        set(list(baseline_per_param.keys()) + list(comparison_per_param.keys()))
    )

    if not param_names:
        logger.info("No parameter data available")
        return None

    mape_type, title_suffix, deviation_pct = _build_comparison_title(config)

    # Create subplots
    fig, axes = plt.subplots(2, 3)
    axes = axes.flatten()

    fig.suptitle(
        f"Per-Parameter {mape_type} Distributions{title_suffix}\n"
        f"($\\pm${deviation_pct}% Constraint-Based Priors)",
    )

    # Get MAPE ranges
    mape_ranges, range_labels = _get_mape_ranges()

    for idx, param_name in enumerate(param_names):
        if idx >= len(axes):
            break

        ax = axes[idx]

        baseline_vals = baseline_per_param.get(param_name, [])
        comparison_vals = comparison_per_param.get(param_name, [])

        # Count values in each range
        baseline_counts = _count_mapes_in_ranges(baseline_vals, mape_ranges)
        comparison_counts = _count_mapes_in_ranges(comparison_vals, mape_ranges)

        # Plot baseline as background
        ax.bar(range(len(baseline_counts)), baseline_counts, alpha=0.3, linewidth=1.0)

        # Overlay comparison as foreground
        ax.bar(
            range(len(comparison_counts)), comparison_counts, alpha=0.8, linewidth=0.5
        )

        ax.set_title(param_name)
        ax.set_xlabel("MAPE Range (%)")
        ax.set_ylabel("Count")
        ax.set_xticks(range(0, len(range_labels), 4))
        ax.set_xticklabels(
            [range_labels[i] for i in range(0, len(range_labels), 4)],
            rotation=45,
            ha="right",
        )

        # Add statistics
        if baseline_vals and comparison_vals:
            baseline_mean = np.mean(baseline_vals)
            comparison_mean = np.mean(comparison_vals)
            improvement = ((baseline_mean - comparison_mean) / baseline_mean) * 100

            stats = f"{comparison_label[:4]}: {comparison_mean:.1f}%\n"
            stats += f"{baseline_label[:4]}: {baseline_mean:.1f}%\n"
            stats += f"$\\Delta$: {improvement:+.1f}%"
            ax.text(
                0.98,
                0.98,
                stats,
                transform=ax.transAxes,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="none"),
                family="monospace",
            )

    # Hide unused subplots
    for idx in range(len(param_names), len(axes)):
        axes[idx].axis("off")

    if save:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_file = output_dir / "param_comparison.pdf"
        plt.savefig(plot_file)
        plt.close()
        logger.info(f"Parameter comparison plot saved to: {plot_file}")
        return plot_file
    else:
        plt.show()
        return None
