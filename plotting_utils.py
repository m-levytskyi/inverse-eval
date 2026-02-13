#!/usr/bin/env python3
"""
Plotting utilities for reflectometry analysis.

This module contains all plotting functions used in the reflectometry pipeline,
keeping plotting logic separate from the main inference pipeline.

Uses SciencePlots for publication-quality styling via paper.mplstyle.
"""

import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401 - registers 'science' style
from pathlib import Path

# Apply publication-quality style globally
paper_mplstyle = Path(__file__).parent / "paper.mplstyle"
plt.style.use(["science", str(paper_mplstyle)])


# ============================================================================
# SHARED HELPERS
# ============================================================================


def _extract_batch_config(batch_results):
    """Extract priors_type and fix_sld_mode from the first successful result.

    Returns:
        (successful_results, priors_type, fix_sld_mode)
    """
    successful = {k: v for k, v in batch_results.items() if v.get("success", False)}
    priors_type = None
    fix_sld_mode = "none"

    for result in successful.values():
        if "priors_config" in result:
            priors_type = result["priors_config"].get("priors_type")
            fix_sld_mode = result["priors_config"].get("fix_sld_mode", "none")
            break

    if priors_type is None:
        raise ValueError("Could not determine priors_type from any successful result")

    return successful, priors_type, fix_sld_mode


def _get_overall_mape(param_metrics, priors_type):
    """Extract the appropriate overall MAPE value based on priors type."""
    if priors_type == "constraint_based" and "overall" in param_metrics:
        if "constraint_mape" in param_metrics["overall"]:
            return param_metrics["overall"]["constraint_mape"]

    if "overall_mape" in param_metrics:
        return param_metrics["overall_mape"]

    if "overall" in param_metrics and isinstance(param_metrics["overall"], dict):
        if "mape" in param_metrics["overall"]:
            return param_metrics["overall"]["mape"]

    return None


def _get_param_mape(param_data, priors_type):
    """Extract MAPE value from a by_type parameter entry."""
    if priors_type == "constraint_based" and "constraint_mape" in param_data:
        return param_data["constraint_mape"]
    if "mape" in param_data:
        return param_data["mape"]
    return None


def _mape_label(priors_type):
    """Return the appropriate MAPE label string."""
    return "Constraint MAPE" if priors_type == "constraint_based" else "MAPE"


def _build_filename(prefix, layer_count, use_prominent_features=False):
    """Build a standard plot filename."""
    parts = [f"{prefix}_{layer_count}layer"]
    if use_prominent_features:
        parts.append("prominent")
    return "_".join(parts) + ".pdf"


def _get_mape_ranges():
    """Return standard MAPE ranges and labels."""
    mape_ranges = list(range(0, 105, 5))
    range_labels = [f"{i}-{i + 5}%" for i in range(0, 100, 5)]
    return mape_ranges, range_labels


def _count_mapes_in_ranges(mapes, mape_ranges):
    """Count MAPE values in each range bin."""
    return [
        sum(1 for m in mapes if mape_ranges[i] <= m < mape_ranges[i + 1])
        for i in range(len(mape_ranges) - 1)
    ]


def _build_comparison_title(config, priors_type="constraint_based"):
    """Build title for comparison plots with config-based suffix."""
    mape_type = "Constraint-Based MAPE" if priors_type == "constraint_based" else "MAPE"
    
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


def _format_model_stats(mapes, label, mape_label, meta=None):
    """Format statistics text for a model."""
    meta = meta or {}
    stats = f"{label}:\n"
    stats += f"  Total: {len(mapes)} experiments\n"
    stats += f"  Mean {mape_label}: {np.mean(mapes):.1f}%\n"
    stats += f"  Median {mape_label}: {np.median(mapes):.1f}%\n"
    stats += f"  Std Dev: {np.std(mapes):.1f}%\n"
    if "failed" in meta:
        stats += f"  Failed: {meta['failed']}\n"
    if "outliers" in meta:
        stats += f"  Outliers: {meta['outliers']}\n"
    return stats


# ============================================================================
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

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Plot reflectivity curves
    ax.set_yscale("log")
    ax.set_xlabel("Q [$\\AA^{-1}$]", fontsize=14)
    ax.set_ylabel("R(Q)", fontsize=14)
    ax.tick_params(axis="both", which="both", labelsize=12, length=0)

    # Experimental data with error bars
    ax.errorbar(
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

    # Predicted curves
    ax.plot(q_model, predicted_curve, lw=2, label="Predicted")
    ax.plot(q_model, polished_curve, ls="--", lw=2, label="Polished")

    ax.legend(loc="upper right", fontsize=12)
    if not hide_title:
        ax.set_title(f"Reflectivity - {display_name}", fontsize=14)

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
    **kwargs,
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
    successful, priors_type, _ = _extract_batch_config(batch_results)

    if not successful:
        print("No successful results available for MAPE distribution plot")
        return None

    # Collect overall MAPE values
    mapes = []
    for result in successful.values():
        if "param_metrics" in result and result["param_metrics"]:
            mape = _get_overall_mape(result["param_metrics"], priors_type)
            if mape is not None:
                mapes.append(mape)

    if not mapes:
        print("No MAPE data available for plotting")
        return None

    label = _mape_label(priors_type)

    # Create distribution plot
    fig, ax = plt.subplots()

    # Fixed 5% bins from 0-100%
    bin_edges = list(range(0, 105, 5))
    range_labels = [f"{i}-{i + 5}\\%" for i in range(0, 100, 5)]

    counts = []
    for i in range(len(bin_edges) - 1):
        counts.append(sum(1 for m in mapes if bin_edges[i] <= m < bin_edges[i + 1]))

    bars = ax.bar(range_labels, counts, alpha=0.7)

    # Value labels on bars
    for bar, count in zip(bars, counts):
        if count > 0:
            pct = (count / len(mapes)) * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{count}\n({pct:.1f}\\%)",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    ax.set_xlabel("MAPE Range")
    ax.set_ylabel("Number of Experiments")
    ax.set_xticks(range(len(range_labels)))
    ax.set_xticklabels(range_labels, rotation=45, ha="right")

    # Statistics text box
    stats_text = f"Mean {label}: {np.mean(mapes):.1f}\\%\n"
    stats_text += f"Median {label}: {np.median(mapes):.1f}\\%"
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
        print(f"MAPE distribution plot saved to: {plot_file}")
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
    **kwargs,
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
    successful, priors_type, _ = _extract_batch_config(batch_results)

    if not successful:
        print("No successful results available for parameter breakdown plot")
        return None

    # Collect parameter-specific MAPE values
    param_mapes = {"thickness": [], "roughness": [], "sld": [], "overall": []}

    for result in successful.values():
        if "param_metrics" not in result or not result["param_metrics"]:
            continue
        pm = result["param_metrics"]

        # Overall MAPE
        overall = _get_overall_mape(pm, priors_type)
        if overall is not None:
            param_mapes["overall"].append(overall)

        # Per-type MAPEs
        if "by_type" in pm:
            for param_type in ["thickness", "roughness", "sld"]:
                if param_type in pm["by_type"] and isinstance(
                    pm["by_type"][param_type], dict
                ):
                    val = _get_param_mape(pm["by_type"][param_type], priors_type)
                    if val is not None:
                        param_mapes[param_type].append(val)

    # Filter out empty types
    param_mapes = {k: v for k, v in param_mapes.items() if v}

    if not param_mapes:
        print("No parameter-specific MAPE data available for plotting")
        return None

    label = _mape_label(priors_type)
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
    ax.set_ylabel(f"{label} (\\%)")

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
        plt.savefig(plot_file)
        plt.close()
        print(f"Parameter breakdown plot saved to: {plot_file}")
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
    **kwargs,
):
    """
    Create all batch analysis plots (MAPE distribution and parameter breakdown).

    Extra kwargs are accepted for backward compatibility but ignored.

    Returns:
        Dictionary with paths to saved plots
    """
    plot_paths = {}

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
    
    priors_type = baseline_meta.get("priors_type", "constraint_based")
    mape_label = _mape_label(priors_type)
    mape_type, title_suffix, deviation_pct = _build_comparison_title(config, priors_type)

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle(
        f"{mape_type} Distribution - Model Comparison{title_suffix}\n"
        f"(±{deviation_pct}% Constraint-Based Priors)",
        fontsize=16,
        fontweight="bold",
    )

    # Get MAPE ranges and count values
    mape_ranges, range_labels = _get_mape_ranges()
    baseline_counts = _count_mapes_in_ranges(baseline_mapes, mape_ranges)
    comparison_counts = _count_mapes_in_ranges(comparison_mapes, mape_ranges)

    # Plot baseline as background
    ax.bar(
        range(len(baseline_counts)),
        baseline_counts,
        alpha=0.3,
        linewidth=1.5,
        label=baseline_label,
    )

    # Overlay comparison as foreground
    bars = ax.bar(
        range(len(comparison_counts)),
        comparison_counts,
        alpha=0.8,
        label=comparison_label,
    )

    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, comparison_counts)):
        if count > 0:
            percentage = (count / len(comparison_mapes)) * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{count}\n({percentage:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("MAPE Range", fontsize=14)
    ax.set_ylabel("Number of Experiments", fontsize=14)
    ax.set_xticks(range(len(range_labels)))
    ax.set_xticklabels(range_labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9)

    # Statistics text
    stats_text = _format_model_stats(comparison_mapes, comparison_label, mape_label, comparison_meta)
    stats_text += "\n" + _format_model_stats(baseline_mapes, baseline_label, mape_label, baseline_meta)
    
    # Calculate improvement
    improvement = ((np.mean(baseline_mapes) - np.mean(comparison_mapes)) / 
                   np.mean(baseline_mapes)) * 100
    stats_text += f"\nImprovement: {improvement:+.1f}%"

    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", alpha=0.9),
        family="monospace",
    )

    plt.tight_layout()

    if save:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_file = output_dir / "model_comparison_mape.pdf"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Model comparison plot saved to: {plot_file}")
        return plot_file
    else:
        plt.show()
        return None


def plot_random_guessing_comparison(
    model_mapes,
    random_mapes,
    output_dir=".",
    save=True,
    priors_type="constraint_based",
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
        priors_type: Type of priors used
        layer_count: Number of layers
        use_prominent_features: Whether prominent features filtering was used

    Returns:
        Figure path if saved, None otherwise
    """
    mape_label = _mape_label(priors_type)

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Get MAPE ranges and count values
    mape_ranges, range_labels = _get_mape_ranges()
    model_counts = _count_mapes_in_ranges(model_mapes, mape_ranges)
    random_counts = _count_mapes_in_ranges(random_mapes, mape_ranges)

    # Side-by-side bars
    x = np.arange(len(range_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, model_counts, width, alpha=0.8, label='Model')
    bars2 = ax.bar(x + width/2, random_counts, width, alpha=0.8, 
                   label='Random Guessing')

    # Add value labels on model bars
    for i, (bar, count) in enumerate(zip(bars1, model_counts)):
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
    ax.set_xticks(x)
    ax.set_xticklabels(range_labels, rotation=45, ha="right")
    ax.legend()

    # Statistics text
    random_mean = np.mean(random_mapes) if random_mapes else 0
    stats_text = f"Total: {len(model_mapes)} experiments\n"
    stats_text += f"Mean {mape_label}: {np.mean(model_mapes):.1f}%\n"
    stats_text += f"Median {mape_label}: {np.median(model_mapes):.1f}%\n"
    stats_text += f"\nRandom Guessing Mean: {random_mean:.1f}%"

    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor='none'),
    )

    ax.tick_params(axis="both", which="both", length=0)
    plt.tight_layout()

    if save:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = _build_filename("random_comparison", layer_count, 
                                   use_prominent_features)
        plot_file = output_dir / filename
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Random guessing comparison plot saved to: {plot_file}")
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
    priors_type="constraint_based",
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
        priors_type: Type of priors used

    Returns:
        Figure path if saved, None otherwise
    """
    # Get all parameter names
    param_names = sorted(
        set(list(baseline_per_param.keys()) + list(comparison_per_param.keys()))
    )

    if not param_names:
        print(f"No parameter data available")
        return None

    mape_label = _mape_label(priors_type)
    mape_type, title_suffix, deviation_pct = _build_comparison_title(config, priors_type)

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    fig.suptitle(
        f"Per-Parameter {mape_type} Distributions - Model Comparison{title_suffix}\n"
        f"(±{deviation_pct}% Constraint-Based Priors)",
        fontsize=16,
        fontweight="bold",
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
        ax.bar(range(len(comparison_counts)), comparison_counts, alpha=0.8, linewidth=0.5)

        ax.set_title(param_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("MAPE Range", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_xticks(range(0, len(range_labels), 4))
        ax.set_xticklabels(
            [range_labels[i] for i in range(0, len(range_labels), 4)],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        ax.grid(True, alpha=0.3, axis="y")

        # Add statistics
        if baseline_vals and comparison_vals:
            baseline_mean = np.mean(baseline_vals)
            comparison_mean = np.mean(comparison_vals)
            improvement = ((baseline_mean - comparison_mean) / baseline_mean) * 100

            stats = f"{comparison_label[:4]}: {comparison_mean:.1f}%\n"
            stats += f"{baseline_label[:4]}: {baseline_mean:.1f}%\n"
            stats += f"Δ: {improvement:+.1f}%"
            ax.text(
                0.98,
                0.98,
                stats,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.8),
                family="monospace",
            )

    # Hide unused subplots
    for idx in range(len(param_names), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if save:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_file = output_dir / "param_comparison.pdf"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Parameter comparison plot saved to: {plot_file}")
        return plot_file
    else:
        plt.show()
        return None
