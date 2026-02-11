#!/usr/bin/env python3
"""
Plotting utilities for reflectometry analysis.

This module contains all plotting functions used in the reflectometry pipeline,
keeping plotting logic separate from the main inference pipeline.

Uses SciencePlots for publication-quality styling with configurable options.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots  # Required for consistent styling
from pathlib import Path

# ============================================================================
# CONFIGURABLE PLOT SETTINGS
# ============================================================================
USE_LATEX = True  # Enable LaTeX text rendering for publication-quality plots
SAVE_AS_PDF = True  # Save plots as PDF instead of PNG

# ============================================================================
# APPLY PUBLICATION-QUALITY STYLE GLOBALLY
# ============================================================================
# Load paper.mplstyle with SciencePlots baseline
paper_mplstyle = Path(__file__).parent / "paper.mplstyle"
plt.style.use(["science", str(paper_mplstyle)])


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
    print(f"\nDEBUG [plot_simple_comparison]: Starting plot for {experiment_name}")
    print(f"  q_exp shape: {np.array(q_exp).shape if hasattr(q_exp, '__len__') else 'scalar'}")
    print(f"  curve_exp shape: {np.array(curve_exp).shape if hasattr(curve_exp, '__len__') else 'scalar'}")
    print(f"  predicted_sld_x shape: {np.array(predicted_sld_x).shape if hasattr(predicted_sld_x, '__len__') else 'scalar'}")
    print(f"  hide_title: {hide_title}")
    
    # Add SLD fixing mode to the experiment name if available
    display_name = experiment_name
    if priors_config and "fix_sld_mode" in priors_config:
        fix_sld_mode = priors_config["fix_sld_mode"]
        if fix_sld_mode != "none":
            display_name = f"{experiment_name} (SLD fixed: {fix_sld_mode})"

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Plot reflectivity curves
    ax.set_yscale("log")
    ax.set_xlabel("Q [Å$^{-1}$]", fontsize=14)
    ax.set_ylabel("R(Q)", fontsize=14)
    ax.tick_params(axis="both", which="both", labelsize=12, length=0)

    # Experimental data with error bars
    el = ax.errorbar(
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


def plot_batch_edge_case_detection(
    batch_results,
    layer_count=1,
    output_dir=".",
    save=True,
    use_prominent_features=False,
    failed_count=0,
    outlier_count=0,
    hide_title=False,
):
    """
    Create edge case detection plot showing experiments with high MAPE values.

    Args:
        batch_results: Dictionary of batch results from BatchInferencePipeline
        layer_count: Number of layers for plot title
        output_dir: Directory to save plot
        save: Whether to save the plot
        use_prominent_features: Whether prominent features filtering was used
        failed_count: Number of failed experiments (excluding outliers)
        outlier_count: Number of outlier experiments (excluded)
        hide_title: Hide plot titles (for publication)

    Returns:
        Figure path if saved, None otherwise
    """

    # Collect experiment data and SLD fixing mode
    exp_data = {}
    fix_sld_mode = "none"
    priors_type_used = None

    for exp_id, exp_result in batch_results.items():
        if exp_result.get("success", False) and "param_metrics" in exp_result:
            param_metrics = exp_result["param_metrics"]

            # Extract SLD fixing mode from first successful result
            if priors_type_used is None and "priors_config" in exp_result:
                fix_sld_mode = exp_result["priors_config"].get("fix_sld_mode", "none")
                priors_type_used = exp_result["priors_config"].get("priors_type")
                if priors_type_used is None:
                    raise ValueError(
                        f"Missing priors_type in priors_config for experiment {exp_id}"
                    )

            if priors_type_used is None:
                raise ValueError(
                    "Could not determine priors_type from any successful result"
                )  # Get the appropriate MAPE value based on priors type
            overall_mape = None

            # For constraint-based priors, prefer constraint_mape if available
            if priors_type_used == "constraint_based" and "overall" in param_metrics:
                if "constraint_mape" in param_metrics["overall"]:
                    overall_mape = param_metrics["overall"]["constraint_mape"]

            # Fallback to regular MAPE
            if overall_mape is None and "overall_mape" in param_metrics:
                overall_mape = param_metrics["overall_mape"]

            if overall_mape is not None:
                exp_data[exp_id] = overall_mape

    if not exp_data:
        print("No data available for edge case detection")
        return None

    # Create edge case detection plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))

    # Create title with all configuration information
    title_parts = [f"{len(batch_results)} {layer_count}-Layer Experiments"]

    if fix_sld_mode != "none":
        title_parts.append(f"SLD fix: {fix_sld_mode}")

    if use_prominent_features:
        title_parts.append("Prominent Features")

    # Add exclusion statistics to title if present
    if outlier_count > 0 or failed_count > 0:
        exclusion_info = []
        if outlier_count > 0:
            exclusion_info.append(f"{outlier_count} outliers")
        if failed_count > 0:
            exclusion_info.append(f"{failed_count} failed")
        title_parts.append(f"Excluded: {', '.join(exclusion_info)}")

    title_suffix = f" ({', '.join(title_parts[1:])})" if len(title_parts) > 1 else ""

    if not hide_title:
        fig.suptitle(
            f"Edge Case Detection - {title_parts[0]}{title_suffix}",
            fontsize=16,
            fontweight="bold",
        )

    exp_ids = list(exp_data.keys())
    exp_vals = list(exp_data.values())
    exp_indices = range(len(exp_ids))

    # Plot all experiments
    ax.plot(
        exp_indices,
        exp_vals,
        "o-",
        alpha=0.7,
        linewidth=1,
        markersize=4,
        label="Experiments",
    )

    # Calculate threshold for edge cases (mean + 2*std)
    mean_mape = np.mean(exp_vals)
    std_mape = np.std(exp_vals)
    threshold = mean_mape + 2 * std_mape

    # Highlight edge cases
    edge_cases = [
        (i, exp_id, mape)
        for i, (exp_id, mape) in enumerate(zip(exp_ids, exp_vals))
        if mape > threshold
    ]

    if edge_cases:
        edge_indices = [i for i, _, _ in edge_cases]
        edge_mapes = [mape for _, _, mape in edge_cases]
        ax.scatter(
            edge_indices,
            edge_mapes,
            s=80,
            alpha=0.8,
            label=f"Edge Cases (>{threshold:.1f}\\%)",
            zorder=5,
        )

        # Annotate worst edge cases (top 3)
        worst_cases = sorted(edge_cases, key=lambda x: x[2], reverse=True)[:3]
        for i, exp_id, mape in worst_cases:
            ax.annotate(
                f"{exp_id}\n{mape:.1f}%",
                xy=(i, mape),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='none', edgecolor='none'),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
            )

    # Add threshold line
    ax.axhline(
        y=threshold,
        linestyle="--",
        alpha=0.7,
        label=f"Threshold ($\\mu+2\\sigma$ = {threshold:.1f}\\%)",
    )

    # Add mean line
    ax.axhline(
        y=mean_mape, linestyle="--", alpha=0.7, label=f"Mean MAPE ({mean_mape:.1f}\\%)"
    )

    ax.set_xlabel("Experiment Index")
    ax.set_ylabel("MAPE (\\%)")
    ax.tick_params(axis="both", which="both", length=0)
    if not hide_title:
        ax.set_title("Edge Case Detection")
    ax.legend()

    # Add statistics text
    stats_text = f"Total: {len(exp_vals)} experiments\n"
    stats_text += f"Mean MAPE: {mean_mape:.1f}% ± {std_mape:.1f}%\n"
    stats_text += f"Threshold: {threshold:.1f}%\n"
    stats_text += (
        f"Edge Cases: {len(edge_cases)} ({100 * len(edge_cases) / len(exp_vals):.1f}%)"
    )

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", alpha=0.8),
    )

    plt.tight_layout()

    if save:
        filename_parts = [f"edge_case_detection_{layer_count}layer"]

        if use_prominent_features:
            filename_parts.append("prominent")

        filename = "_".join(filename_parts) + ".png"
        plot_path = Path(output_dir) / filename
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Edge case detection plot saved to: {plot_path}")

        # Print edge cases summary
        if edge_cases:
            print(f"\nEdge Cases (MAPE > {threshold:.1f}%):")
            for i, exp_id, mape in sorted(edge_cases, key=lambda x: x[2], reverse=True):
                print(f"  {exp_id}: {mape:.1f}% MAPE")

        return plot_path
    else:
        plt.show()
        return None


def plot_batch_mape_distribution(
    batch_results,
    layer_count=1,
    output_dir=".",
    save=True,
    narrow_priors_deviation=0.99,
    use_prominent_features=False,
    failed_count=0,
    outlier_count=0,
    hide_title=False,
    minimal_stats=False,
):
    """
    Create MAPE distribution plot showing how experiments are distributed across MAPE ranges.

    Args:
        batch_results: Dictionary of batch results from BatchInferencePipeline
        layer_count: Number of layers for plot title
        output_dir: Directory to save plot
        save: Whether to save the plot
        narrow_priors_deviation: Deviation for narrow priors display in title
        use_prominent_features: Whether prominent features filtering was used
        failed_count: Number of failed experiments (excluding outliers)
        outlier_count: Number of outlier experiments (excluded)
        hide_title: Hide plot titles (for publication)
        minimal_stats: Show minimal statistics (only mean and median)

    Returns:
        Figure path if saved, None otherwise
    """
    # Filter successful results
    successful_results = {
        k: v for k, v in batch_results.items() if v.get("success", False)
    }

    if not successful_results:
        print("No successful results available for MAPE distribution plot")
        return None

    # Collect real overall MAPE values with debugging
    mape_data = {"narrow": []}
    fix_sld_mode = "none"
    priors_type_used = None

    # Extract priors type from first successful result (all should have same config)
    for result in successful_results.values():
        if "priors_config" in result:
            priors_type_used = result["priors_config"].get("priors_type")
            if priors_type_used is None:
                raise ValueError(
                    f"Missing priors_type in priors_config for experiment {list(successful_results.keys())[0]}"
                )
            fix_sld_mode = result["priors_config"].get("fix_sld_mode", "none")
            break

    if priors_type_used is None:
        raise ValueError("Could not determine priors_type from any successful result")

    print(f"\nDEBUG - MAPE distribution collection:")
    print(f"Detected priors_type: {priors_type_used}")
    print(
        f"Using MAPE field: {'constraint_mape' if priors_type_used == 'constraint_based' else 'mape'}"
    )

    for exp_id, result in successful_results.items():
        if "param_metrics" in result and result["param_metrics"]:
            param_metrics = result["param_metrics"]

            # Get the appropriate MAPE value based on priors type
            overall_mape = None

            # For constraint-based priors, use constraint_mape field
            if priors_type_used == "constraint_based":
                if (
                    "overall" in param_metrics
                    and "constraint_mape" in param_metrics["overall"]
                ):
                    overall_mape = param_metrics["overall"]["constraint_mape"]
                    print(f"  {exp_id}: constraint_mape = {overall_mape:.2f}%")
            else:
                # For non-constraint priors, use standard MAPE
                if "overall" in param_metrics and isinstance(
                    param_metrics["overall"], dict
                ):
                    if "mape" in param_metrics["overall"]:
                        overall_mape = param_metrics["overall"]["mape"]
                        print(f"  {exp_id}: overall.mape = {overall_mape:.2f}%")

            if overall_mape is not None:
                mape_data["narrow"].append(overall_mape)

    if not mape_data["narrow"]:
        print("No MAPE data available for plotting")
        return None

    mapes = mape_data["narrow"]
    print(f"\nCollected {len(mapes)} real MAPE values")
    print(f"MAPE range: {np.min(mapes):.1f}% - {np.max(mapes):.1f}%")
    print(f"Mean MAPE: {np.mean(mapes):.1f}% ± {np.std(mapes):.1f}%")
    print(f"Median MAPE: {np.median(mapes):.1f}%")

    # Create title with all configuration information
    title_parts = []
    if fix_sld_mode != "none":
        title_parts.append(f"SLD fix: {fix_sld_mode}")
    if use_prominent_features:
        title_parts.append("Prominent Features")

    # Add MAPE type indicator for constraint-based priors
    mape_type_label = (
        "Constraint-Based MAPE" if priors_type_used == "constraint_based" else "MAPE"
    )

    title_suffix = f" ({', '.join(title_parts)})" if title_parts else ""

    # Create distribution plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    if not hide_title:
        fig.suptitle(
            f"{mape_type_label} Distribution - {len(successful_results)} {layer_count}-Layer Experiments{title_suffix}\n"
            f"(Narrow Priors ±{int(narrow_priors_deviation * 100)}%)",
            fontsize=16,
            fontweight="bold",
        )

    # Define MAPE ranges - fixed 5% bins from 0-100%
    mape_ranges = list(range(0, 105, 5))  # [0, 5, 10, 15, ..., 95, 100]
    range_labels = [f"{i}-{i + 5}\\%" for i in range(0, 100, 5)]

    # Count experiments in each MAPE range
    counts = []
    for i in range(len(mape_ranges) - 1):
        count = sum(1 for mape in mapes if mape_ranges[i] <= mape < mape_ranges[i + 1])
        counts.append(count)

    # Create bar chart
    bars = ax.bar(range_labels, counts, alpha=0.7)

    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        if count > 0:
            percentage = (count / len(mapes)) * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{count}\n({percentage:.1f}\\%)",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    ax.set_xlabel("MAPE Range")
    ax.set_ylabel("Number of Experiments")
    ax.tick_params(axis="both", which="major", length=0)
    ax.set_xticks(range(len(range_labels)))
    ax.set_xticklabels(range_labels, rotation=45, ha="right")


    # Determine MAPE label for statistics
    mape_label = "Constraint MAPE" if priors_type_used == "constraint_based" else "MAPE"

    # Add statistics text with outlier and failure counts
    if mapes:
        if minimal_stats:
            # Minimal stats for paper mode
            stats_text = f"Total: {len(mapes)} experiments\n"
            stats_text += f"Mean {mape_label}: {np.mean(mapes):.1f}%\n"
            stats_text += f"Median {mape_label}: {np.median(mapes):.1f}%"
        else:
            # Full stats for regular mode
            stats_text = f"Total: {len(mapes)} experiments\n"
            stats_text += f"Mean {mape_label}: {np.mean(mapes):.1f}%\n"
            stats_text += f"Median {mape_label}: {np.median(mapes):.1f}%\n"
            stats_text += f"Std Dev: {np.std(mapes):.1f}%\n"
            stats_text += f"Min {mape_label}: {np.min(mapes):.1f}%\n"
            stats_text += f"Max {mape_label}: {np.max(mapes):.1f}%"

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
            bbox=dict(boxstyle="round,pad=0.5", facecolor='none'),
        )

    plt.tight_layout()

    if save:
        # Save plot
        filename_parts = [f"mape_distribution_{layer_count}layer"]

        if use_prominent_features:
            filename_parts.append("prominent")

        filename = "_".join(filename_parts) + ".png"
        plot_file = Path(output_dir) / filename
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
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
    narrow_priors_deviation=0.99,
    use_prominent_features=False,
    failed_count=0,
    outlier_count=0,
    hide_title=False,
):
    """
    Create parameter-specific MAPE breakdown plot with detailed debugging.

    Args:
        batch_results: Dictionary of batch results from BatchInferencePipeline
        layer_count: Number of layers for plot title
        output_dir: Directory to save plot
        save: Whether to save the plot
        narrow_priors_deviation: Deviation for narrow priors display in title
        use_prominent_features: Whether prominent features filtering was used
        failed_count: Number of failed experiments (excluding outliers)
        outlier_count: Number of outlier experiments (excluded)
        hide_title: Hide plot titles (for publication)

    Returns:
        Figure path if saved, None otherwise
    """
    # Filter successful results
    successful_results = {
        k: v for k, v in batch_results.items() if v.get("success", False)
    }

    if not successful_results:
        print("No successful results available for parameter breakdown plot")
        return None

    # Collect parameter-specific MAPE values from by_type structure
    param_mapes = {"thickness": [], "roughness": [], "sld": [], "overall": []}

    # Extract SLD fixing mode and priors type from results
    fix_sld_mode = "none"
    priors_type_used = None
    for result in successful_results.values():
        if "priors_config" in result:
            fix_sld_mode = result["priors_config"].get("fix_sld_mode", "none")
            priors_type_used = result["priors_config"].get("priors_type")
            if priors_type_used is None:
                raise ValueError(f"Missing priors_type in priors_config")
            break

    if priors_type_used is None:
        raise ValueError("Could not determine priors_type from any successful result")

    print("\nDEBUG - Parameter breakdown collection:")

    for exp_id, result in successful_results.items():
        if "param_metrics" in result and result["param_metrics"]:
            param_metrics = result["param_metrics"]

            print(f"\nExperiment {exp_id}:")

            # Overall MAPE - prefer constraint_mape for constraint-based priors
            overall_mape = None

            # For constraint-based priors, prefer constraint_mape if available
            if priors_type_used == "constraint_based" and "overall" in param_metrics:
                if "constraint_mape" in param_metrics["overall"]:
                    overall_mape = param_metrics["overall"]["constraint_mape"]
                    param_mapes["overall"].append(overall_mape)
                    print(f"  Overall Constraint MAPE: {overall_mape:.2f}%")

            # Fallback to regular MAPE
            if overall_mape is None:
                if "overall_mape" in param_metrics:
                    overall_mape = param_metrics["overall_mape"]
                    param_mapes["overall"].append(overall_mape)
                    print(f"  Overall MAPE: {overall_mape:.2f}%")
                elif "overall" in param_metrics and isinstance(
                    param_metrics["overall"], dict
                ):
                    if "mape" in param_metrics["overall"]:
                        overall_mape = param_metrics["overall"]["mape"]
                        param_mapes["overall"].append(overall_mape)
                        print(f"  Overall MAPE: {overall_mape:.2f}%")

            # Individual parameter MAPEs from by_type structure
            if "by_type" in param_metrics:
                by_type = param_metrics["by_type"]
                print(f"  by_type data:")
                for param_type in ["thickness", "roughness", "sld"]:
                    if param_type in by_type and isinstance(by_type[param_type], dict):
                        # For constraint-based priors, prefer constraint_mape if available
                        mape_val = None
                        if (
                            priors_type_used == "constraint_based"
                            and "constraint_mape" in by_type[param_type]
                        ):
                            mape_val = by_type[param_type]["constraint_mape"]
                            print(f"    {param_type}: {mape_val:.2f}% (constraint)")
                        elif "mape" in by_type[param_type]:
                            mape_val = by_type[param_type]["mape"]
                            print(f"    {param_type}: {mape_val:.2f}%")

                        if mape_val is not None:
                            param_mapes[param_type].append(mape_val)
                        else:
                            print(f"    {param_type}: no MAPE data")
                    else:
                        print(f"    {param_type}: not found in by_type")

    # Filter out empty parameter types
    param_mapes = {k: v for k, v in param_mapes.items() if v}

    print(f"\nFinal parameter counts:")
    for param_type, values in param_mapes.items():
        print(f"  {param_type}: {len(values)} values")

    if not param_mapes:
        print("No parameter-specific MAPE data available for plotting")
        return None

    # Create box plot
    fig, ax = plt.subplots(figsize=(12, 8))

    param_names = list(param_mapes.keys())

    # Separate outliers (>100% MAPE) from regular data for each parameter type
    param_values_regular = []
    outlier_info = []

    for name in param_names:
        values = param_mapes[name]
        regular_values = [v for v in values if v <= 100]
        outliers = [v for v in values if v > 100]

        param_values_regular.append(regular_values)
        outlier_info.append(
            {"count": len(outliers), "max_value": max(outliers) if outliers else 0}
        )

    # Create box plot with fixed scale 0-100%
    box_plot = ax.boxplot(
        param_values_regular,
        tick_labels=param_names,
        patch_artist=True,
        showfliers=False,
    )  # Don't show regular fliers, we'll handle outliers separately

    # Set fixed scale from 0-100%
    ax.set_ylim(0, 100)
    ax.set_xlabel("Parameter Type", fontsize=12)

    # Determine MAPE label based on priors type
    mape_label = (
        "Constraint-Based MAPE" if priors_type_used == "constraint_based" else "MAPE"
    )
    ax.set_ylabel(f"{mape_label} (\\%)", fontsize=12)
    ax.tick_params(axis="both", which="both", length=0)

    # Create title with all configuration information
    title_parts = []
    if fix_sld_mode != "none":
        title_parts.append(f"SLD fix: {fix_sld_mode}")
    if use_prominent_features:
        title_parts.append("Prominent Features")

    # Add exclusion statistics if present
    if outlier_count > 0 or failed_count > 0:
        exclusion_info = []
        if outlier_count > 0:
            exclusion_info.append(f"{outlier_count} outliers")
        if failed_count > 0:
            exclusion_info.append(f"{failed_count} failed")
        title_parts.append(f"Excluded: {', '.join(exclusion_info)}")

    title_suffix = f" ({', '.join(title_parts)})" if title_parts else ""

    if not hide_title:
        ax.set_title(
            f"Parameter-Specific {mape_label} Distribution - {len(successful_results)} {layer_count}-Layer Experiments{title_suffix}\n"
            f"(Narrow Priors ±{int(narrow_priors_deviation * 100)}%)",
            fontsize=14,
            fontweight="bold",
        )
    ax.grid(True, alpha=0.3, axis="y")

    # Add outlier indicators above the plot
    outlier_y_pos = 105  # Just above the 100% line
    has_outliers = False

    for i, (name, info) in enumerate(zip(param_names, outlier_info)):
        if info["count"] > 0:
            has_outliers = True
            outlier_text = f"{info['count']} outliers\n(max: {info['max_value']:.0f}%)"
            ax.text(
                i + 1,
                outlier_y_pos,
                outlier_text,
                ha="center",
                va="bottom",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='none'),
                fontweight="bold",
            )

    # Extend y-axis slightly to accommodate outlier indicators
    if has_outliers:
        ax.set_ylim(0, 115)

    # Add statistical annotations (updated for regular values only)
    for i, (name, regular_values, info) in enumerate(
        zip(param_names, param_values_regular, outlier_info)
    ):
        if regular_values:  # Only add annotation if there are regular values
            median_val = np.median(regular_values)
            mean_val = np.mean(regular_values)

            # Position annotations lower if there are outliers
            y_pos = 95 if not has_outliers else 85

            stats_text = f"Med: {median_val:.1f}%\nMean: {mean_val:.1f}%"
            if info["count"] > 0:
                stats_text += f"\n{info['count']} outliers"

            ax.text(
                i + 1,
                y_pos,
                stats_text,
                ha="center",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='none'),
            )

    plt.tight_layout()

    if save:
        # Save plot
        filename_parts = [f"parameter_breakdown_{layer_count}layer"]

        if use_prominent_features:
            filename_parts.append("prominent")

        filename = "_".join(filename_parts) + ".png"
        plot_file = Path(output_dir) / filename
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
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
    narrow_priors_deviation=0.99,
    failed_count=0,
    outlier_count=0,
    hide_title=False,
    minimal_stats=False,
):
    """
    Create all batch analysis plots (MAPE distribution, edge case detection, parameter breakdown).

    Args:
        batch_results: Dictionary of batch results from BatchInferencePipeline
        layer_count: Number of layers for plot titles
        output_dir: Directory to save plots
        save: Whether to save the plots
        use_prominent_features: Whether prominent features filtering was used
        narrow_priors_deviation: Deviation for narrow priors display in titles
        failed_count: Number of failed experiments (excluding outliers)
        outlier_count: Number of outlier experiments (excluded)
        hide_title: Hide plot titles (for publication)
        minimal_stats: Show minimal statistics (only mean and median)

    Returns:
        Dictionary with paths to saved plots
    """
    print("Creating batch analysis plots...")

    plot_paths = {}

    # Create MAPE distribution plot
    plot_paths["mape_distribution"] = plot_batch_mape_distribution(
        batch_results,
        layer_count,
        output_dir,
        save,
        narrow_priors_deviation,
        use_prominent_features,
        failed_count,
        outlier_count,
        hide_title,
        minimal_stats,
    )

    # Create edge case detection plot
    plot_paths["edge_case_detection"] = plot_batch_edge_case_detection(
        batch_results,
        layer_count,
        output_dir,
        save,
        use_prominent_features,
        failed_count,
        outlier_count,
        hide_title,
    )

    # Create parameter breakdown plot
    plot_paths["parameter_breakdown"] = plot_batch_parameter_breakdown(
        batch_results,
        layer_count,
        output_dir,
        save,
        narrow_priors_deviation,
        use_prominent_features,
        failed_count,
        outlier_count,
        hide_title,
    )

    print("Batch analysis plots completed!")
    return plot_paths
