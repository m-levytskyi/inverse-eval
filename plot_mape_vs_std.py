#!/usr/bin/env python3
"""
Plot constraint-based MAPE vs Standard Deviation from NF samples.

This script analyzes the relationship between prediction uncertainty (measured by
standard deviation across NF samples) and prediction accuracy (measured by MAPE).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Apply paper plotting style
plt.style.use("paper.mplstyle")


def load_batch_results(results_file):
    """Load batch results from JSON file."""
    print(f"Loading results from: {results_file}")
    with open(results_file, "r") as f:
        return json.load(f)


def extract_mape_std_data(results):
    """
    Extract MAPE and constraint-normalized standard deviation data from batch results.

    Returns:
        dict: Dictionary with parameter names as keys, each containing:
            - constraint_mape: list of constraint-based MAPE values
            - constraint_std: list of constraint-normalized std dev values (%)
            - raw_std: list of raw standard deviation values
            - exp_ids: list of experiment IDs
    """
    # Initialize data structure
    data = {
        "overall": {"constraint_mape": [], "constraint_std_mean": [], "exp_ids": []},
        "by_parameter": {},
    }

    # Parameter name mapping (NF model names -> canonical names)
    param_map = {
        "Thickness L1": "thickness",
        "Roughness L1": "amb_rough",
        "Roughness sub": "sub_rough",
        "SLD L1": "layer_sld",
        "SLD sub": "sub_sld",
    }

    for exp_id, exp_data in results.items():
        # Skip failed experiments and outliers
        if not exp_data.get("success") or exp_data.get("excluded_as_outlier"):
            continue

        # Skip if missing required data
        if "param_metrics" not in exp_data or "prediction_dict" not in exp_data:
            continue

        pred_dict = exp_data["prediction_dict"]
        param_metrics = exp_data["param_metrics"]

        # Skip if NF statistics not available
        if "nf_params_std" not in pred_dict or "nf_params_mean" not in pred_dict:
            continue

        # Extract overall constraint MAPE and constraint-normalized mean std
        overall_mape = param_metrics["overall"].get("constraint_mape")
        if overall_mape is not None:
            std_values = pred_dict["nf_params_std"]
            param_names = pred_dict.get("param_names", [])
            by_param = param_metrics.get("by_parameter", {})

            # Calculate constraint-normalized std for each physical parameter
            constraint_std_values = []
            for i, nf_param_name in enumerate(param_names):
                canonical_name = param_map.get(nf_param_name)
                if canonical_name is None:
                    continue  # Skip nuisance parameters

                param_data = by_param.get(canonical_name, {})
                constraint_width = param_data.get("constraint_width")

                if constraint_width is not None and constraint_width > 0:
                    # Normalize std by constraint width: (std / constraint_width) * 100
                    constraint_std = (std_values[i] / constraint_width) * 100
                    constraint_std_values.append(constraint_std)

            # Mean constraint-normalized std across all physical parameters
            if len(constraint_std_values) > 0:
                mean_constraint_std = np.mean(constraint_std_values)
                data["overall"]["constraint_mape"].append(overall_mape)
                data["overall"]["constraint_std_mean"].append(mean_constraint_std)
                data["overall"]["exp_ids"].append(exp_id)

        # Extract per-parameter data
        param_names = pred_dict.get("param_names", [])
        std_values = pred_dict["nf_params_std"]
        by_param = param_metrics.get("by_parameter", {})

        for i, nf_param_name in enumerate(param_names):
            # Map to canonical name
            canonical_name = param_map.get(nf_param_name)
            if canonical_name is None:
                continue  # Skip nuisance parameters

            if canonical_name not in data["by_parameter"]:
                data["by_parameter"][canonical_name] = {
                    "constraint_mape": [],
                    "constraint_std": [],
                    "raw_std": [],
                    "exp_ids": [],
                }

            # Get MAPE and constraint width for this parameter
            param_data = by_param.get(canonical_name, {})
            param_mape = param_data.get("constraint_percentage_error")
            constraint_width = param_data.get("constraint_width")

            if (
                param_mape is not None
                and constraint_width is not None
                and constraint_width > 0
            ):
                # Calculate constraint-normalized std
                constraint_std = (std_values[i] / constraint_width) * 100

                data["by_parameter"][canonical_name]["constraint_mape"].append(
                    abs(param_mape)
                )
                data["by_parameter"][canonical_name]["constraint_std"].append(
                    constraint_std
                )
                data["by_parameter"][canonical_name]["raw_std"].append(std_values[i])
                data["by_parameter"][canonical_name]["exp_ids"].append(exp_id)

    print(f"Extracted data for {len(data['overall']['constraint_mape'])} experiments")
    print(f"Parameters: {list(data['by_parameter'].keys())}")

    return data


def plot_mape_vs_std(data, output_dir):
    """Create plots showing MAPE vs Constraint-Normalized Standard Deviation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Overall MAPE vs Mean Constraint-Normalized Std
    fig, ax = plt.subplots(figsize=(10, 7))

    mape = np.array(data["overall"]["constraint_mape"])
    std = np.array(data["overall"]["constraint_std_mean"])

    scatter = ax.scatter(std, mape, alpha=0.5, s=20, c=mape, cmap="viridis")
    ax.set_xlabel("Mean Constraint-Normalized Std Dev (%)", fontsize=12)
    ax.set_ylabel("Overall Constraint-Based MAPE (%)", fontsize=12)
    ax.set_title(
        "Prediction Uncertainty vs Accuracy (Normalized Units)\nConstraint-Based Priors (30%)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("MAPE (%)", fontsize=11)

    # Add correlation coefficient
    if len(std) > 1:
        corr = np.corrcoef(std, mape)[0, 1]
        ax.text(
            0.05,
            0.95,
            f"Correlation: {corr:.3f}\nn = {len(std)}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    output_file = output_dir / "mape_vs_std_overall.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()

    # 2. Overall MAPE vs Mean Constraint-Normalized Std with Binning
    fig, ax = plt.subplots(figsize=(10, 7))

    mape = np.array(data["overall"]["constraint_mape"])
    std = np.array(data["overall"]["constraint_std_mean"])

    # Sort by std
    sorted_indices = np.argsort(std)
    sorted_std = std[sorted_indices]
    sorted_mape = mape[sorted_indices]

    # Create 10 bins
    n_bins = 10
    bin_size = len(sorted_std) // n_bins

    bin_std_means = []
    bin_mape_means = []
    bin_mape_stds = []

    for i in range(n_bins):
        start_idx = i * bin_size
        if i == n_bins - 1:
            # Last bin gets remaining samples
            end_idx = len(sorted_std)
        else:
            end_idx = (i + 1) * bin_size

        bin_std_values = sorted_std[start_idx:end_idx]
        bin_mape_values = sorted_mape[start_idx:end_idx]

        bin_std_means.append(np.mean(bin_std_values))
        bin_mape_means.append(np.mean(bin_mape_values))
        bin_mape_stds.append(np.std(bin_mape_values))

    bin_std_means = np.array(bin_std_means)
    bin_mape_means = np.array(bin_mape_means)
    bin_mape_stds = np.array(bin_mape_stds)

    # Plot scatter (all points, faded)
    ax.scatter(std, mape, alpha=0.15, s=15, c="gray", label="Individual experiments")

    # Plot binned data
    ax.errorbar(
        bin_std_means,
        bin_mape_means,
        yerr=bin_mape_stds,
        fmt="o-",
        markersize=8,
        linewidth=2,
        color="red",
        ecolor="red",
        capsize=5,
        capthick=2,
        label="Binned average (10 bins)",
        zorder=10,
    )

    ax.set_xlabel("Mean Constraint-Normalized Std Dev (%)", fontsize=12)
    ax.set_ylabel("Overall Constraint-Based MAPE (%)", fontsize=12)
    ax.set_title(
        "Prediction Uncertainty vs Accuracy (Binned, Normalized Units)\nConstraint-Based Priors (30%)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best")

    # Add correlation coefficient for binned data
    if len(bin_std_means) > 1:
        corr_binned = np.corrcoef(bin_std_means, bin_mape_means)[0, 1]
        corr_raw = np.corrcoef(std, mape)[0, 1]
        ax.text(
            0.05,
            0.95,
            f"Correlation (raw): {corr_raw:.3f}\nCorrelation (binned): {corr_binned:.3f}\nBin size: {bin_size}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    output_file = output_dir / "mape_vs_std_overall_binned.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()

    # 3. Per-parameter plots (individual)
    param_labels = {
        "thickness": "Thickness",
        "amb_rough": "Ambient Roughness",
        "sub_rough": "Substrate Roughness",
        "layer_sld": "Layer SLD",
        "sub_sld": "Substrate SLD",
    }

    for param_name, param_data in data["by_parameter"].items():
        if len(param_data["constraint_mape"]) == 0:
            continue

        fig, ax = plt.subplots(figsize=(10, 7))

        mape = np.array(param_data["constraint_mape"])
        std = np.array(param_data["constraint_std"])

        scatter = ax.scatter(std, mape, alpha=0.5, s=20, c=mape, cmap="viridis")
        ax.set_xlabel(
            f"{param_labels.get(param_name, param_name)} Constraint-Normalized Std Dev (%)",
            fontsize=12,
        )
        ax.set_ylabel(
            f"{param_labels.get(param_name, param_name)} Constraint MAPE (%)",
            fontsize=12,
        )
        ax.set_title(
            f"Uncertainty vs Accuracy: {param_labels.get(param_name, param_name)}\nConstraint-Based Priors (30%)",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("MAPE (%)", fontsize=11)

        # Add correlation coefficient
        if len(std) > 1:
            corr = np.corrcoef(std, mape)[0, 1]
            ax.text(
                0.05,
                0.95,
                f"Correlation: {corr:.3f}\nn = {len(std)}",
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        output_file = output_dir / f"mape_vs_std_{param_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_file}")
        plt.close()

    # 4. Combined per-parameter plot (all in one)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    param_names = ["thickness", "amb_rough", "sub_rough", "layer_sld", "sub_sld"]

    for idx, param_name in enumerate(param_names):
        ax = axes[idx]
        param_data = data["by_parameter"].get(param_name, {})

        if len(param_data.get("constraint_mape", [])) == 0:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(param_labels.get(param_name, param_name))
            continue

        mape = np.array(param_data["constraint_mape"])
        std = np.array(param_data["constraint_std"])

        scatter = ax.scatter(std, mape, alpha=0.5, s=15, c=mape, cmap="viridis")
        ax.set_xlabel("Constraint-Norm. Std (%)", fontsize=10)
        ax.set_ylabel("Constraint MAPE (%)", fontsize=10)
        ax.set_title(
            param_labels.get(param_name, param_name), fontsize=11, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)

        # Add correlation
        if len(std) > 1:
            corr = np.corrcoef(std, mape)[0, 1]
            ax.text(
                0.05,
                0.95,
                f"r={corr:.2f}",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

    # Remove the extra subplot
    fig.delaxes(axes[5])

    plt.suptitle(
        "Uncertainty vs Accuracy by Parameter\nConstraint-Based Priors (30%)",
        fontsize=16,
        fontweight="bold",
    )
    output_file = output_dir / "mape_vs_std_all_parameters.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()

    # 5. Statistical summary
    print("\n" + "=" * 70)
    print("STATISTICAL SUMMARY: MAPE vs Standard Deviation")
    print("=" * 70)

    print("\nOverall:")
    mape = np.array(data["overall"]["constraint_mape"])
    std = np.array(data["overall"]["constraint_std_mean"])
    corr = np.corrcoef(std, mape)[0, 1] if len(std) > 1 else np.nan
    print(f"  Samples: {len(std)}")
    print(f"  Mean MAPE: {np.mean(mape):.2f}%")
    print(f"  Mean Constraint-Normalized Std: {np.mean(std):.4f}%")
    print(f"  Correlation: {corr:.4f}")

    print("\nBy Parameter:")
    for param_name in ["thickness", "amb_rough", "sub_rough", "layer_sld", "sub_sld"]:
        param_data = data["by_parameter"].get(param_name, {})
        if len(param_data.get("constraint_mape", [])) == 0:
            continue

        mape = np.array(param_data["constraint_mape"])
        std = np.array(param_data["constraint_std"])
        corr = np.corrcoef(std, mape)[0, 1] if len(std) > 1 else np.nan

        print(f"\n  {param_labels.get(param_name, param_name)}:")
        print(f"    Samples: {len(std)}")
        print(f"    Mean MAPE: {np.mean(mape):.2f}%")
        print(f"    Mean Constraint-Normalized Std: {np.mean(std):.4f}%")
        print(f"    Correlation: {corr:.4f}")


def extract_coverage_data(results):
    """
    Compute credible-interval coverage from NF percentiles vs true values.

    Uses all five saved percentile levels (5, 25, 50, 75, 95) as CDF
    calibration points.  For each level k, the empirical coverage is the
    fraction of experiments where true_value <= p_k.

    Args:
        results: Batch results dictionary (experiment_id -> result dict).

    Returns:
        dict with keys:
            - "by_parameter": {param_name: {"n": int,
                                             "empirical": [float x5]}}
            - "average": {"empirical": [float x5], "n_params": int}
            - "nominal": [5, 25, 50, 75, 95]
    """
    # Percentile levels stored in nf_params_percentiles (row order)
    nominal_levels = [5, 25, 50, 75, 95]
    n_levels = len(nominal_levels)

    # Parameter name mapping (NF model names -> canonical names)
    param_map = {
        "Thickness L1": "thickness",
        "Roughness L1": "amb_rough",
        "Roughness sub": "sub_rough",
        "SLD L1": "layer_sld",
        "SLD sub": "sub_sld",
    }

    # Accumulators: param -> {"below": [int x5], "total": int}
    counts = {}

    for exp_id, exp_data in results.items():
        if not exp_data.get("success") or exp_data.get("excluded_as_outlier"):
            continue

        pred_dict = exp_data.get("prediction_dict", {})
        percentiles = pred_dict.get("nf_params_percentiles")
        param_names = pred_dict.get("param_names", [])

        if percentiles is None or len(percentiles) != n_levels:
            continue

        # Resolve true values -- try true_params_dict first, then true_params
        true_params = exp_data.get("true_params_dict") or exp_data.get(
            "true_params", {}
        )
        layer_count = exp_data.get("layer_count", 1)
        layer_key = f"{layer_count}_layer"
        true_entry = true_params.get(layer_key, {})
        true_values = true_entry.get("params", [])
        true_names = true_entry.get("param_names", [])

        # Fallback: use param_metrics.by_parameter which has "true" per param
        if not true_values or not true_names:
            by_param = exp_data.get("param_metrics", {}).get("by_parameter", {})
            if by_param:
                true_names = list(by_param.keys())
                true_values = [by_param[k].get("true") for k in true_names]
                true_values = [v for v in true_values if v is not None]
                if len(true_values) != len(true_names):
                    continue

        if not true_values or not true_names:
            continue

        true_map = dict(zip(true_names, true_values))

        for i, nf_name in enumerate(param_names):
            canonical = param_map.get(nf_name)
            if canonical is None:
                continue  # skip nuisance params

            true_val = true_map.get(canonical)
            if true_val is None:
                continue

            if canonical not in counts:
                counts[canonical] = {"below": [0] * n_levels, "total": 0}

            counts[canonical]["total"] += 1
            for lev_idx in range(n_levels):
                if true_val <= percentiles[lev_idx][i]:
                    counts[canonical]["below"][lev_idx] += 1

    # Build output
    by_parameter = {}
    sum_empirical = [0.0] * n_levels
    n_params = 0
    for param, c in counts.items():
        if c["total"] == 0:
            continue
        empirical = [100.0 * c["below"][j] / c["total"] for j in range(n_levels)]
        by_parameter[param] = {"n": c["total"], "empirical": empirical}
        for j in range(n_levels):
            sum_empirical[j] += empirical[j]
        n_params += 1

    average = {}
    if n_params > 0:
        average = {
            "empirical": [s / n_params for s in sum_empirical],
            "n_params": n_params,
        }

    print(f"Coverage computed for {n_params} parameters")
    for p, d in by_parameter.items():
        parts = ", ".join(
            f"{nl}%={e:.1f}%" for nl, e in zip(nominal_levels, d["empirical"])
        )
        print(f"  {p}: n={d['n']}, {parts}")
    if average:
        parts = ", ".join(
            f"{nl}%={e:.1f}%" for nl, e in zip(nominal_levels, average["empirical"])
        )
        print(f"  Average: {parts}")

    return {
        "by_parameter": by_parameter,
        "average": average,
        "nominal": nominal_levels,
    }


def plot_coverage(coverage_data, output_dir):
    """
    Plot nominal vs empirical credible-interval coverage.

    Produces a single plot with one curve per parameter plus the average,
    overlaid on the ideal (diagonal) line.  Uses all saved percentile
    levels (5, 25, 50, 75, 95) as calibration points.

    Args:
        coverage_data: Output of extract_coverage_data().
        output_dir: Directory to save the plot.

    Returns:
        Path to the saved plot file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    param_labels = {
        "thickness": "Thickness",
        "amb_rough": "Ambient Roughness",
        "sub_rough": "Substrate Roughness",
        "layer_sld": "Layer SLD",
        "sub_sld": "Substrate SLD",
    }

    nominal = np.array(coverage_data["nominal"])

    fig, ax = plt.subplots(figsize=(7, 7))

    # Ideal diagonal
    ax.plot([0, 100], [0, 100], "k--", linewidth=1, label="Ideal")

    markers = ["o", "s", "^", "D", "v"]
    for idx, (param, data) in enumerate(sorted(coverage_data["by_parameter"].items())):
        empirical = np.array(data["empirical"])
        label = f"{param_labels.get(param, param)}"
        ax.plot(
            nominal,
            empirical,
            marker=markers[idx % len(markers)],
            markersize=8,
            linewidth=1.5,
            label=label,
        )

    # Average curve
    avg = coverage_data.get("average", {})
    if avg:
        avg_empirical = np.array(avg["empirical"])
        ax.plot(
            nominal,
            avg_empirical,
            "k-o",
            markersize=10,
            linewidth=2.5,
            label="Average",
            zorder=20,
        )

    ax.set_xlabel("Nominal Coverage (%)", fontsize=12)
    ax.set_ylabel("Empirical Coverage (%)", fontsize=12)
    ax.set_title("Credible-Interval Coverage", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.grid(True, alpha=0.3)

    plot_file = output_dir / "coverage_nominal_vs_empirical.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Coverage plot saved to: {plot_file}")
    return plot_file


def main():
    """Main function."""
    # Path to batch results
    results_file = "batch_inference_results/288_951exps_1layers_30constraint_06february2026_17_59/batch_results.json"

    # Output directory
    output_dir = "batch_inference_results/288_951exps_1layers_30constraint_06february2026_17_59/mape_std_analysis"

    # Load results
    results = load_batch_results(results_file)

    # Extract data
    data = extract_mape_std_data(results)

    # Create plots
    plot_mape_vs_std(data, output_dir)

    print(f"\nAnalysis complete! Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
