#!/usr/bin/env python3
"""
Batch analysis utilities for reflectometry experiments.

This module contains functions for analyzing batch processing results,
calculating statistics, and detecting edge cases.
"""

import numpy as np


def create_summary_statistics(
    successful_results,
    layer_count,
    enable_preprocessing=True,
    priors_type="narrow",
    narrow_priors_deviation=None,
):
    """Create simplified summary statistics focused on MAPE with detailed debugging."""
    print("\nGenerating summary statistics...")

    # Collect MAPE values with detailed debugging
    mape_values = []
    debug_info = []

    for exp_id, result in successful_results.items():
        if "param_metrics" in result and result["param_metrics"]:
            param_metrics = result["param_metrics"]

            # Debug: Show what data structure we have
            print(f"\nDEBUG - Experiment {exp_id}:")
            print(f"  param_metrics keys: {list(param_metrics.keys())}")

            # Get standard MAPE from param_metrics.overall.mape
            overall_mape = None
            if "overall" in param_metrics and isinstance(
                param_metrics["overall"], dict
            ):
                if "mape" in param_metrics["overall"]:
                    overall_mape = param_metrics["overall"]["mape"]
                    print(f"  Found overall.mape: {overall_mape:.2f}%")

            if overall_mape is not None:
                mape_values.append(overall_mape)

                # Debug: Show by_type breakdown if available
                if "by_type" in param_metrics:
                    print("  Parameter breakdown:")
                    by_type = param_metrics["by_type"]
                    for param_type, metrics in by_type.items():
                        if isinstance(metrics, dict) and "mape" in metrics:
                            print(f"    {param_type}: {metrics['mape']:.2f}%")

                # Debug: Show individual parameter details if available
                if "by_parameter" in param_metrics:
                    print("  Individual parameters:")
                    for param_name, metrics in param_metrics["by_parameter"].items():
                        if isinstance(metrics, dict):
                            pred = metrics.get("predicted", "N/A")
                            true = metrics.get("true", "N/A")
                            rel_err = metrics.get("relative_error_percent", "N/A")
                            print(
                                f"    {param_name}: pred={pred}, true={true}, error={rel_err}%"
                            )

            # Don't add redundant overall_mape - it's already in param_metrics['overall']['mape']
            debug_info.append({"exp_id": exp_id, "param_metrics": param_metrics})

    print(
        f"\nCollected {len(mape_values)} MAPE values from {len(successful_results)} successful experiments"
    )

    summary = {
        "total_experiments": len(successful_results),
        "layer_count": layer_count,
        "preprocessing_enabled": enable_preprocessing,
        "priors_type": priors_type,
        "narrow_priors_deviation": narrow_priors_deviation
        if priors_type == "narrow"
        else None,
        "debug_info": debug_info,  # Add debug info to summary
    }

    if mape_values:
        summary["parameter_accuracy"] = {
            "overall_mape": {
                "median": float(np.median(mape_values)),
                "mean": float(np.mean(mape_values)),
                "std": float(np.std(mape_values)),
                "min": float(np.min(mape_values)),
                "max": float(np.max(mape_values)),
                "count": len(mape_values),
            }
        }

    return summary


def print_summary_statistics(summary):
    """Print simplified summary statistics focusing on MAPE."""
    print("\nBATCH PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total successful experiments: {summary['total_experiments']}")
    print(f"Layer count: {summary['layer_count']}")
    print(f"Preprocessing enabled: {summary['preprocessing_enabled']}")
    print(f"Prior bounds: {summary.get('priors_type', 'unknown')}")

    if summary.get("narrow_priors_deviation"):
        deviation_percent = summary["narrow_priors_deviation"] * 100
        print(f"Narrow priors deviation: ±{deviation_percent:.1f}%")

    if "parameter_accuracy" in summary and summary["parameter_accuracy"]:
        print("\nParameter Accuracy (MAPE):")
        mape_stats = summary["parameter_accuracy"]["overall_mape"]
        print(f"  Median: {mape_stats['median']:.2f}%")
        print(f"  Mean: {mape_stats['mean']:.2f}% ± {mape_stats['std']:.2f}%")
        print(f"  Range: {mape_stats['min']:.2f}% - {mape_stats['max']:.2f}%")
        print(f"  Experiments: {mape_stats['count']}")
    else:
        print("\nNo MAPE data available")


def print_mape_distribution(successful_results):
    """Print MAPE distribution summary using real overall MAPE values."""
    # Collect real overall MAPE values and constraint-based MAPE if available
    mape_values = []
    constraint_mape_values = []
    has_constraint_mape = False

    for result in successful_results.values():
        if "param_metrics" in result and result["param_metrics"]:
            param_metrics = result["param_metrics"]

            # Get standard MAPE from param_metrics.overall.mape
            overall_mape = None
            if "overall" in param_metrics and isinstance(
                param_metrics["overall"], dict
            ):
                if "mape" in param_metrics["overall"]:
                    overall_mape = param_metrics["overall"]["mape"]

            if overall_mape is not None:
                mape_values.append(overall_mape)

            # Get constraint-based MAPE if available
            constraint_mape = None
            if "overall" in param_metrics and isinstance(
                param_metrics["overall"], dict
            ):
                if "constraint_mape" in param_metrics["overall"]:
                    constraint_mape = param_metrics["overall"]["constraint_mape"]
                    has_constraint_mape = True

            if constraint_mape is not None:
                constraint_mape_values.append(constraint_mape)

    if not mape_values:
        print("\nNo MAPE data available")
        return

    print("\nREAL MAPE DISTRIBUTION:")
    print("-" * 35)

    # Count experiments in quality ranges
    excellent = sum(1 for mape in mape_values if mape < 5)
    good = sum(1 for mape in mape_values if 5 <= mape < 10)
    acceptable = sum(1 for mape in mape_values if 10 <= mape < 20)
    poor = sum(1 for mape in mape_values if mape >= 20)

    total = len(mape_values)
    print(f"Excellent (< 5%): {excellent} ({100 * excellent / total:.1f}%)")
    print(f"Good (5-10%): {good} ({100 * good / total:.1f}%)")
    print(f"Acceptable (10-20%): {acceptable} ({100 * acceptable / total:.1f}%)")
    print(f"Poor (≥ 20%): {poor} ({100 * poor / total:.1f}%)")

    print("\nStatistics:")
    print(f"Mean: {np.mean(mape_values):.1f}% ± {np.std(mape_values):.1f}%")
    print(f"Median: {np.median(mape_values):.1f}%")
    print(f"Range: {np.min(mape_values):.1f}% - {np.max(mape_values):.1f}%")

    # Print constraint-based MAPE distribution if available
    if has_constraint_mape and constraint_mape_values:
        print("\nCONSTRAINT-BASED MAPE DISTRIBUTION:")
        print("-" * 35)

        # Count experiments in quality ranges (using same thresholds for comparison)
        c_excellent = sum(1 for mape in constraint_mape_values if mape < 5)
        c_good = sum(1 for mape in constraint_mape_values if 5 <= mape < 10)
        c_acceptable = sum(1 for mape in constraint_mape_values if 10 <= mape < 20)
        c_poor = sum(1 for mape in constraint_mape_values if mape >= 20)

        c_total = len(constraint_mape_values)
        print(f"Excellent (< 5%): {c_excellent} ({100 * c_excellent / c_total:.1f}%)")
        print(f"Good (5-10%): {c_good} ({100 * c_good / c_total:.1f}%)")
        print(
            f"Acceptable (10-20%): {c_acceptable} ({100 * c_acceptable / c_total:.1f}%)"
        )
        print(f"Poor (≥ 20%): {c_poor} ({100 * c_poor / c_total:.1f}%)")

        print("\nStatistics:")
        print(
            f"Mean: {np.mean(constraint_mape_values):.1f}% ± {np.std(constraint_mape_values):.1f}%"
        )
        print(f"Median: {np.median(constraint_mape_values):.1f}%")
        print(
            f"Range: {np.min(constraint_mape_values):.1f}% - {np.max(constraint_mape_values):.1f}%"
        )


def detect_edge_cases(successful_results):
    """Detect edge cases with poor performance using real MAPE values."""
    edge_cases = []

    print("\nDEBUG - Edge case detection:")

    for exp_name, result in successful_results.items():
        if "param_metrics" not in result or not result["param_metrics"]:
            continue

        param_metrics = result["param_metrics"]

        # Get real overall MAPE
        overall_mape = None
        if "overall_mape" in param_metrics:
            overall_mape = param_metrics["overall_mape"]
        elif "overall" in param_metrics and isinstance(param_metrics["overall"], dict):
            if "mape" in param_metrics["overall"]:
                overall_mape = param_metrics["overall"]["mape"]

        if overall_mape is not None:
            print(f"  {exp_name}: {overall_mape:.1f}% MAPE")

            # Flag as edge case if MAPE > 50%
            if overall_mape > 50:
                # Get individual parameter details if available
                thickness_mape = None
                roughness_mape = None
                sld_mape = None

                if "by_type" in param_metrics:
                    by_type = param_metrics["by_type"]
                    thickness_mape = by_type.get("thickness", {}).get("mape", 0)
                    roughness_mape = by_type.get("roughness", {}).get("mape", 0)
                    sld_mape = by_type.get("sld", {}).get("mape", 0)

                edge_cases.append(
                    {
                        "experiment": exp_name,
                        "overall_mape": overall_mape,
                        "thickness_mape": thickness_mape,
                        "roughness_mape": roughness_mape,
                        "sld_mape": sld_mape,
                    }
                )

    # Sort by worst performance
    edge_cases.sort(key=lambda x: x["overall_mape"], reverse=True)

    if edge_cases:
        print(
            f"\n🚨 Edge Cases Detected ({len(edge_cases)} experiments with MAPE > 50%):"
        )
        print("-" * 80)
        for i, case in enumerate(edge_cases[:5], 1):  # Show top 5 worst
            print(f"{i}. {case['experiment']}")
            print(f"   Overall MAPE: {case['overall_mape']:.1f}%")
            if case["thickness_mape"] is not None:
                print(f"   Thickness: {case['thickness_mape']:.1f}%")
            if case["roughness_mape"] is not None:
                print(f"   Roughness: {case['roughness_mape']:.1f}%")
            if case["sld_mape"] is not None:
                print(f"   SLD: {case['sld_mape']:.1f}%")
            print()
    else:
        print("\nNo edge cases detected (all experiments < 50% MAPE)")

    return edge_cases


if __name__ == "__main__":
    print("Batch analysis module loaded successfully.")
    print("Available functions:")
    print("  - create_summary_statistics()")
    print("  - print_summary_statistics()")
    print("  - print_mape_distribution()")
    print("  - detect_edge_cases()")
    print("Note: Plotting functions are available in plotting_utils module")
