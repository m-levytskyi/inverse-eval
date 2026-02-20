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
    """Create summary statistics focused on MAPE."""
    mape_values = []
    constraint_mape_values = []

    for exp_id, result in successful_results.items():
        if "param_metrics" in result and result["param_metrics"]:
            param_metrics = result["param_metrics"]
            overall = param_metrics.get("overall", {})
            if isinstance(overall, dict):
                if "mape" in overall:
                    mape_values.append(overall["mape"])
                if "constraint_mape" in overall:
                    constraint_mape_values.append(overall["constraint_mape"])

    summary = {
        "total_experiments": len(successful_results),
        "layer_count": layer_count,
        "preprocessing_enabled": enable_preprocessing,
        "priors_type": priors_type,
        "narrow_priors_deviation": narrow_priors_deviation
        if priors_type == "narrow"
        else None,
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

    if constraint_mape_values:
        summary["constraint_accuracy"] = {
            "constraint_mape": {
                "median": float(np.median(constraint_mape_values)),
                "mean": float(np.mean(constraint_mape_values)),
                "std": float(np.std(constraint_mape_values)),
                "min": float(np.min(constraint_mape_values)),
                "max": float(np.max(constraint_mape_values)),
                "count": len(constraint_mape_values),
            }
        }

    return summary


def print_summary_statistics(summary):
    """Print summary statistics focusing on constraint-based MAPE."""
    print("\nBATCH PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total successful experiments: {summary['total_experiments']}")
    print(f"Layer count: {summary['layer_count']}")
    print(f"Preprocessing enabled: {summary['preprocessing_enabled']}")
    print(f"Prior bounds: {summary.get('priors_type', 'unknown')}")

    if summary.get("narrow_priors_deviation"):
        deviation_percent = summary["narrow_priors_deviation"] * 100
        print(f"Narrow priors deviation: +/-{deviation_percent:.1f}%")

    if "constraint_accuracy" in summary and summary["constraint_accuracy"]:
        print("\nParameter Accuracy (constraint-based MAPE):")
        stats = summary["constraint_accuracy"]["constraint_mape"]
        print(f"  Median: {stats['median']:.2f}%")
        print(f"  Mean: {stats['mean']:.2f}% +/- {stats['std']:.2f}%")
        print(f"  Range: {stats['min']:.2f}% - {stats['max']:.2f}%")
        print(f"  Experiments: {stats['count']}")
    elif "parameter_accuracy" in summary and summary["parameter_accuracy"]:
        print("\nParameter Accuracy (MAPE):")
        stats = summary["parameter_accuracy"]["overall_mape"]
        print(f"  Median: {stats['median']:.2f}%")
        print(f"  Mean: {stats['mean']:.2f}% +/- {stats['std']:.2f}%")
        print(f"  Range: {stats['min']:.2f}% - {stats['max']:.2f}%")
        print(f"  Experiments: {stats['count']}")
    else:
        print("\nNo MAPE data available")


def print_mape_distribution(successful_results, show_traditional=False):
    """Print MAPE distribution summary.

    Args:
        successful_results: dict of successful experiment results
        show_traditional: if True, also print the standard (non-constraint) MAPE section
    """
    mape_values = []
    constraint_mape_values = []

    for result in successful_results.values():
        if "param_metrics" in result and result["param_metrics"]:
            overall = result["param_metrics"].get("overall", {})
            if isinstance(overall, dict):
                if "mape" in overall:
                    mape_values.append(overall["mape"])
                if "constraint_mape" in overall:
                    constraint_mape_values.append(overall["constraint_mape"])

    if not mape_values and not constraint_mape_values:
        print("\nNo MAPE data available")
        return

    if show_traditional and mape_values:
        print("\nMAP DISTRIBUTION (standard):")
        print("-" * 35)
        total = len(mape_values)
        excellent = sum(1 for m in mape_values if m < 5)
        good = sum(1 for m in mape_values if 5 <= m < 10)
        acceptable = sum(1 for m in mape_values if 10 <= m < 20)
        poor = sum(1 for m in mape_values if m >= 20)
        print(f"Excellent (< 5%):    {excellent} ({100 * excellent / total:.1f}%)")
        print(f"Good (5-10%):        {good} ({100 * good / total:.1f}%)")
        print(f"Acceptable (10-20%): {acceptable} ({100 * acceptable / total:.1f}%)")
        print(f"Poor (>= 20%):       {poor} ({100 * poor / total:.1f}%)")
        print("\nStatistics:")
        print(f"Mean:   {np.mean(mape_values):.1f}% +/- {np.std(mape_values):.1f}%")
        print(f"Median: {np.median(mape_values):.1f}%")
        print(f"Range:  {np.min(mape_values):.1f}% - {np.max(mape_values):.1f}%")

    if constraint_mape_values:
        c_total = len(constraint_mape_values)
        c_excellent = sum(1 for m in constraint_mape_values if m < 5)
        c_good = sum(1 for m in constraint_mape_values if 5 <= m < 10)
        c_acceptable = sum(1 for m in constraint_mape_values if 10 <= m < 20)
        c_poor = sum(1 for m in constraint_mape_values if m >= 20)
        print("\nCONSTRAINT-BASED MAPE DISTRIBUTION:")
        print("-" * 35)
        print(f"Excellent (< 5%):    {c_excellent} ({100 * c_excellent / c_total:.1f}%)")
        print(f"Good (5-10%):        {c_good} ({100 * c_good / c_total:.1f}%)")
        print(f"Acceptable (10-20%): {c_acceptable} ({100 * c_acceptable / c_total:.1f}%)")
        print(f"Poor (>= 20%):       {c_poor} ({100 * c_poor / c_total:.1f}%)")
        print("\nStatistics:")
        print(f"Mean:   {np.mean(constraint_mape_values):.1f}% +/- {np.std(constraint_mape_values):.1f}%")
        print(f"Median: {np.median(constraint_mape_values):.1f}%")
        print(f"Range:  {np.min(constraint_mape_values):.1f}% - {np.max(constraint_mape_values):.1f}%")


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
