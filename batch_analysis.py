#!/usr/bin/env python3
"""
Batch analysis utilities for reflectometry experiments.

This module contains functions for analyzing batch processing results,
calculating statistics, and detecting edge cases.
"""

import numpy as np
import logging


logger = logging.getLogger(__name__)


def _constraint_mape_values(results):
    values = []
    for result in results.values():
        if not result.get("success", True) or result.get("excluded_as_outlier"):
            continue
        metrics = result.get("param_metrics") or {}
        value = metrics.get("overall", {}).get("constraint_mape")
        if value is not None:
            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    "Successful results must contain finite, nonnegative constraint MAPE"
                )
            values.append(value)
    return values


def create_summary_statistics(
    successful_results,
    layer_count,
    enable_preprocessing=True,
    priors_type="narrow",
    narrow_priors_deviation=None,
):
    """Create summary statistics focused on constraint-based MAPE."""
    constraint_mape_values = _constraint_mape_values(successful_results)
    n_outliers = sum(
        bool(result.get("excluded_as_outlier"))
        for result in successful_results.values()
    )
    n_success = sum(
        bool(result.get("success", True)) and not result.get("excluded_as_outlier")
        for result in successful_results.values()
    )

    summary = {
        "total_experiments": len(successful_results),
        "successful_experiments": n_success,
        "failed_experiments": len(successful_results) - n_success - n_outliers,
        "excluded_experiments": n_outliers,
        "layer_count": layer_count,
        "preprocessing_enabled": enable_preprocessing,
        "priors_type": priors_type,
        "priors_deviation": narrow_priors_deviation,
        "narrow_priors_deviation": narrow_priors_deviation
        if priors_type == "narrow"
        else None,
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
    print(f"Total experiments: {summary['total_experiments']}")
    print(
        f"Successful experiments: {summary.get('successful_experiments', summary['total_experiments'])}"
    )
    print(f"Failed experiments: {summary.get('failed_experiments', 0)}")
    print(f"Excluded outliers: {summary.get('excluded_experiments', 0)}")
    print(f"Layer count: {summary['layer_count']}")
    print(f"Preprocessing enabled: {summary['preprocessing_enabled']}")
    print(f"Prior bounds: {summary.get('priors_type', 'unknown')}")

    deviation = summary.get("priors_deviation", summary.get("narrow_priors_deviation"))
    if deviation is not None:
        if summary.get("priors_type") == "constraint_based":
            print(
                f"Requested prior width: {deviation * 100:.1f}% of fixed evaluation range"
            )
        else:
            print(f"Prior bounds deviation: +/-{deviation * 100:.1f}%")

    if "constraint_accuracy" in summary and summary["constraint_accuracy"]:
        print("\nParameter Accuracy (constraint-based MAPE):")
        stats = summary["constraint_accuracy"]["constraint_mape"]
        print(f"  Median: {stats['median']:.2f}%")
        print(f"  Mean: {stats['mean']:.2f}% +/- {stats['std']:.2f}%")
        print(f"  Range: {stats['min']:.2f}% - {stats['max']:.2f}%")
        print(f"  Experiments: {stats['count']}")
    else:
        print("\nNo constraint-based MAPE data available")


def print_constraint_mape_summary(successful_results):
    """Print the constraint-based MAPE summary."""
    constraint_mape_values = _constraint_mape_values(successful_results)

    if not constraint_mape_values:
        print("\nNo constraint-based MAPE data available")
        return

    print("\nCONSTRAINT-BASED MAPE SUMMARY:")
    print("-" * 35)
    print(f"Experiments: {len(constraint_mape_values)}")
    print(
        f"Mean:   {np.mean(constraint_mape_values):.1f}% +/- {np.std(constraint_mape_values):.1f}%"
    )
    print(f"Median: {np.median(constraint_mape_values):.1f}%")
    print(
        f"Range:  {np.min(constraint_mape_values):.1f}% - {np.max(constraint_mape_values):.1f}%"
    )


def detect_edge_cases(successful_results):
    """Detect edge cases with poor constraint-based MAPE values."""
    edge_cases = []

    logger.debug("Edge case detection:")

    for exp_name, result in successful_results.items():
        if "param_metrics" not in result or not result["param_metrics"]:
            continue

        param_metrics = result["param_metrics"]

        overall_mape = param_metrics.get("overall", {}).get("constraint_mape")

        if overall_mape is not None:
            logger.debug("  %s: %.1f%% constraint MAPE", exp_name, overall_mape)

            # Flag as edge case if MAPE > 50%
            if overall_mape > 50:
                # Get individual parameter details if available
                thickness_mape = None
                roughness_mape = None
                sld_mape = None

                if "by_type" in param_metrics:
                    by_type = param_metrics["by_type"]
                    thickness_mape = by_type.get("thickness", {}).get(
                        "constraint_mape", 0
                    )
                    roughness_mape = by_type.get("roughness", {}).get(
                        "constraint_mape", 0
                    )
                    sld_mape = by_type.get("sld", {}).get("constraint_mape", 0)

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
        logger.warning(
            "Edge cases detected (%s experiments with constraint MAPE > 50%%):",
            len(edge_cases),
        )
        for i, case in enumerate(edge_cases[:5], 1):  # Show top 5 worst
            logger.warning("%s. %s", i, case["experiment"])
            logger.warning("   Overall constraint MAPE: %.1f%%", case["overall_mape"])
            if case["thickness_mape"] is not None:
                logger.warning("   Thickness: %.1f%%", case["thickness_mape"])
            if case["roughness_mape"] is not None:
                logger.warning("   Roughness: %.1f%%", case["roughness_mape"])
            if case["sld_mape"] is not None:
                logger.warning("   SLD: %.1f%%", case["sld_mape"])
    else:
        logger.info("No edge cases detected (all experiments < 50%% constraint MAPE)")

    return edge_cases
