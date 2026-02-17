#!/usr/bin/env python3
"""
Compare two model versions by plotting MAPE distributions.

This script compares NF model performance between:
- Batches 233-250: Q-weighted model trained on mixed (generated + experimental) data
- Batches 269-286: Baseline model trained on generated data only

The baseline model appears as grey background bars, q-weighted as colored foreground bars.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from plotting_utils import plot_model_comparison_histogram, plot_parameter_comparison_grid


def load_batch_summary(batch_dir: Path) -> Dict:
    """
    Load batch summary from JSON file.

    Args:
        batch_dir: Path to batch directory

    Returns:
        Dictionary with batch summary data
    """
    summary_file = batch_dir / "batch_summary_1layer.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")

    with open(summary_file, "r") as f:
        return json.load(f)


def parse_batch_config(batch_name: str) -> Dict:
    """
    Parse batch configuration from directory name.

    Args:
        batch_name: Batch directory name (e.g., "233_951exps_1layers_5constraint_10january2026_18_23")

    Returns:
        Dictionary with parsed configuration
    """
    parts = batch_name.split("_")

    config = {
        "batch_num": int(parts[0]),
        "num_exps": int(parts[1].replace("exps", "")),
        "layer_count": int(parts[2].replace("layers", "")),
        "prominent": "PROMINENT" in batch_name.upper(),
        "sld_fix_mode": "none",
    }

    # Parse constraint/prior deviation
    for part in parts:
        if "constraint" in part.lower():
            constraint_val = int(part.replace("constraint", ""))
            config["constraint"] = constraint_val
            config["deviation"] = constraint_val / 100.0
            break

    # Parse SLD fix mode
    if "allSLDfix" in batch_name or "allsldfix" in batch_name.lower():
        config["sld_fix_mode"] = "all"
    elif "backSLDfix" in batch_name or "backsldfix" in batch_name.lower():
        config["sld_fix_mode"] = "backing"

    return config


def extract_mapes_from_batch(
    batch_dir: Path, use_constraint_mape: bool = True
) -> Tuple[List[float], Dict]:
    """
    Extract MAPE values from batch summary.

    Args:
        batch_dir: Path to batch directory
        use_constraint_mape: Whether to use constraint-based MAPE (default: True)

    Returns:
        Tuple of (mape_list, per_param_mapes_dict)
    """
    summary = load_batch_summary(batch_dir)

    mapes = []
    per_param_mapes = defaultdict(list)

    if "debug_info" not in summary:
        return mapes, per_param_mapes

    for exp_info in summary["debug_info"]:
        if "param_metrics" not in exp_info or not exp_info["param_metrics"]:
            continue

        # Extract overall MAPE
        if use_constraint_mape and "overall" in exp_info["param_metrics"]:
            if "constraint_mape" in exp_info["param_metrics"]["overall"]:
                mape = exp_info["param_metrics"]["overall"]["constraint_mape"]
                mapes.append(mape)
        elif "overall" in exp_info["param_metrics"]:
            if "mape" in exp_info["param_metrics"]["overall"]:
                mape = exp_info["param_metrics"]["overall"]["mape"]
                mapes.append(mape)

        # Extract per-parameter MAPE
        if "by_parameter" in exp_info["param_metrics"]:
            for param_name, param_data in exp_info["param_metrics"][
                "by_parameter"
            ].items():
                if use_constraint_mape and "constraint_percentage_error" in param_data:
                    per_param_mapes[param_name].append(
                        param_data["constraint_percentage_error"]
                    )
                elif "percentage_error" in param_data:
                    per_param_mapes[param_name].append(param_data["percentage_error"])

    return mapes, dict(per_param_mapes)


def get_batch_metadata(batch_dir: Path) -> Dict:
    """
    Get batch metadata including failure and outlier counts.

    Args:
        batch_dir: Path to batch directory

    Returns:
        Dictionary with metadata
    """
    summary = load_batch_summary(batch_dir)

    total = summary.get("total_experiments", 0)
    successful = 0
    outliers = 0

    if "debug_info" in summary:
        for exp_info in summary["debug_info"]:
            if exp_info.get("param_metrics"):
                successful += 1
                if exp_info.get("excluded_as_outlier", False):
                    outliers += 1

    return {
        "total": total,
        "successful": successful,
        "failed": total - successful,
        "outliers": outliers,
        "priors_type": summary.get("priors_type", "unknown"),
    }


def _plot_model_comparison_histogram_old(
    baseline_mapes: List[float],
    qweighted_mapes: List[float],
    config: Dict,
    output_path: Path,
    baseline_meta: Dict,
    qweighted_meta: Dict,
):
    """
    DEPRECATED: Use plotting_utils.plot_model_comparison_histogram instead.
    
    Create comparison histogram with baseline (grey) and q-weighted (colored) MAPEs.

    Args:
        baseline_mapes: List of baseline model MAPE values
        qweighted_mapes: List of q-weighted model MAPE values
        config: Configuration dictionary (constraint, sld_fix_mode, prominent, etc.)
        output_path: Path to save the plot
        baseline_meta: Metadata for baseline model
        qweighted_meta: Metadata for q-weighted model
    """
    from plotting_utils import plot_model_comparison_histogram as plot_comparison
    
    plot_comparison(
        baseline_mapes=baseline_mapes,
        comparison_mapes=qweighted_mapes,
        config=config,
        output_dir=output_path.parent,
        save=True,
        baseline_label="Baseline (Generated Data Only)",
        comparison_label="Q-Weighted (Mixed Data)",
        baseline_meta=baseline_meta,
        comparison_meta=qweighted_meta,
    )


def _plot_param_comparison_old(
    baseline_per_param: Dict[str, List[float]],
    qweighted_per_param: Dict[str, List[float]],
    config: Dict,
    output_path: Path,
    baseline_meta: Dict,
    qweighted_meta: Dict,
):
    """
    DEPRECATED: Use plotting_utils.plot_parameter_comparison_grid instead.
    
    Create per-parameter MAPE comparison plots.

    Args:
        baseline_per_param: Dictionary of parameter name -> list of MAPEs for baseline
        qweighted_per_param: Dictionary of parameter name -> list of MAPEs for q-weighted
        config: Configuration dictionary
        output_path: Path to save the plot
        baseline_meta: Metadata for baseline model
        qweighted_meta: Metadata for q-weighted model
    """
    from plotting_utils import plot_parameter_comparison_grid
    
    plot_parameter_comparison_grid(
        baseline_per_param=baseline_per_param,
        comparison_per_param=qweighted_per_param,
        config=config,
        output_dir=output_path.parent,
        save=True,
        baseline_label="Baseline",
        comparison_label="Q-Weighted",
        priors_type=baseline_meta.get("priors_type", "constraint_based"),
    )


def find_matching_batches(
    baseline_batches: List[Path], qweighted_batches: List[Path]
) -> List[Tuple[Path, Path, Dict]]:
    """
    Find matching batch pairs between baseline and q-weighted runs.

    Args:
        baseline_batches: List of baseline batch directories
        qweighted_batches: List of q-weighted batch directories

    Returns:
        List of tuples (baseline_path, qweighted_path, config)
    """
    # Parse configurations
    baseline_configs = [(b, parse_batch_config(b.name)) for b in baseline_batches]
    qweighted_configs = [(q, parse_batch_config(q.name)) for q in qweighted_batches]

    # Match by constraint level, SLD fix mode, and prominent flag
    matches = []

    for baseline_path, baseline_config in baseline_configs:
        for qweighted_path, qweighted_config in qweighted_configs:
            if (
                baseline_config["constraint"] == qweighted_config["constraint"]
                and baseline_config["sld_fix_mode"] == qweighted_config["sld_fix_mode"]
                and baseline_config["prominent"] == qweighted_config["prominent"]
            ):
                # Use the q-weighted config for labeling (they should be identical)
                matches.append((baseline_path, qweighted_path, qweighted_config))
                break

    return matches


def generate_comparison_plots(
    baseline_range: Tuple[int, int],
    qweighted_range: Tuple[int, int],
    base_dir: Path,
    output_dir: Path,
):
    """
    Generate all comparison plots for the two model versions.

    Args:
        baseline_range: Tuple of (start, end) batch numbers for baseline model
        qweighted_range: Tuple of (start, end) batch numbers for q-weighted model
        base_dir: Base directory containing batch results
        output_dir: Output directory for plots
    """
    print(f"\n{'=' * 80}")
    print(f"MODEL COMPARISON: Q-Weighted vs Baseline")
    print(f"{'=' * 80}")
    print(
        f"Baseline batches: {baseline_range[0]}-{baseline_range[1]} (Generated data only)"
    )
    print(f"Q-Weighted batches: {qweighted_range[0]}-{qweighted_range[1]} (Mixed data)")
    print()

    # Find batch directories
    all_batches = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name[0].isdigit()]
    )

    baseline_batches = []
    qweighted_batches = []

    for batch_dir in all_batches:
        batch_num = int(batch_dir.name.split("_")[0])
        if baseline_range[0] <= batch_num <= baseline_range[1]:
            baseline_batches.append(batch_dir)
        elif qweighted_range[0] <= batch_num <= qweighted_range[1]:
            qweighted_batches.append(batch_dir)

    print(f"Found {len(baseline_batches)} baseline batches")
    print(f"Found {len(qweighted_batches)} q-weighted batches")
    print()

    # Find matching pairs
    matches = find_matching_batches(baseline_batches, qweighted_batches)
    print(f"Found {len(matches)} matching batch pairs")
    print()

    # Create output directory structure
    for subdir in ["mape", "mape_prom", "param_mape", "param_mape_prom"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Generate plots for each matching pair
    for baseline_path, qweighted_path, config in matches:
        print(f"\nProcessing pair:")
        print(f"  Baseline: {baseline_path.name}")
        print(f"  Q-Weighted: {qweighted_path.name}")
        print(
            f"  Config: {config['constraint']}% constraint, SLD fix: {config['sld_fix_mode']}, "
            f"Prominent: {config['prominent']}"
        )

        try:
            # Extract MAPEs
            baseline_mapes, baseline_per_param = extract_mapes_from_batch(baseline_path)
            qweighted_mapes, qweighted_per_param = extract_mapes_from_batch(
                qweighted_path
            )

            # Get metadata
            baseline_meta = get_batch_metadata(baseline_path)
            qweighted_meta = get_batch_metadata(qweighted_path)

            print(f"  Baseline: {len(baseline_mapes)} MAPEs extracted")
            print(f"  Q-Weighted: {len(qweighted_mapes)} MAPEs extracted")

            if not baseline_mapes or not qweighted_mapes:
                print(f"  Skipping - insufficient data")
                continue

            # Determine output filenames
            constraint = config["constraint"]
            sld_suffix = {"none": "nofix", "backing": "backing", "all": "allSLDfixed"}[
                config["sld_fix_mode"]
            ]

            filename = f"{constraint}_{sld_suffix}.png"

            # Generate overall MAPE plot
            if config["prominent"]:
                overall_output = output_dir / "mape_prom" / filename
            else:
                overall_output = output_dir / "mape" / filename

            plot_model_comparison_histogram(
                baseline_mapes=baseline_mapes,
                comparison_mapes=qweighted_mapes,
                config=config,
                output_dir=overall_output.parent,
                save=True,
                baseline_label="Baseline (Generated Data Only)",
                comparison_label="Q-Weighted (Mixed Data)",
                baseline_meta=baseline_meta,
                comparison_meta=qweighted_meta,
            )

            # Generate per-parameter MAPE plot
            if config["prominent"]:
                param_output = output_dir / "param_mape_prom" / filename
            else:
                param_output = output_dir / "param_mape" / filename

            plot_parameter_comparison_grid(
                baseline_per_param=baseline_per_param,
                comparison_per_param=qweighted_per_param,
                config=config,
                output_dir=param_output.parent,
                save=True,
                baseline_label="Baseline",
                comparison_label="Q-Weighted",
                priors_type=baseline_meta.get("priors_type", "constraint_based"),
            )

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()
            continue

    print(f"\n{'=' * 80}")
    print(f"Comparison complete! Plots saved to: {output_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    # Configuration
    baseline_range = (269, 286)  # Baseline model (generated data only)
    qweighted_range = (233, 250)  # Q-weighted model (mixed data)

    base_dir = Path("batch_inference_results")
    output_dir = Path("model_comparison_plots")

    # Generate all comparison plots
    generate_comparison_plots(baseline_range, qweighted_range, base_dir, output_dir)
