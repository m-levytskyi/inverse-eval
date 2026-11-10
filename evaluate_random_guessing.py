#!/usr/bin/env python3
"""
Evaluate model performance against random guessing baseline.

This script loads batch inference results and compares model predictions
against random predictions that are consistent with constraints and prior bounds.
"""

import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from constraints_utils import get_constraint_range
from error_calculation import calculate_parameter_metrics
from plotting_utils import (
    plot_random_guessing_comparison,
    _get_mape_ranges,
    _count_mapes_in_ranges,
)


logger = logging.getLogger(__name__)


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
    bounds = np.asarray(prior_bounds, dtype=float)
    if bounds.shape != (len(param_names), 2) or not np.all(np.isfinite(bounds)):
        raise ValueError(
            "Prior bounds must contain one finite interval per physical parameter"
        )
    if np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("Prior bounds must be ordered lower <= upper")
    return np.random.uniform(bounds[:, 0], bounds[:, 1])


def calculate_mape_for_prediction(
    pred_params: np.ndarray,
    true_params: np.ndarray,
    param_names: List[str],
) -> float:
    """
    Calculate constraint-based MAPE for a single prediction.

    Args:
        pred_params: Predicted parameter values
        true_params: True parameter values
        param_names: List of parameter names
    Returns:
        MAPE value as percentage
    """
    return calculate_parameter_metrics(pred_params, true_params, param_names)[
        "overall"
    ]["constraint_mape"]


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
    logger.info(f"\nProcessing batch: {batch_dir.name}")

    batch_results = load_batch_results(batch_dir)

    # Filter successful experiments
    successful_results = {
        k: v for k, v in batch_results.items() if v.get("success", False)
    }

    if not successful_results:
        logger.info("  No successful experiments in batch")
        return [], []

    logger.info(f"  Found {len(successful_results)} successful experiments")

    model_mapes = []
    random_mapes = []

    for exp_id, result in successful_results.items():
        model_mape = (
            (result.get("param_metrics") or {})
            .get("overall", {})
            .get("constraint_mape")
        )
        if model_mape is None or not np.isfinite(model_mape) or model_mape < 0:
            continue
        layer_key = f"{result.get('layer_count', 1)}_layer"
        truth = (result.get("true_params_dict") or {}).get(layer_key)
        if not truth or not result.get("prior_bounds"):
            continue
        try:
            # NF prediction labels include nuisance parameters; truth defines the scored physical parameters.
            param_names = truth["param_names"]
            random_pred = generate_random_prediction(
                param_names, result["prior_bounds"]
            )
            random_mape = calculate_mape_for_prediction(
                random_pred, truth["params"], param_names
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Skipping unpaired random baseline for %s: %s", exp_id, exc)
            continue
        if np.isfinite(random_mape) and random_mape >= 0:
            model_mapes.append(model_mape)
            random_mapes.append(random_mape)

    logger.info(f"  Collected {len(model_mapes)} model MAPEs")
    logger.info(f"  Collected {len(random_mapes)} random MAPEs")

    return model_mapes, random_mapes


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
        batch_dirs = sorted(
            path
            for path in base_dir.iterdir()
            if path.is_dir()
            and path.name.split("_")[0].isdigit()
            and int(path.name.split("_")[0]) == batch_num
        )

        if not batch_dirs:
            logger.info(f"Batch {batch_num} not found")
            continue

        if len(batch_dirs) > 1:
            raise ValueError(f"Ambiguous batch {batch_num}: {batch_dirs}")

        batch_dir = batch_dirs[0]

        try:
            # Evaluate batch
            model_mapes, random_mapes = evaluate_batch_against_random(batch_dir)

            if not model_mapes or not random_mapes:
                logger.info(f"  Skipping batch {batch_num} - no valid data")
                continue

            # Load batch results to determine the layer count.
            batch_results = load_batch_results(batch_dir)

            use_prominent_features = "PROMINENT" in batch_dir.name
            layer_count = 1

            for result in batch_results.values():
                if result.get("success") and "priors_config" in result:
                    layer_count = result.get("layer_count", 1)
                    break

            plot_random_guessing_comparison(
                model_mapes=model_mapes,
                random_mapes=random_mapes,
                output_dir=output_dir / batch_dir.name,
                layer_count=layer_count,
                use_prominent_features=use_prominent_features,
            )

        except (OSError, ValueError, KeyError, RuntimeError) as e:
            logger.exception("  Error processing batch %s: %s", batch_num, e)
            continue

    logger.info(f"\nEvaluation complete! Plots saved to: {output_dir}")


def generate_synthetic_random_evaluation(
    num_experiments: int = 3000,
    layer_count: int = 1,
    deviation: float = 0.99,
    output_path: Path = None,
):
    """Sample truths globally, then random predictions within truth-conditioned priors."""
    from parameter_discovery import (
        get_parameter_names_for_layer_count,
        get_constraint_based_prior_bounds,
    )

    if num_experiments <= 0:
        raise ValueError("num_experiments must be positive")
    param_names = get_parameter_names_for_layer_count(layer_count)
    constraint_ranges = [get_constraint_range(name) for name in param_names]
    mapes = []
    per_param_mapes = {name: [] for name in param_names}
    for _ in range(num_experiments):
        true_params = np.array(
            [np.random.uniform(lo, hi) for lo, hi in constraint_ranges]
        )
        truth = {
            f"{layer_count}_layer": {
                "params": true_params.tolist(),
                "param_names": param_names,
            }
        }
        prior_bounds = get_constraint_based_prior_bounds(truth, layer_count, deviation)
        prediction = generate_random_prediction(param_names, prior_bounds)
        metrics = calculate_parameter_metrics(prediction, true_params, param_names)
        mapes.append(metrics["overall"]["constraint_mape"])
        for name in param_names:
            per_param_mapes[name].append(
                metrics["by_parameter"][name]["constraint_percentage_error"]
            )

    fig, ax = plt.subplots(figsize=(12, 6))
    ranges, labels = _get_mape_ranges()
    counts = _count_mapes_in_ranges(mapes, ranges)
    bars = ax.bar(labels, counts)
    for bar, count in zip(bars, counts):
        if count:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                count,
                f"{count}\n({count / len(mapes):.1%})",
                ha="center",
                va="bottom",
            )
    ax.set_title(
        f"Random within truth-conditioned priors: {num_experiments} experiments\n"
        f"{layer_count} layers, prior fraction {deviation:.0%}"
    )
    ax.set_xlabel("Constraint MAPE (%)")
    ax.set_ylabel("Experiments")
    ax.tick_params(axis="x", rotation=45)
    statistics = (
        f"Total: {len(mapes)} experiments\nMean: {np.mean(mapes):.2f}%\n"
        f"Median: {np.median(mapes):.2f}%\nStd: {np.std(mapes):.2f}%\n"
        f"Min: {np.min(mapes):.2f}%\nMax: {np.max(mapes):.2f}%"
    )
    ax.text(0.98, 0.98, statistics, transform=ax.transAxes, ha="right", va="top")
    logger.info("Overall constraint MAPE statistics:\n%s", statistics)
    fig.tight_layout()
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
        logger.info("Saved plot: %s", output_path)
    else:
        plt.show()
    plt.close(fig)

    rows = (len(param_names) + 2) // 3
    fig_params, axes = plt.subplots(rows, 3, squeeze=False, figsize=(15, 4 * rows))
    axes = axes.ravel()
    fig_params.suptitle("Random within truth-conditioned priors: physical parameters")
    for index, name in enumerate(param_names):
        ax = axes[index]
        values = per_param_mapes[name]
        ax.hist(values, bins=50, alpha=0.7)
        actual_mean = np.mean(values)
        ax.axvline(
            actual_mean, linestyle="--", label=f"Actual mean: {actual_mean:.2f}%"
        )
        lower, upper = constraint_ranges[index]
        ax.set_title(
            f"{name}\nConstraint: [{lower:g}, {upper:g}] (width: {upper - lower:g})"
        )
        ax.set_xlabel("Constraint MAPE (%)")
        ax.set_ylabel("Experiments")
        parameter_stats = f"Mean: {actual_mean:.2f}%\nMedian: {np.median(values):.2f}%\nStd: {np.std(values):.2f}%"
        ax.text(
            0.98, 0.98, parameter_stats, transform=ax.transAxes, ha="right", va="top"
        )
        ax.legend(loc="upper left")
        logger.info(
            "%s: mean=%.2f%% median=%.2f%% std=%.2f%%",
            name,
            actual_mean,
            np.median(values),
            np.std(values),
        )
    for ax in axes[len(param_names) :]:
        ax.set_visible(False)
    fig_params.tight_layout()
    if output_path:
        per_param_path = output_path.with_name(
            f"{output_path.stem}_per_parameter{output_path.suffix}"
        )
        fig_params.savefig(per_param_path, dpi=300)
        logger.info("Saved per-parameter plot: %s", per_param_path)
    else:
        plt.show()
    plt.close(fig_params)
    logger.info(
        "Mean constraint MAPE: %.2f%%; priors use ground truth, so this is a truth-conditioned baseline.",
        np.mean(mapes),
    )
    return {"mapes": mapes, "by_parameter": per_param_mapes}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import sys

    # Check if synthetic random evaluation is requested
    if len(sys.argv) > 1 and sys.argv[1] == "--synthetic":
        logger.info(
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
                logger.warning("Ignoring invalid batch number %r", arg)

        if batch_numbers:
            if len(batch_numbers) == 1:
                logger.info(
                    f"Evaluating batch {batch_numbers[0]} against random guessing"
                )
            else:
                logger.info(
                    f"Evaluating batches {batch_numbers} against random guessing"
                )
            logger.info("Generating ONE random prediction per experiment\n")

            process_batches(batch_numbers)
        else:
            logger.info("No valid batch numbers provided!")
    else:
        # Default: Process batches 102 to 119 inclusive
        batch_numbers = list(range(102, 120))

        logger.info(
            f"Evaluating batches {batch_numbers[0]} to {batch_numbers[-1]} against random guessing"
        )
        logger.info("Generating ONE random prediction per experiment\n")

        process_batches(batch_numbers)
