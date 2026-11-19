#!/usr/bin/env python3
"""
Compare two model versions by plotting MAPE distributions.

This script compares NF model performance between:
- Batches 233-250: Q-weighted model trained on mixed (generated + experimental) data
- Batches 269-286: Baseline model trained on generated data only

The baseline model appears as grey background bars, q-weighted as colored foreground bars.
"""

import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from plotting_utils import (
    plot_model_comparison_histogram,
    plot_parameter_comparison_grid,
)


logger = logging.getLogger(__name__)


def load_batch_results(batch_dir: Path) -> Dict:
    """Load current per-experiment records, including historical batch files."""
    with (batch_dir / "batch_results.json").open() as f:
        return json.load(f)


def parse_batch_config(batch_name: str) -> Dict:
    """
    Parse batch configuration from directory name.

    Args:
        batch_name: Batch directory name (e.g., "233_951exps_1layers_5constraint_10january2026_18_23")

    Returns:
        Dictionary with parsed configuration
    """
    match = re.match(
        r"^(\d+)_(?:custom_)?(\d+|None)exps_(\d+)layers_(\d+)(constraint|priors)(?:_|$)",
        batch_name,
        re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"Cannot parse batch setup: {batch_name}")
    batch_num, count, layers, deviation, prior_kind = match.groups()
    config = {
        "batch_num": int(batch_num),
        "num_exps": None if count.lower() == "none" else int(count),
        "layer_count": int(layers),
        "constraint": int(deviation),
        "deviation": int(deviation) / 100.0,
        "priors_type": "constraint_based"
        if prior_kind.lower() == "constraint"
        else "narrow",
        "prominent": "PROMINENT" in batch_name.upper(),
        "sld_fix_mode": "none",
        "use_theoretical": False,
    }

    # Parse SLD fix mode
    if "allSLDfix" in batch_name or "allsldfix" in batch_name.lower():
        config["sld_fix_mode"] = "all"
    elif "backSLDfix" in batch_name or "backsldfix" in batch_name.lower():
        config["sld_fix_mode"] = "backing"

    return config


def load_batch_config(batch_dir: Path) -> Dict:
    """Prefer persisted settings; older runs retain directory-name lookup."""
    metadata_path = batch_dir / "run_metadata.json"
    if not metadata_path.exists():
        return parse_batch_config(batch_dir.name)
    config = json.loads(metadata_path.read_text())["config"]
    return {
        "layer_count": config["layer_count"],
        "constraint": config["priors_deviation"] * 100,
        "deviation": config["priors_deviation"],
        "priors_type": config["priors_type"],
        "sld_fix_mode": config["fix_sld_mode"],
        "prominent": config["use_prominent_features"],
        "use_theoretical": config.get("use_theoretical", False),
    }


def _scored_records(batch_results):
    """Keep successful records with a usable production score."""
    records = {}
    for exp_id, result in batch_results.items():
        score = (
            (result.get("param_metrics") or {})
            .get("overall", {})
            .get("constraint_mape")
        )
        if (
            result.get("success")
            and score is not None
            and math.isfinite(score)
            and score >= 0
        ):
            records[exp_id] = result
    return records


def extract_mapes_from_batch(
    batch_dir: Path, experiment_ids=None
) -> Tuple[List[float], Dict]:
    """Extract current scores, optionally in an explicit paired ID order."""
    records = _scored_records(load_batch_results(batch_dir))
    ids = sorted(records) if experiment_ids is None else experiment_ids
    mapes = []
    per_param_mapes = defaultdict(list)
    for exp_id in ids:
        metrics = records[exp_id]["param_metrics"]
        mapes.append(metrics["overall"]["constraint_mape"])
        for name, data in metrics.get("by_parameter", {}).items():
            score = data.get("constraint_percentage_error")
            if score is not None and math.isfinite(score) and score >= 0:
                per_param_mapes[name].append(score)
    return mapes, dict(per_param_mapes)


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
    keys = (
        "layer_count",
        "priors_type",
        "deviation",
        "sld_fix_mode",
        "prominent",
        "use_theoretical",
    )
    groups = []
    for batches in (baseline_batches, qweighted_batches):
        setups = {}
        for path in batches:
            config = load_batch_config(path)
            setup = tuple(config[key] for key in keys)
            if setup in setups:
                raise ValueError(
                    f"Ambiguous duplicate setup: {setups[setup][0]} and {path}"
                )
            setups[setup] = (path, config)
        groups.append(setups)
    return [
        (path, groups[1][setup][0], config)
        for setup, (path, config) in groups[0].items()
        if setup in groups[1]
    ]


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
    logger.info(f"\n{'=' * 80}")
    logger.info("MODEL COMPARISON: Q-Weighted vs Baseline")
    logger.info(f"{'=' * 80}")
    logger.info(
        f"Baseline batches: {baseline_range[0]}-{baseline_range[1]} (Generated data only)"
    )
    logger.info(
        f"Q-Weighted batches: {qweighted_range[0]}-{qweighted_range[1]} (Mixed data)"
    )
    logger.info("")

    # Find batch directories
    all_batches = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.split("_")[0].isdigit()]
    )

    baseline_batches = []
    qweighted_batches = []

    selected_numbers = {}
    for batch_dir in all_batches:
        batch_num = int(batch_dir.name.split("_")[0])
        selected = (
            baseline_range[0] <= batch_num <= baseline_range[1]
            or qweighted_range[0] <= batch_num <= qweighted_range[1]
        )
        if selected and batch_num in selected_numbers:
            raise ValueError(
                f"Ambiguous batch {batch_num}: {selected_numbers[batch_num]} and {batch_dir}"
            )
        if selected:
            selected_numbers[batch_num] = batch_dir
        if baseline_range[0] <= batch_num <= baseline_range[1]:
            baseline_batches.append(batch_dir)
        elif qweighted_range[0] <= batch_num <= qweighted_range[1]:
            qweighted_batches.append(batch_dir)

    logger.info(f"Found {len(baseline_batches)} baseline batches")
    logger.info(f"Found {len(qweighted_batches)} q-weighted batches")
    logger.info("")

    # Find matching pairs
    matches = find_matching_batches(baseline_batches, qweighted_batches)
    logger.info(f"Found {len(matches)} matching batch pairs")
    logger.info("")

    saved_paths = []
    # Generate plots for each matching pair
    for baseline_path, qweighted_path, config in matches:
        logger.info("\nProcessing pair:")
        logger.info(f"  Baseline: {baseline_path.name}")
        logger.info(f"  Q-Weighted: {qweighted_path.name}")
        logger.info(
            f"  Config: {config['constraint']}% constraint, SLD fix: {config['sld_fix_mode']}, "
            f"Prominent: {config['prominent']}"
        )

        try:
            baseline = _scored_records(load_batch_results(baseline_path))
            comparison = _scored_records(load_batch_results(qweighted_path))
            paired_ids = sorted(baseline.keys() & comparison.keys())
            logger.info(
                "  Paired %s experiments; unmatched baseline=%s, comparison=%s",
                len(paired_ids),
                len(baseline) - len(paired_ids),
                len(comparison) - len(paired_ids),
            )
            if not paired_ids:
                logger.info("  Skipping - insufficient data")
                continue
            baseline_mapes, qweighted_mapes = [
                [
                    records[exp_id]["param_metrics"]["overall"]["constraint_mape"]
                    for exp_id in paired_ids
                ]
                for records in (baseline, comparison)
            ]
            baseline_per_param, qweighted_per_param = (
                defaultdict(list),
                defaultdict(list),
            )
            for exp_id in paired_ids:
                left = baseline[exp_id]["param_metrics"].get("by_parameter", {})
                right = comparison[exp_id]["param_metrics"].get("by_parameter", {})
                for name in left.keys() & right.keys():
                    scores = [
                        side[name].get("constraint_percentage_error")
                        for side in (left, right)
                    ]
                    if all(
                        value is not None and math.isfinite(value) and value >= 0
                        for value in scores
                    ):
                        baseline_per_param[name].append(scores[0])
                        qweighted_per_param[name].append(scores[1])

            pair_output = (
                output_dir
                / f"{baseline_path.name.split('_')[0]}_vs_{qweighted_path.name.split('_')[0]}"
            )
            pair_output.mkdir(parents=True, exist_ok=True)
            (pair_output / "comparison_manifest.json").write_text(
                json.dumps(
                    {
                        "baseline": str(baseline_path),
                        "comparison": str(qweighted_path),
                        "experiment_ids": paired_ids,
                        "unmatched_baseline": sorted(
                            baseline.keys() - comparison.keys()
                        ),
                        "unmatched_comparison": sorted(
                            comparison.keys() - baseline.keys()
                        ),
                    },
                    indent=2,
                )
            )
            saved_paths.append(
                plot_model_comparison_histogram(
                    baseline_mapes=baseline_mapes,
                    comparison_mapes=qweighted_mapes,
                    config=config,
                    output_dir=pair_output,
                    save=True,
                    baseline_label="Baseline (Generated Data Only)",
                    comparison_label="Q-Weighted (Mixed Data)",
                )
            )
            saved_paths.append(
                plot_parameter_comparison_grid(
                    baseline_per_param=baseline_per_param,
                    comparison_per_param=qweighted_per_param,
                    config=config,
                    output_dir=pair_output,
                    save=True,
                    baseline_label="Baseline",
                    comparison_label="Q-Weighted",
                )
            )

        except (OSError, ValueError, KeyError, RuntimeError) as e:
            logger.exception("  Error processing pair: %s", e)
            continue

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Comparison complete! Plots saved to: {output_dir}")
    logger.info(f"{'=' * 80}")
    return [path for path in saved_paths if path is not None]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Configuration
    baseline_range = (269, 286)  # Baseline model (generated data only)
    qweighted_range = (233, 250)  # Q-weighted model (mixed data)

    base_dir = Path("batch_inference_results")
    output_dir = Path("model_comparison_plots")

    # Generate all comparison plots
    generate_comparison_plots(baseline_range, qweighted_range, base_dir, output_dir)
