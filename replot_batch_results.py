#!/usr/bin/env python3
"""
Re-run plotting utilities on completed batch results.
Regenerates MAPE distribution and parameter breakdown plots.

Supports both generic batch replotting and paper-specific categorized replotting.
"""

import json
import sys
import pickle
import argparse
from pathlib import Path
from plotting_utils import create_batch_analysis_plots


def replot_batch_results(results_dir, output_dir=None):
    """
    Re-run plotting utilities on completed batch results.

    Args:
        results_dir: Path to directory containing batch_results.json
        output_dir: Optional custom output directory for plots

    Returns:
        List of saved plot paths
    """
    results_dir = Path(results_dir)

    batch_results_file = results_dir / "batch_results.json"
    if not batch_results_file.exists():
        print(f"Error: {batch_results_file} not found!")
        return []

    print(f"Loading batch results from: {batch_results_file}")
    with open(batch_results_file, "r") as f:
        batch_results = json.load(f)

    if output_dir is None:
        output_dir = results_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Extract layer count from directory name
    dir_name = results_dir.name
    if "_2_layer_" in dir_name:
        layer_count = 2
    else:
        layer_count = 1

    successful = {k: v for k, v in batch_results.items() if v.get("success", False)}
    print(f"Processing {len(successful)}/{len(batch_results)} successful experiments ({layer_count} layer(s))")
    print(f"Saving plots to: {output_dir}")

    try:
        plot_paths = create_batch_analysis_plots(
            successful,
            layer_count=layer_count,
            output_dir=output_dir,
            save=True,
        )
        saved = [str(p) for p in plot_paths.values() if p is not None]
        print(f"\nRegenerated {len(saved)} plots:")
        for path in saved:
            print(f"  - {path}")
        return saved

    except Exception as e:
        print(f"Error creating plots: {e}")
        import traceback
        traceback.print_exc()
        return []


def find_batch_directory(batch_id, base_dir="batch_inference_results"):
    """
    Find batch directory by batch ID.

    Args:
        batch_id: Batch ID number
        base_dir: Base directory to search in

    Returns:
        Path to batch directory or None if not found
    """
    base_path = Path(base_dir)
    pattern = f"{batch_id:03d}_*"

    matches = list(base_path.glob(pattern))
    if matches:
        return matches[0]
    return None


def get_batch_category(batch_id):
    """
    Determine which category a batch belongs to for paper organization.

    Args:
        batch_id: Batch ID number

    Returns:
        Category name string
    """
    if 302 <= batch_id <= 319:
        return "reflectorch"
    elif 269 <= batch_id <= 286:
        return "NF_baseline"
    elif 233 <= batch_id <= 250:
        return "NF_qweighted"
    elif batch_id == 291:
        return "NF_mean_conditioned"
    elif batch_id == 299:
        return "NF_qweighted_exp1_alpha2_beta2"
    elif batch_id == 301:
        return "NF_qweighted_exp2_alpha4_beta4"
    elif batch_id == 320:
        return "NF_baseline_with_stats"
    else:
        return "other"


def replot_batch_paper(batch_id, base_output_dir="paper_batches"):
    """
    Replot batch results into categorized output directories for paper.

    Args:
        batch_id: Batch ID number
        base_output_dir: Base directory for categorized outputs

    Returns:
        List of saved plot paths
    """
    batch_dir = find_batch_directory(batch_id)

    if not batch_dir:
        print(f"Batch {batch_id} not found")
        return []

    print(f"\nProcessing batch {batch_id}: {batch_dir.name}")

    category = get_batch_category(batch_id)
    output_dir = Path(base_output_dir) / category / batch_dir.name

    return replot_batch_results(batch_dir, output_dir=output_dir)


def replot_paper_batches(batch_ids=None):
    """
    Replot specified batches into categorized output directories.

    Args:
        batch_ids: List of batch IDs to replot. If None, uses default set.

    Returns:
        Number of successfully processed batches
    """
    if batch_ids is None:
        # Default batch ranges for paper
        batch_ids = []
        batch_ids.extend(range(269, 287))  # NF baseline: 269-286
        batch_ids.extend(range(232, 251))  # NF qweighted: 232-250
        batch_ids.append(291)  # NF mean conditioned
        batch_ids.append(299)  # NF qweighted exp1, alpha=beta=2
        batch_ids.extend(range(102, 120))  # Reflectorch: 102-119

    batch_ids = sorted(set(batch_ids))
    print(f"Replotting {len(batch_ids)} batches")
    print(f"Batch IDs: {batch_ids}")

    success_count = 0
    for batch_id in batch_ids:
        try:
            result = replot_batch_paper(batch_id)
            if result:
                success_count += 1
        except Exception as e:
            print(f"Error processing batch {batch_id}: {e}")

    print(f"\nCompleted: {success_count}/{len(batch_ids)} batches")
    return success_count


def replot_anaklasis(pickle_path=None, output_dir=None, layer_count=1):
    """
    Replot batch results from anaklasis evaluation pickle files.

    Loads pickle predictions, converts to batch_results format using
    anaklasis_eval/evaluate_pickle_predictions.py, then generates
    MAPE distribution and parameter breakdown plots.

    Args:
        pickle_path: Path to pickle results file. Defaults to
                     anaklasis_eval/results_exp_1L_fitconstraints0_width0.3_simple.pkl
        output_dir: Output directory (defaults to paper_batches/anaklasis/)
        layer_count: Number of layers (default 1)

    Returns:
        List of saved plot paths
    """
    anaklasis_dir = Path("anaklasis_eval")

    if pickle_path is None:
        pickle_path = anaklasis_dir / f"results_exp_{layer_count}L_fitconstraints0_width0.3_simple.pkl"
    else:
        pickle_path = Path(pickle_path)

    if not pickle_path.exists():
        print(f"Error: {pickle_path} not found!")
        return []

    manifest_path = anaklasis_dir / f"manifest_exp_{layer_count}L.pkl"
    if not manifest_path.exists():
        print(f"Error: {manifest_path} not found!")
        return []

    # Import conversion functions from anaklasis_eval
    sys.path.insert(0, str(anaklasis_dir.resolve()))
    try:
        from evaluate_pickle_predictions import (
            load_pickle_data,
            load_manifest,
            convert_pickle_to_batch_results,
        )
    finally:
        sys.path.pop(0)

    # Load pickle data and manifest
    targets, predictions, indices, bounds_flags, width = load_pickle_data(
        str(pickle_path))

    # Validate: convert_pickle_to_batch_results only supports fitconstraints=0 (6 params)
    if targets.shape[1] != 6:
        print(f"Error: Pickle has {targets.shape[1]} parameters per experiment. "
              f"Only fitconstraints=0 (6 parameters) is supported.")
        return []

    manifest_samples = load_manifest(str(manifest_path))

    # Convert to batch_results format
    batch_results, outlier_count = convert_pickle_to_batch_results(
        targets, predictions, indices, bounds_flags, manifest_samples, width)

    if not batch_results:
        print("Error: No valid experiments after conversion")
        return []

    if output_dir is None:
        # Derive subdirectory from pickle filename
        stem = pickle_path.stem  # e.g. results_exp_1L_fitconstraints0_width0.3_simple
        output_dir = Path("paper_batches") / "anaklasis" / stem
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    successful = {k: v for k, v in batch_results.items() if v.get("success", False)}
    print(f"Processing {len(successful)} experiments "
          f"({layer_count} layer(s), {outlier_count} outliers excluded)")
    print(f"Saving plots to: {output_dir}")

    try:
        plot_paths = create_batch_analysis_plots(
            successful,
            layer_count=layer_count,
            output_dir=output_dir,
            save=True,
        )
        saved = [str(p) for p in plot_paths.values() if p is not None]
        print(f"\nRegenerated {len(saved)} plots:")
        for path in saved:
            print(f"  - {path}")
        return saved

    except Exception as e:
        print(f"Error creating plots: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Re-run plotting utilities on completed batch results"
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        help="Directory containing batch_results.json (for single batch mode)",
    )
    parser.add_argument(
        "--output-dir",
        help="Custom output directory for plots",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Replot paper batches into categorized directories",
    )
    parser.add_argument(
        "--anaklasis",
        nargs="?",
        const="default",
        default=None,
        metavar="PICKLE_PATH",
        help="Replot anaklasis results from pickle file "
             "(default: anaklasis_eval/results_exp_1L_fitconstraints0_width0.3_simple.pkl)",
    )
    parser.add_argument(
        "--batch-ids",
        type=int,
        nargs="+",
        help="Specific batch IDs to replot (for paper mode)",
    )

    args = parser.parse_args()

    if args.anaklasis is not None:
        pkl = None if args.anaklasis == "default" else args.anaklasis
        replot_anaklasis(pickle_path=pkl, output_dir=args.output_dir)
    elif args.paper:
        # Paper mode: replot batches into categorized directories
        replot_paper_batches(batch_ids=args.batch_ids)
    elif args.results_dir:
        # Single batch mode
        replot_batch_results(args.results_dir, args.output_dir)
    else:
        parser.error("Either provide results_dir, use --paper, or use --anaklasis")


if __name__ == "__main__":
    main()
