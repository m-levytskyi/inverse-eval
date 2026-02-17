#!/usr/bin/env python3
"""
Re-run plotting utilities on completed batch results.
Regenerates MAPE distribution and parameter breakdown plots.

Supports both generic batch replotting and paper-specific categorized replotting.
"""

import json
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
    if 102 <= batch_id <= 119:
        return "reflectorch"
    elif 269 <= batch_id <= 286:
        return "NF_baseline"
    elif 232 <= batch_id <= 250:
        return "NF_qweighted"
    elif batch_id == 291:
        return "NF_mean_conditioned"
    elif batch_id == 299:
        return "NF_qweighted_exp1_alpha2_beta2"
    elif batch_id == 301:
        return "NF_qweighted_exp2_alpha4_beta4"
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
        "--batch-ids",
        type=int,
        nargs="+",
        help="Specific batch IDs to replot (for paper mode)",
    )

    args = parser.parse_args()

    if args.paper:
        # Paper mode: replot batches into categorized directories
        replot_paper_batches(batch_ids=args.batch_ids)
    elif args.results_dir:
        # Single batch mode
        replot_batch_results(args.results_dir, args.output_dir)
    else:
        parser.error("Either provide results_dir or use --paper mode")


if __name__ == "__main__":
    main()
