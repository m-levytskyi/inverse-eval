#!/usr/bin/env python3
"""
Re-run plotting utilities on completed batch results.
Regenerates MAPE distribution and parameter breakdown plots.
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
    """
    results_dir = Path(results_dir)

    batch_results_file = results_dir / "batch_results.json"
    if not batch_results_file.exists():
        print(f"Error: {batch_results_file} not found!")
        return

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


def main():
    parser = argparse.ArgumentParser(
        description="Re-run plotting utilities on completed batch results"
    )
    parser.add_argument("results_dir", help="Directory containing batch_results.json")
    parser.add_argument("--output-dir", help="Custom output directory for plots")
    args = parser.parse_args()

    replot_batch_results(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
