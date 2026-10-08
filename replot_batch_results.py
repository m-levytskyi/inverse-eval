#!/usr/bin/env python3
"""Regenerate plots from a completed batch-results directory."""

import argparse
import json
import logging
from pathlib import Path

from plotting_utils import create_batch_analysis_plots


logger = logging.getLogger(__name__)


def replot_batch_results(results_dir, output_dir=None):
    """Regenerate plots from a directory containing batch_results.json."""
    results_dir = Path(results_dir)
    batch_results_file = results_dir / "batch_results.json"
    if not batch_results_file.exists():
        raise FileNotFoundError(f"Batch results not found: {batch_results_file}")

    with batch_results_file.open() as file:
        batch_results = json.load(file)

    output_dir = Path(output_dir) if output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_count = (
        2 if "_2_layer_" in results_dir.name or "2layers" in results_dir.name else 1
    )
    successful = {
        key: value for key, value in batch_results.items() if value.get("success")
    }

    logger.info(
        "Processing %s/%s successful experiments (%s layer(s))",
        len(successful),
        len(batch_results),
        layer_count,
    )
    plot_paths = create_batch_analysis_plots(
        successful, layer_count=layer_count, output_dir=output_dir, save=True
    )
    saved = [str(path) for path in plot_paths.values() if path is not None]
    logger.info("Regenerated %s plots in %s", len(saved), output_dir)
    return saved


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Regenerate plots from completed batch results"
    )
    parser.add_argument("results_dir", help="Directory containing batch_results.json")
    parser.add_argument("--output-dir", help="Custom output directory for plots")
    args = parser.parse_args()
    replot_batch_results(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
