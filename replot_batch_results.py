#!/usr/bin/env python3
"""
Script to re-run plotting utilities on completed batch results.
This allows you to regenerate plots with updated plotting functions.
"""

import json
import argparse
from pathlib import Path
from plotting_utils import (
    plot_batch_edge_case_detection
)
from batch_analysis import (
    create_summary_statistics,
    print_summary_statistics, 
    print_mape_distribution,
    detect_edge_cases
)
from plotting_utils import (
    plot_batch_mape_distribution,
    plot_batch_parameter_breakdown
)

def replot_batch_results(results_dir, output_dir=None):
    """
    Re-run plotting utilities on completed batch results.
    
    Args:
        results_dir: Path to directory containing batch_results.json
        output_dir: Optional custom output directory for plots
    """
    results_dir = Path(results_dir)
    
    # Load batch results
    batch_results_file = results_dir / "batch_results.json"
    if not batch_results_file.exists():
        print(f"Error: {batch_results_file} not found!")
        return
    
    print(f"Loading batch results from: {batch_results_file}")
    with open(batch_results_file, 'r') as f:
        batch_results = json.load(f)
    
    # Determine output directory - save directly to batch directory
    if output_dir is None:
        output_dir = results_dir  # Save directly to batch directory, not subdirectory
    else:
        output_dir = Path(output_dir)
    
    print(f"Saving plots to: {output_dir}")
    
    # Extract layer count from directory name
    dir_name = results_dir.name
    if "_1_layer_" in dir_name:
        layer_count = 1
    elif "_2_layer_" in dir_name:
        layer_count = 2
    else:
        layer_count = 1  # Default
    
    # Set default narrow priors deviation
    narrow_priors_deviation = 0.99
    
    num_experiments = len(batch_results)
    print(f"Processing {num_experiments} experiments with {layer_count} layer(s)")
    
    # Filter successful results for batch_analysis functions
    successful_results = {k: v for k, v in batch_results.items() if v.get('success', False)}
    print(f"Found {len(successful_results)} successful experiments")
    
    # Count failed and outlier experiments
    # Outliers are a subset of failed, so we separate them
    outlier_count = sum(1 for v in batch_results.values() if v.get('excluded_as_outlier', False))
    total_failed = sum(1 for v in batch_results.values() if not v.get('success', False))
    failed_count = total_failed - outlier_count  # Failed by other reasons
    
    print(f"Failed experiments (excluding outliers): {failed_count}")
    print(f"Outlier experiments: {outlier_count}")
    print(f"Total failed (failed + outliers): {total_failed}")
    
    # Generate all plots
    plot_files = []
    
    try:
        # 1. MAPE Distribution Plot (from plotting_utils)
        print("\n1. Creating MAPE distribution plot...")
        mape_plot = plot_batch_mape_distribution(
            successful_results, 
            layer_count=layer_count, 
            output_dir=output_dir,
            save=True,
            narrow_priors_deviation=narrow_priors_deviation,
            failed_count=failed_count,
            outlier_count=outlier_count
        )
        if mape_plot:
            plot_files.append(mape_plot)
            print(f"   ✓ Saved: {mape_plot}")
    
        # 2. Parameter Breakdown Plot (from plotting_utils) 
        print("\n2. Creating parameter breakdown plot...")
        param_plot = plot_batch_parameter_breakdown(
            successful_results, 
            layer_count=layer_count, 
            output_dir=output_dir,
            save=True,
            narrow_priors_deviation=narrow_priors_deviation,
            failed_count=failed_count,
            outlier_count=outlier_count
        )
        if param_plot:
            plot_files.append(param_plot)
            print(f"   ✓ Saved: {param_plot}")
    
        # 3. Edge Case Detection Plot (from plotting_utils)
        print("\n3. Creating edge case detection plot...")
        edge_plot = plot_batch_edge_case_detection(
            batch_results, layer_count, output_dir, save=True,
            failed_count=failed_count,
            outlier_count=outlier_count
        )
        if edge_plot:
            plot_files.append(edge_plot)
            print(f"   ✓ Saved: {edge_plot}")
    
        # Note: Removed duplicate parameter breakdown plot call
    
    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\nSuccessfully regenerated {len(plot_files)} plots!")
    print("\nGenerated plots:")
    for plot_file in plot_files:
        print(f"  - {plot_file}")
    
    return plot_files

def main():
    parser = argparse.ArgumentParser(description="Re-run plotting utilities on completed batch results")
    parser.add_argument("results_dir", help="Directory containing batch_results.json")
    parser.add_argument("--output-dir", help="Custom output directory for plots")
    
    args = parser.parse_args()
    
    replot_batch_results(args.results_dir, args.output_dir)

if __name__ == "__main__":
    main()
