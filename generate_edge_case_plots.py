#!/usr/bin/env python3
"""
Generate individual plots for edge cases (worst predictions) and good predictions (best 3)
for both narrow and broad priors.

This script:
1. Loads the batch summary JSON
2. Identifies edge cases (highest MAPE) and good cases (lowest MAPE)
3. Creates individual plots showing predicted vs original SLD profiles
4. Saves two summary images: one for bad predictions, one for good predictions
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

def load_batch_summary(json_file):
    """Load the batch summary JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)

def extract_mape_scores(all_results):
    """Extract MAPE scores for all experiments and both prior types."""
    mape_scores = {
        'broad': {},  # exp_id: mape
        'narrow': {}  # exp_id: mape
    }
    
    for exp_id, exp_result in all_results.items():
        for priors_type in ['broad', 'narrow']:
            if priors_type not in exp_result.get('priors', {}):
                continue
                
            priors_result = exp_result['priors'][priors_type]
            if not priors_result.get('success', False):
                continue
            
            # Get average MAPE across all models for this experiment and priors type
            exp_mapes = []
            for model_name, model_result in priors_result.get('models_results', {}).items():
                if model_result.get('success', False) and 'parameter_metrics' in model_result:
                    param_metrics = model_result['parameter_metrics']
                    if param_metrics and 'overall' in param_metrics:
                        overall_mape = param_metrics['overall']['mape']
                        exp_mapes.append(overall_mape)
            
            if exp_mapes:
                avg_mape = np.mean(exp_mapes)
                mape_scores[priors_type][exp_id] = avg_mape
    
    return mape_scores

def get_best_model_result(exp_result, priors_type):
    """Get the best model result (lowest MAPE) for a given experiment and priors type."""
    priors_result = exp_result['priors'][priors_type]
    best_model_name = None
    best_mape = float('inf')
    best_result = None
    
    for model_name, model_result in priors_result.get('models_results', {}).items():
        if model_result.get('success', False) and 'parameter_metrics' in model_result:
            param_metrics = model_result['parameter_metrics']
            if param_metrics and 'overall' in param_metrics:
                overall_mape = param_metrics['overall']['mape']
                if overall_mape < best_mape:
                    best_mape = overall_mape
                    best_model_name = model_name
                    best_result = model_result
    
    return best_model_name, best_result, best_mape

def plot_sld_comparison(exp_id, model_result, priors_type, mape, ax):
    """Plot SLD profile comparison for a single experiment."""
    # Check if SLD profile data exists
    if ('sld_profile_x' not in model_result or 
        'sld_profile_predicted' not in model_result or 
        not model_result['sld_profile_x'] or 
        not model_result['sld_profile_predicted']):
        ax.text(0.5, 0.5, f'{exp_id}\nNo SLD data\nMAPE: {mape:.1f}%', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{exp_id} - {priors_type.title()} Priors')
        return
    
    # Convert to numpy arrays for safety
    x_axis = np.array(model_result['sld_profile_x'])
    predicted_sld = np.array(model_result['sld_profile_predicted'])
    
    # Use polished SLD as "actual" if available, otherwise use predicted
    if ('sld_profile_polished' in model_result and 
        model_result['sld_profile_polished'] and 
        len(model_result['sld_profile_polished']) > 0):
        try:
            actual_sld = np.array(model_result['sld_profile_polished'])
            actual_label = 'Polished'
        except (ValueError, TypeError):
            # Fall back to predicted if polished data is malformed
            actual_sld = predicted_sld
            actual_label = 'Predicted (only)'
    else:
        # If no polished data, we'll show predicted only
        actual_sld = predicted_sld
        actual_label = 'Predicted (only)'
    
    # Validate data
    if len(x_axis) == 0 or len(predicted_sld) == 0:
        ax.text(0.5, 0.5, f'{exp_id}\nInvalid SLD data\nMAPE: {mape:.1f}%', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{exp_id} - {priors_type.title()} Priors')
        return
    
    # Plot
    if (len(actual_sld) > 0 and 
        len(actual_sld) == len(predicted_sld) and 
        not np.array_equal(actual_sld, predicted_sld)):
        # Plot both polished and predicted
        ax.plot(x_axis, actual_sld, 'b-', linewidth=2, label=actual_label, alpha=0.8)
        ax.plot(x_axis, predicted_sld, 'r--', linewidth=2, label='Predicted', alpha=0.8)
        ax.legend()
    else:
        # Only one line to show (usually predicted)
        ax.plot(x_axis, predicted_sld, 'r-', linewidth=2, label='Predicted', alpha=0.8)
        ax.legend()
    
    ax.set_xlabel('Depth (Å)')
    ax.set_ylabel('SLD (×10⁻⁶ Å⁻²)')
    ax.set_title(f'{exp_id} - {priors_type.title()}\nMAPE: {mape:.1f}%')
    ax.grid(True, alpha=0.3)

def create_summary_plots(cases_data, plot_type, output_dir, timestamp):
    """Create summary plots for either edge cases or good cases."""
    n_cases = len(cases_data)
    
    if n_cases == 0:
        print(f"No {plot_type} cases to plot")
        return
    
    # Calculate subplot layout
    n_cols = min(3, n_cases)  # Max 3 columns
    n_rows = (n_cases + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_cases == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, (exp_id, priors_type, model_result, mape) in enumerate(cases_data):
        ax = axes[i] if n_cases > 1 else axes[0]
        plot_sld_comparison(exp_id, model_result, priors_type, mape, ax)
    
    # Hide unused subplots
    for i in range(n_cases, len(axes) if n_cases > 1 else 1):
        if n_cases > 1:
            axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    filename = f"{plot_type}_predictions_{timestamp}.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"{plot_type.title()} predictions plot saved to: {filepath}")
    return filepath

def main():
    parser = argparse.ArgumentParser(description='Generate edge case and good prediction plots')
    parser.add_argument('--json-file', type=str, required=True,
                       help='Path to batch summary JSON file')
    parser.add_argument('--output-dir', type=str, 
                       default='batch_inference_results/individual_plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading batch summary from: {args.json_file}")
    batch_data = load_batch_summary(args.json_file)
    all_results = batch_data['all_results']
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract MAPE scores
    print("Extracting MAPE scores...")
    mape_scores = extract_mape_scores(all_results)
    
    # Find edge cases (worst predictions)
    edge_cases = []
    
    for priors_type in ['broad', 'narrow']:
        if not mape_scores[priors_type]:
            continue
            
        # Sort by MAPE (highest first)
        sorted_cases = sorted(mape_scores[priors_type].items(), 
                            key=lambda x: x[1], reverse=True)
        
        # Take top edge cases
        n_edge_cases = min(7, len(sorted_cases))  # Max 7 edge cases per priors type
        
        for exp_id, mape in sorted_cases[:n_edge_cases]:
            if exp_id in all_results:
                best_model_name, best_model_result, _ = get_best_model_result(
                    all_results[exp_id], priors_type)
                if best_model_result:
                    edge_cases.append((exp_id, priors_type, best_model_result, mape))
    
    # Find good cases (best predictions) 
    good_cases = []
    
    for priors_type in ['broad', 'narrow']:
        if not mape_scores[priors_type]:
            continue
            
        # Sort by MAPE (lowest first)
        sorted_cases = sorted(mape_scores[priors_type].items(), 
                            key=lambda x: x[1])
        
        # Take top 3 good cases per priors type
        for exp_id, mape in sorted_cases[:3]:
            if exp_id in all_results:
                best_model_name, best_model_result, _ = get_best_model_result(
                    all_results[exp_id], priors_type)
                if best_model_result:
                    good_cases.append((exp_id, priors_type, best_model_result, mape))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\nFound {len(edge_cases)} edge cases and {len(good_cases)} good cases")
    
    # Create plots
    if edge_cases:
        print(f"\nCreating edge cases plot...")
        create_summary_plots(edge_cases, "edge_case", output_dir, timestamp)
    
    if good_cases:
        print(f"\nCreating good predictions plot...")
        create_summary_plots(good_cases, "good", output_dir, timestamp)
    
    # Print summary
    print(f"\n{'='*60}")
    print("EDGE CASES (Worst Predictions):")
    print("-" * 60)
    for exp_id, priors_type, _, mape in edge_cases:
        print(f"  {exp_id} ({priors_type}): {mape:.1f}% MAPE")
    
    print(f"\nGOOD CASES (Best Predictions):")
    print("-" * 60)
    for exp_id, priors_type, _, mape in good_cases:
        print(f"  {exp_id} ({priors_type}): {mape:.1f}% MAPE")
    
    print(f"\nPlots saved to: {output_dir}")

if __name__ == "__main__":
    main()
