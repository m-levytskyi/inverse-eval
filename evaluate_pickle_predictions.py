#!/usr/bin/env python3
"""
Bridge script to evaluate pickle predictions using the reflectorch pipeline.

This script:
1. Loads pickle predictions (5 parameters, excluding sld_fronting)
2. Converts them to reflectorch batch_results format
3. Uses constraint-based prior bounds with proper clipping
4. Calculates constraint-based MAPE using reflectorch error_calculation
5. Generates overall + parameter-specific MAPE histograms

Pickle file: results_exp_1L_fitconstraints0_width0.3_simple.pkl
- 1 layer experiments
- 30% constraint-based priors (width=0.3)
- No SLD fixing (fitconstraints=0, all parameters used)
- Should match batch_results_143.json setup
"""

import pickle
import numpy as np
from pathlib import Path

from error_calculation import calculate_parameter_metrics
from plotting_utils import plot_batch_mape_distribution, plot_batch_parameter_breakdown
from constraints_utils import get_constraint_ranges


def load_pickle_data(pickle_file='results_exp_1L_fitconstraints0_width0.3_simple.pkl'):
    """Load pickle prediction data."""
    print(f"Loading pickle file: {pickle_file}")
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    targets = data[0]  # True values (3169, 6)
    predictions = data[1]  # Predictions (3169, 6)
    indices = data[2]  # Manifest indices
    fitconstraints = data[3]  # 0 = all parameters
    width = data[4]  # 0.3 = 30% constraint-based
    bounds_flags = data[5]  # Out of bounds indicator
    
    print(f"  Loaded {len(indices)} experiments")
    print(f"  fitconstraints: {fitconstraints} (0=all params)")
    print(f"  width: {width} (constraint deviation)")
    print(f"  Out of bounds: {np.sum(bounds_flags)} experiments")
    
    return targets, predictions, indices, bounds_flags, width


def load_manifest(manifest_file='manifest_exp_1L.pkl'):
    """Load manifest to get experiment IDs."""
    print(f"Loading manifest: {manifest_file}")
    with open(manifest_file, 'rb') as f:
        manifest = pickle.load(f)
    
    samples = manifest['samples']
    print(f"  Loaded {len(samples)} manifest entries")
    
    return samples


def get_constraint_based_prior_bounds_for_5params(true_vals_5, width=0.3):
    """
    Calculate constraint-based prior bounds for 5 parameters.
    
    Uses the same logic as reflectorch's parameter_discovery.get_constraint_based_prior_bounds():
    - Calculate width as percentage of constraint span
    - Clip to allowed widths
    - Center around true value
    - Adjust if bounds exceed model constraints
    
    Args:
        true_vals_5: True values for 5 parameters in order:
                     [thickness_1, roughness_fronting, roughness_1, sld_1, sld_backing]
        width: Constraint percentage (0.3 for 30%)
    
    Returns:
        List of (min, max) tuples for prior bounds
    """
    # Parameter mapping to constraint names
    param_types = ['thickness', 'sub_rough', 'amb_rough', 'layer_sld', 'sub_sld']
    
    # Get model constraints
    model_constraints = get_constraint_ranges()
    
    # Allowed widths (from reflectorch parameter_discovery.py)
    allowed_widths = {
        'thickness': (0.01, 1000.0),
        'amb_rough': (0.01, 60.0),
        'sub_rough': (0.01, 60.0),
        'layer_sld': (0.01, 5.0),
        'sub_sld': (0.01, 5.0),
    }
    
    bounds = []
    
    for param_value, param_type in zip(true_vals_5, param_types):
        # Get constraints for this parameter
        model_min, model_max = model_constraints.get(param_type, (-1e6, 1e6))
        width_min, width_max = allowed_widths.get(param_type, (0.01, 1e6))
        
        # Calculate span of model constraints
        constraint_span = model_max - model_min
        
        # Calculate width as percentage of constraint span
        target_width = width * constraint_span
        
        # Clip to allowed widths
        target_width = max(width_min, min(target_width, width_max))
        
        # Center around true value
        half_width = target_width / 2
        min_val = param_value - half_width
        max_val = param_value + half_width
        
        # Adjust if bounds exceed model constraints
        if max_val > model_max:
            # Shift left
            shift = max_val - model_max
            max_val = model_max
            min_val = max(min_val - shift, model_min)
        
        if min_val < model_min:
            # Shift right
            shift = model_min - min_val
            min_val = model_min
            max_val = min(max_val + shift, model_max)
        
        # Final safety check - ensure bounds are within model constraints
        min_val = max(min_val, model_min)
        max_val = min(max_val, model_max)
        
        bounds.append([min_val, max_val])
    
    return bounds


def convert_pickle_to_batch_results(targets, predictions, indices, bounds_flags, 
                                    manifest_samples, width=0.3):
    """
    Convert pickle predictions to reflectorch batch_results format.
    
    Note: We only use 5 parameters (excluding sld_fronting) to match reflectorch.
    """
    print("\nConverting pickle predictions to batch_results format...")
    
    batch_results = {}
    outlier_count = 0
    
    # Reflectorch parameter names for 5 parameters (lowercase as used in error_calculation.py)
    # Order matches batch_results_143.json: thickness, amb_rough, sub_rough, layer_sld, sub_sld
    reflectorch_param_names = ['thickness', 'amb_rough', 'sub_rough', 'layer_sld', 'sub_sld']
    
    # Pickle order: sld_fronting, roughness_fronting, sld_1, thickness_1, roughness_1, sld_backing
    # Map to reflectorch order: thickness_1, roughness_fronting, roughness_1, sld_1, sld_backing
    pickle_to_reflectorch_indices = [3, 1, 4, 2, 5]  # Skip index 0 (sld_fronting)
    
    for pkl_pos in range(len(indices)):
        manifest_idx = indices[pkl_pos]
        
        if manifest_idx >= len(manifest_samples):
            continue
        
        experiment_id = manifest_samples[manifest_idx]['base_id']
        
        # Check if out of bounds - mark as outlier
        is_outlier = bool(bounds_flags[pkl_pos] == 1)
        
        if is_outlier:
            outlier_count += 1
            # Skip outliers - don't add to batch_results
            continue
        
        # Extract 5 parameters (skip sld_fronting)
        true_vals_5 = [targets[pkl_pos][i] for i in pickle_to_reflectorch_indices]
        pred_vals_5 = [predictions[pkl_pos][i] for i in pickle_to_reflectorch_indices]
        
        # Convert SLD values from Å⁻² to 10⁻⁶ Å⁻² (reflectometry standard unit)
        # Indices 3 and 4 in reflectorch order are layer_sld and sub_sld
        true_vals_5[3] *= 1e6  # layer_sld
        true_vals_5[4] *= 1e6  # sub_sld
        pred_vals_5[3] *= 1e6  # layer_sld
        pred_vals_5[4] *= 1e6  # sub_sld
        
        # Calculate constraint-based prior bounds
        prior_bounds_5 = get_constraint_based_prior_bounds_for_5params(true_vals_5, width)
        
        # Calculate parameter metrics using reflectorch's function
        param_metrics = calculate_parameter_metrics(
            pred_params=pred_vals_5,
            true_params=true_vals_5,
            param_names=reflectorch_param_names,
            prior_bounds=prior_bounds_5,
            priors_type="constraint_based"
        )
        
        # Build result entry in reflectorch format
        batch_results[experiment_id] = {
            'experiment_id': experiment_id,
            'success': True,
            'param_metrics': param_metrics,
            'priors_config': {
                'priors_type': 'constraint_based',
                'priors_deviation': width,
                'fix_sld_mode': 'none'
            },
            'layer_count': 1,
            # Add pickle-specific metadata
            '_pickle_metadata': {
                'manifest_index': int(manifest_idx),
                'pickle_position': int(pkl_pos),
                'was_outlier': is_outlier
            }
        }
        
        if (pkl_pos + 1) % 500 == 0:
            print(f"  Processed {pkl_pos + 1}/{len(indices)} experiments...")
    
    print("\nConversion complete:")
    print(f"  Total experiments in pickle: {len(indices)}")
    print(f"  Outliers (excluded): {outlier_count}")
    print(f"  Valid experiments: {len(batch_results)}")
    
    return batch_results, outlier_count


def main():
    """Main execution function."""
    print("="*80)
    print("PICKLE PREDICTIONS EVALUATION")
    print("Using Reflectorch Pipeline with Constraint-Based Priors")
    print("="*80)
    
    # Load data
    targets, predictions, indices, bounds_flags, width = load_pickle_data()
    manifest_samples = load_manifest()
    
    # Convert to batch_results format
    batch_results, outlier_count = convert_pickle_to_batch_results(
        targets, predictions, indices, bounds_flags, manifest_samples, width)
    
    if not batch_results:
        print("ERROR: No valid experiments to evaluate")
        return
    
    # Generate plots using reflectorch plotting utilities
    print("\n" + "="*80)
    print("GENERATING MAPE DISTRIBUTION PLOTS")
    print("="*80)
    
    output_dir = Path(".")
    
    # 1. Overall MAPE histogram
    print("\n1. Overall MAPE Distribution...")
    plot_path_overall = plot_batch_mape_distribution(
        batch_results=batch_results,
        layer_count=1,
        output_dir=str(output_dir),
        save=True,
        narrow_priors_deviation=width,
        use_prominent_features=False,
        failed_count=0,
        outlier_count=outlier_count
    )
    
    if plot_path_overall:
        print(f"   Saved to: {plot_path_overall}")
    
    # 2. Parameter-specific MAPE breakdown
    print("\n2. Parameter-Specific MAPE Breakdown...")
    plot_path_params = plot_batch_parameter_breakdown(
        batch_results=batch_results,
        layer_count=1,
        output_dir=str(output_dir),
        save=True,
        narrow_priors_deviation=width,
        use_prominent_features=False,
        failed_count=0,
        outlier_count=outlier_count
    )
    
    if plot_path_params:
        print(f"   Saved to: {plot_path_params}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    constraint_mapes = []
    param_stats = {
        'thickness': [],
        'roughness': [],
        'sld': []
    }
    
    for result in batch_results.values():
        if 'param_metrics' in result and result['param_metrics']:
            pm = result['param_metrics']
            
            # Overall constraint MAPE
            if 'overall' in pm and 'constraint_mape' in pm['overall']:
                constraint_mapes.append(pm['overall']['constraint_mape'])
            
            # Parameter-specific constraint MAPEs
            if 'by_type' in pm:
                for param_type in ['thickness', 'roughness', 'sld']:
                    if param_type in pm['by_type'] and 'constraint_mape' in pm['by_type'][param_type]:
                        param_stats[param_type].append(pm['by_type'][param_type]['constraint_mape'])
    
    if constraint_mapes:
        print("\nOverall Constraint-Based MAPE Statistics:")
        print(f"  Total experiments: {len(constraint_mapes)}")
        print(f"  Mean: {np.mean(constraint_mapes):.2f}%")
        print(f"  Median: {np.median(constraint_mapes):.2f}%")
        print(f"  Std Dev: {np.std(constraint_mapes):.2f}%")
        print(f"  Min: {np.min(constraint_mapes):.2f}%")
        print(f"  Max: {np.max(constraint_mapes):.2f}%")
        
        # Distribution
        excellent = sum(1 for m in constraint_mapes if m < 5)
        good = sum(1 for m in constraint_mapes if 5 <= m < 10)
        acceptable = sum(1 for m in constraint_mapes if 10 <= m < 20)
        poor = sum(1 for m in constraint_mapes if m >= 20)
        
        total = len(constraint_mapes)
        print("\n  Distribution:")
        print(f"    Excellent (< 5%):     {excellent:4d} ({excellent/total*100:5.1f}%)")
        print(f"    Good (5-10%):         {good:4d} ({good/total*100:5.1f}%)")
        print(f"    Acceptable (10-20%):  {acceptable:4d} ({acceptable/total*100:5.1f}%)")
        print(f"    Poor (≥ 20%):         {poor:4d} ({poor/total*100:5.1f}%)")
    
    # Parameter-specific statistics
    print("\nParameter-Specific Constraint-Based MAPE:")
    for param_type, values in param_stats.items():
        if values:
            print(f"\n  {param_type.upper()}:")
            print(f"    Count: {len(values)}")
            print(f"    Mean: {np.mean(values):.2f}%")
            print(f"    Median: {np.median(values):.2f}%")
            print(f"    Std: {np.std(values):.2f}%")
            print(f"    Range: [{np.min(values):.2f}%, {np.max(values):.2f}%]")
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
