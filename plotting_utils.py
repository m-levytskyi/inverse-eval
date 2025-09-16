#!/usr/bin/env python3
"""
Plotting utilities for reflectometry analysis.

This module contains all plotting functions used in the reflectometry pipeline,
keeping plotting logic separate from the main inference pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path



def plot_simple_comparison(q_exp, curve_exp, sigmas_exp, q_model, 
                          predicted_curve, polished_curve, 
                          predicted_sld_x, predicted_sld_y, polished_sld_y,
                          experiment_name="Analysis", show=True, priors_config=None):
    """
    Simple plot for single model comparison (used by simple_pipeline).
    
    Args:
        q_exp: Experimental Q values
        curve_exp: Experimental reflectivity values
        sigmas_exp: Experimental uncertainties
        q_model: Model Q values
        predicted_curve: Predicted reflectivity curve
        polished_curve: Polished reflectivity curve  
        predicted_sld_x: SLD profile x-axis
        predicted_sld_y: Predicted SLD profile
        polished_sld_y: Polished SLD profile
        experiment_name: Name for plot title
        show: Whether to show the plot
        priors_config: Configuration dictionary containing SLD fixing mode
    """
    # Add SLD fixing mode to the experiment name if available
    display_name = experiment_name
    if priors_config and 'fix_sld_mode' in priors_config:
        fix_sld_mode = priors_config['fix_sld_mode']
        if fix_sld_mode != 'none':
            display_name = f"{experiment_name} (SLD fixed: {fix_sld_mode})"
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot reflectivity curves
    ax[0].set_yscale('log')
    ax[0].set_xlabel('Q [Å⁻¹]', fontsize=14)
    ax[0].set_ylabel('R(Q)', fontsize=14)
    ax[0].tick_params(axis='both', which='major', labelsize=12)
    
    # Experimental data with error bars
    el = ax[0].errorbar(q_exp, curve_exp, yerr=sigmas_exp, xerr=None, 
                        c='b', ecolor='purple', elinewidth=1, 
                        marker='o', linestyle='none', markersize=3, 
                        label='Experimental', zorder=1)
    el.get_children()[1].set_color('purple')
    
    # Predicted curves
    ax[0].plot(q_model, predicted_curve, 
               c='red', lw=2, label='Predicted')
    ax[0].plot(q_model, polished_curve, 
               c='orange', ls='--', lw=2, label='Polished')
    
    ax[0].legend(loc='upper right', fontsize=12)
    ax[0].set_title(f'Reflectivity - {display_name}', fontsize=14)
    ax[0].grid(True, which='both', linestyle='--', alpha=0.3)
    
    # Plot SLD profiles
    ax[1].set_xlabel('Depth [Å]', fontsize=14)
    ax[1].set_ylabel('SLD [×10⁻⁶ Å⁻²]', fontsize=14)
    ax[1].tick_params(axis='both', which='major', labelsize=12)
    
    ax[1].plot(predicted_sld_x, predicted_sld_y, 
               c='red', label='Predicted', linewidth=2)
    ax[1].plot(predicted_sld_x, polished_sld_y, 
               c='orange', ls='--', label='Polished', linewidth=2)
    
    ax[1].legend(loc='best', fontsize=12)
    ax[1].set_title('SLD Profile', fontsize=14)
    ax[1].grid(True, which='both', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig




def plot_batch_edge_case_detection(batch_results, layer_count=1, output_dir=".", save=True):
    """
    Create edge case detection plot showing experiments with high MAPE values.
    
    Args:
        batch_results: Dictionary of batch results from BatchInferencePipeline
        layer_count: Number of layers for plot title
        output_dir: Directory to save plot
        save: Whether to save the plot
        
    Returns:
        Figure path if saved, None otherwise
    """
    # Collect experiment data and SLD fixing mode
    exp_data = {}
    fix_sld_mode = "none"  # Default value
    
    for exp_id, exp_result in batch_results.items():
        if exp_result.get('success', False) and 'param_metrics' in exp_result:
            param_metrics = exp_result['param_metrics']
            # Extract SLD fixing mode from first successful result
            if fix_sld_mode == "none" and 'priors_config' in exp_result:
                fix_sld_mode = exp_result['priors_config'].get('fix_sld_mode', 'none')
            if param_metrics and 'overall_mape' in param_metrics:
                exp_data[exp_id] = param_metrics['overall_mape']
    
    if not exp_data:
        print("No data available for edge case detection")
        return None
    
    # Create edge case detection plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    # Create SLD mode display text
    sld_text = ""
    if fix_sld_mode != "none":
        sld_text = f" (SLD fix: {fix_sld_mode})"
    
    fig.suptitle(f'Edge Case Detection - {len(batch_results)} {layer_count}-Layer Experiments{sld_text}', 
                 fontsize=16, fontweight='bold')
    
    exp_ids = list(exp_data.keys())
    exp_vals = list(exp_data.values())
    exp_indices = range(len(exp_ids))
    
    # Plot all experiments
    ax.plot(exp_indices, exp_vals, 'o-', alpha=0.7, linewidth=1, markersize=4,
           color='darkblue', label='Experiments')
    
    # Calculate threshold for edge cases (mean + 2*std)
    mean_mape = np.mean(exp_vals)
    std_mape = np.std(exp_vals)
    threshold = mean_mape + 2 * std_mape
    
    # Highlight edge cases
    edge_cases = [(i, exp_id, mape) for i, (exp_id, mape) in enumerate(zip(exp_ids, exp_vals)) if mape > threshold]
    
    if edge_cases:
        edge_indices = [i for i, _, _ in edge_cases]
        edge_mapes = [mape for _, _, mape in edge_cases]
        ax.scatter(edge_indices, edge_mapes, color='red', s=80, alpha=0.8, 
                 label=f'Edge Cases (>{threshold:.1f}%)', zorder=5)
        
        # Annotate worst edge cases (top 3)
        worst_cases = sorted(edge_cases, key=lambda x: x[2], reverse=True)[:3]
        for i, exp_id, mape in worst_cases:
            ax.annotate(f'{exp_id}\n{mape:.1f}%', 
                      xy=(i, mape), xytext=(10, 10), 
                      textcoords='offset points', fontsize=8,
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                      arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Add threshold line
    ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
              label=f'Threshold (μ+2σ = {threshold:.1f}%)')
    
    # Add mean line
    ax.axhline(y=mean_mape, color='green', linestyle='--', alpha=0.7, 
              label=f'Mean MAPE ({mean_mape:.1f}%)')
    
    ax.set_xlabel('Experiment Index')
    ax.set_ylabel('MAPE (%)')
    ax.set_title('Edge Case Detection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Total: {len(exp_vals)} experiments\n'
    stats_text += f'Mean MAPE: {mean_mape:.1f}% ± {std_mape:.1f}%\n'
    stats_text += f'Threshold: {threshold:.1f}%\n'
    stats_text += f'Edge Cases: {len(edge_cases)} ({100*len(edge_cases)/len(exp_vals):.1f}%)'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           ha='left', va='top', fontsize=10, 
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save:
        timestamp = datetime.now().strftime("%d%b%Y_%H_%M")
        filename = f"edge_case_detection_{layer_count}layer_{timestamp}.png"
        plot_path = Path(output_dir) / filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Edge case detection plot saved to: {plot_path}")
        
        # Print edge cases summary
        if edge_cases:
            print(f"\nEdge Cases (MAPE > {threshold:.1f}%):")
            for i, exp_id, mape in sorted(edge_cases, key=lambda x: x[2], reverse=True):
                print(f"  {exp_id}: {mape:.1f}% MAPE")
        
        return plot_path
    else:
        plt.show()
        return None




def plot_batch_mape_distribution(batch_results, layer_count=1, output_dir=".", save=True, narrow_priors_deviation=0.99):
    """
    Create MAPE distribution plot showing how experiments are distributed across MAPE ranges.
    
    Args:
        batch_results: Dictionary of batch results from BatchInferencePipeline
        layer_count: Number of layers for plot title
        output_dir: Directory to save plot
        save: Whether to save the plot
        narrow_priors_deviation: Deviation for narrow priors display in title
        
    Returns:
        Figure path if saved, None otherwise
    """
    # Filter successful results
    successful_results = {k: v for k, v in batch_results.items() if v.get('success', False)}
    
    if not successful_results:
        print("No successful results available for MAPE distribution plot")
        return None
    
    # Collect real overall MAPE values with debugging
    mape_data = {'narrow': []}
    fix_sld_mode = "none"  # Default value
    
    print("\nDEBUG - MAPE distribution collection:")
    
    for exp_id, result in successful_results.items():
        if 'param_metrics' in result and result['param_metrics']:
            param_metrics = result['param_metrics']
            
            # Extract SLD fixing mode from first successful result
            if fix_sld_mode == "none" and 'priors_config' in result:
                fix_sld_mode = result['priors_config'].get('fix_sld_mode', 'none')
            
            # Get the real overall MAPE - no artificial calculations
            overall_mape = None
            if 'overall_mape' in param_metrics:
                overall_mape = param_metrics['overall_mape']
                print(f"  {exp_id}: overall_mape = {overall_mape:.2f}%")
            elif 'overall' in param_metrics and isinstance(param_metrics['overall'], dict):
                if 'mape' in param_metrics['overall']:
                    overall_mape = param_metrics['overall']['mape']
                    print(f"  {exp_id}: overall.mape = {overall_mape:.2f}%")
            
            if overall_mape is not None:
                mape_data['narrow'].append(overall_mape)
    
    if not mape_data['narrow']:
        print("No MAPE data available for plotting")
        return None
    
    mapes = mape_data['narrow']
    print(f"\nCollected {len(mapes)} real MAPE values")
    print(f"MAPE range: {np.min(mapes):.1f}% - {np.max(mapes):.1f}%")
    print(f"Mean MAPE: {np.mean(mapes):.1f}% ± {np.std(mapes):.1f}%")
    print(f"Median MAPE: {np.median(mapes):.1f}%")
    
    # Create SLD mode text for title
    sld_mode_text = f" (SLD fix: {fix_sld_mode})" if fix_sld_mode != 'none' else ""
    
    # Create distribution plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle(f'MAPE Distribution - {len(successful_results)} {layer_count}-Layer Experiments{sld_mode_text}\n'
                f'(Narrow Priors ±{int(narrow_priors_deviation * 100)}%)', 
                fontsize=16, fontweight='bold')
    
    # Define MAPE ranges
    mape_ranges = [0, 5, 10, 15, 20, 25, 30, 40, 50, 100]
    range_labels = ['0-5%', '5-10%', '10-15%', '15-20%', '20-25%', '25-30%', '30-40%', '40-50%', '50%+']
    
    # Count experiments in each MAPE range
    counts = []
    for i in range(len(mape_ranges) - 1):
        count = sum(1 for mape in mapes if mape_ranges[i] <= mape < mape_ranges[i+1])
        counts.append(count)
    
    # Add count for 50%+ range
    counts.append(sum(1 for mape in mapes if mape >= 50))
    
    # Create bar chart
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(counts)))
    bars = ax.bar(range(len(counts)), counts, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        if count > 0:
            percentage = (count / len(mapes)) * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
    
    ax.set_xlabel('MAPE Range')
    ax.set_ylabel('Number of Experiments')
    ax.set_xticks(range(len(range_labels)))
    ax.set_xticklabels(range_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    if mapes:
        stats_text = f'Total: {len(mapes)} experiments\n'
        stats_text += f'Mean MAPE: {np.mean(mapes):.1f}%\n'
        stats_text += f'Median MAPE: {np.median(mapes):.1f}%\n'
        stats_text += f'Std Dev: {np.std(mapes):.1f}%\n'
        stats_text += f'Min MAPE: {np.min(mapes):.1f}%\n'
        stats_text += f'Max MAPE: {np.max(mapes):.1f}%'
        
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
               ha='right', va='top', fontsize=10, 
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save:
        # Save plot
        plot_file = Path(output_dir) / f"mape_distribution_{layer_count}layer.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"MAPE distribution plot saved to: {plot_file}")
        return plot_file
    else:
        plt.show()
        return None


def plot_batch_parameter_breakdown(batch_results, layer_count=1, output_dir=".", save=True, narrow_priors_deviation=0.99):
    """
    Create parameter-specific MAPE breakdown plot with detailed debugging.
    
    Args:
        batch_results: Dictionary of batch results from BatchInferencePipeline
        layer_count: Number of layers for plot title
        output_dir: Directory to save plot
        save: Whether to save the plot
        narrow_priors_deviation: Deviation for narrow priors display in title
        
    Returns:
        Figure path if saved, None otherwise
    """
    # Filter successful results
    successful_results = {k: v for k, v in batch_results.items() if v.get('success', False)}
    
    if not successful_results:
        print("No successful results available for parameter breakdown plot")
        return None
    
    # Collect parameter-specific MAPE values from by_type structure
    param_mapes = {
        'thickness': [],
        'roughness': [], 
        'sld': [],
        'overall': []
    }
    
    # Extract SLD fixing mode from results
    fix_sld_mode = "none"  # Default value
    for result in successful_results.values():
        if 'priors_config' in result and fix_sld_mode == "none":
            fix_sld_mode = result['priors_config'].get('fix_sld_mode', 'none')
            break
    
    print("\nDEBUG - Parameter breakdown collection:")
    
    for exp_id, result in successful_results.items():
        if 'param_metrics' in result and result['param_metrics']:
            param_metrics = result['param_metrics']
            
            print(f"\nExperiment {exp_id}:")
            
            # Overall MAPE
            if 'overall_mape' in param_metrics:
                overall_mape = param_metrics['overall_mape']
                param_mapes['overall'].append(overall_mape)
                print(f"  Overall MAPE: {overall_mape:.2f}%")
            elif 'overall' in param_metrics and isinstance(param_metrics['overall'], dict):
                if 'mape' in param_metrics['overall']:
                    overall_mape = param_metrics['overall']['mape']
                    param_mapes['overall'].append(overall_mape)
                    print(f"  Overall MAPE: {overall_mape:.2f}%")
            
            # Individual parameter MAPEs from by_type structure
            if 'by_type' in param_metrics:
                by_type = param_metrics['by_type']
                print(f"  by_type data:")
                for param_type in ['thickness', 'roughness', 'sld']:
                    if param_type in by_type and isinstance(by_type[param_type], dict):
                        if 'mape' in by_type[param_type]:
                            mape_val = by_type[param_type]['mape']
                            param_mapes[param_type].append(mape_val)
                            print(f"    {param_type}: {mape_val:.2f}%")
                        else:
                            print(f"    {param_type}: no MAPE data")
                    else:
                        print(f"    {param_type}: not found in by_type")
    
    # Filter out empty parameter types
    param_mapes = {k: v for k, v in param_mapes.items() if v}
    
    print(f"\nFinal parameter counts:")
    for param_type, values in param_mapes.items():
        print(f"  {param_type}: {len(values)} values")
    
    if not param_mapes:
        print("No parameter-specific MAPE data available for plotting")
        return None
    
    # Create box plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    param_names = list(param_mapes.keys())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95E1D3'][:len(param_names)]
    
    # Separate outliers (>100% MAPE) from regular data for each parameter type
    param_values_regular = []
    outlier_info = []
    
    for name in param_names:
        values = param_mapes[name]
        regular_values = [v for v in values if v <= 100]
        outliers = [v for v in values if v > 100]
        
        param_values_regular.append(regular_values)
        outlier_info.append({
            'count': len(outliers),
            'max_value': max(outliers) if outliers else 0
        })
    
    # Create box plot with fixed scale 0-100%
    box_plot = ax.boxplot(param_values_regular, tick_labels=param_names, patch_artist=True,
                         showfliers=False)  # Don't show regular fliers, we'll handle outliers separately
    
    # Color the boxes
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Set fixed scale from 0-100%
    ax.set_ylim(0, 100)
    ax.set_xlabel('Parameter Type', fontsize=12)
    ax.set_ylabel('MAPE (%)', fontsize=12)
    
    # Create SLD mode text for title
    sld_mode_text = ""
    if fix_sld_mode != "none":
        sld_mode_text = f" (SLD fix: {fix_sld_mode})"
    
    ax.set_title(f'Parameter-Specific MAPE Distribution - {len(successful_results)} {layer_count}-Layer Experiments{sld_mode_text}\n'
                f'(Narrow Priors ±{int(narrow_priors_deviation * 100)}%)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add outlier indicators above the plot
    outlier_y_pos = 105  # Just above the 100% line
    has_outliers = False
    
    for i, (name, info) in enumerate(zip(param_names, outlier_info)):
        if info['count'] > 0:
            has_outliers = True
            outlier_text = f"{info['count']} outliers\n(max: {info['max_value']:.0f}%)"
            ax.text(i + 1, outlier_y_pos, outlier_text, 
                   ha='center', va='bottom', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3),
                   color='red', fontweight='bold')
    
    # Extend y-axis slightly to accommodate outlier indicators
    if has_outliers:
        ax.set_ylim(0, 115)

    # Add statistical annotations (updated for regular values only)
    for i, (name, regular_values, info) in enumerate(zip(param_names, param_values_regular, outlier_info)):
        if regular_values:  # Only add annotation if there are regular values
            median_val = np.median(regular_values)
            mean_val = np.mean(regular_values)
            
            # Position annotations lower if there are outliers
            y_pos = 95 if not has_outliers else 85
            
            stats_text = f'Med: {median_val:.1f}%\nMean: {mean_val:.1f}%'
            if info['count'] > 0:
                stats_text += f'\n{info["count"]} outliers'
                
            ax.text(i + 1, y_pos, stats_text, 
                   ha='center', va='top', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save:
        # Save plot
        plot_file = Path(output_dir) / f"parameter_breakdown_{layer_count}layer.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Parameter breakdown plot saved to: {plot_file}")
        return plot_file
    else:
        plt.show()
        return None


def create_batch_analysis_plots(batch_results, layer_count=1, output_dir=".", save=True):
    """
    Create all batch analysis plots (MAPE distribution, edge case detection, parameter breakdown).
    
    Args:
        batch_results: Dictionary of batch results from BatchInferencePipeline
        layer_count: Number of layers for plot titles
        output_dir: Directory to save plots
        save: Whether to save the plots
        
    Returns:
        Dictionary with paths to saved plots
    """
    print("Creating batch analysis plots...")
    
    plot_paths = {}
    
    # Create MAPE distribution plot
    plot_paths['mape_distribution'] = plot_batch_mape_distribution(
        batch_results, layer_count, output_dir, save
    )
    
    # Create edge case detection plot
    plot_paths['edge_case_detection'] = plot_batch_edge_case_detection(
        batch_results, layer_count, output_dir, save
    )
    
    # Create parameter breakdown plot
    plot_paths['parameter_breakdown'] = plot_batch_parameter_breakdown(
        batch_results, layer_count, output_dir, save
    )
    
    print("Batch analysis plots completed!")
    return plot_paths
