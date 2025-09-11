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
from collections import defaultdict


def plot_reflectivity_comparison(q_exp, curve_exp, sigmas_exp, results, 
                                experiment_id=None, output_dir=".", save=True):
    """
    Plot comparison of experimental and predicted reflectivity curves.
    
    Args:
        q_exp: Experimental Q values
        curve_exp: Experimental reflectivity values  
        sigmas_exp: Experimental uncertainties
        results: Dictionary of model results
        experiment_id: Experiment identifier for title
        output_dir: Directory to save plot
        save: Whether to save the plot
        
    Returns:
        Figure path if saved, None otherwise
    """
    plt.figure(figsize=(12, 8))
    
    # Plot experimental data with error bars
    plt.errorbar(q_exp, curve_exp, yerr=sigmas_exp, fmt='o', 
                 label='Experimental Data', color='black', markersize=4, 
                 capsize=2, alpha=0.7)
    
    # Plot model predictions
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    for i, (model_name, result) in enumerate(results.items()):
        if result.get('success', False):
            color = colors[i % len(colors)]
            
            # Plot predicted curve
            if 'predicted_curve' in result:
                plt.plot(result['q_model'], result['predicted_curve'], 
                        color=color, linestyle='-', alpha=0.7,
                        label=f'{model_name} (predicted)')
            
            # Plot polished curve
            if 'polished_curve' in result:
                plt.plot(result['q_model'], result['polished_curve'], 
                        color=color, linestyle='--', linewidth=2,
                        label=f'{model_name} (polished)')
    
    plt.yscale('log')
    plt.xlabel('Q (Å⁻¹)', fontsize=14)
    plt.ylabel('Reflectivity', fontsize=14)
    plt.title(f'Reflectivity Comparison - {experiment_id or "Analysis"}', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    if save:
        timestamp = datetime.now().strftime("%d%b%Y_%H_%M")
        filename = f"{experiment_id or 'analysis'}_reflectivity_{timestamp}.png"
        plot_path = Path(output_dir) / filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Reflectivity plot saved to {plot_path}")
        return plot_path
    else:
        plt.show()
        return None


def plot_sld_profiles(results, true_sld_data=None, experiment_id=None, 
                     output_dir=".", save=True):
    """
    Plot SLD profile comparison.
    
    Args:
        results: Dictionary of model results
        true_sld_data: Tuple of (x, y) for true SLD profile, if available
        experiment_id: Experiment identifier for title
        output_dir: Directory to save plot
        save: Whether to save the plot
        
    Returns:
        Figure path if saved, None otherwise
    """
    plt.figure(figsize=(12, 8))
    
    # Plot true SLD profile if available
    if true_sld_data is not None:
        true_x, true_y = true_sld_data
        plt.plot(true_x, true_y, label='True SLD Profile', 
                color='black', linestyle='--', linewidth=3, alpha=0.8)
    
    # Plot model predictions
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    for i, (model_name, result) in enumerate(results.items()):
        if result.get('success', False):
            color = colors[i % len(colors)]
            
            # Plot predicted SLD
            if 'predicted_sld_profile' in result and 'predicted_sld_xaxis' in result:
                plt.plot(result['predicted_sld_xaxis'], result['predicted_sld_profile'], 
                        color=color, linestyle='-', alpha=0.7,
                        label=f'{model_name} (predicted)')
            
            # Plot polished SLD
            if 'sld_profile_polished' in result and 'predicted_sld_xaxis' in result:
                plt.plot(result['predicted_sld_xaxis'], result['sld_profile_polished'], 
                        color=color, linestyle='--', linewidth=2,
                        label=f'{model_name} (polished)')
    
    plt.xlabel('Depth (Å)', fontsize=14)
    plt.ylabel('SLD (×10⁻⁶ Å⁻²)', fontsize=14)
    plt.title(f'SLD Profile Comparison - {experiment_id or "Analysis"}', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    if save:
        timestamp = datetime.now().strftime("%d%b%Y_%H_%M")
        filename = f"{experiment_id or 'analysis'}_sld_{timestamp}.png"
        plot_path = Path(output_dir) / filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"SLD profile plot saved to {plot_path}")
        return plot_path
    else:
        plt.show()
        return None


def plot_simple_comparison(q_exp, curve_exp, sigmas_exp, q_model, 
                          predicted_curve, polished_curve, 
                          predicted_sld_x, predicted_sld_y, polished_sld_y,
                          experiment_name="Analysis", show=True):
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
    """
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
    ax[0].set_title(f'Reflectivity - {experiment_name}', fontsize=14)
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


def save_results_summary_plot(results, experiment_id, output_dir=".", 
                             metrics=None):
    """
    Create a summary plot showing key metrics for all models.
    
    Args:
        results: Dictionary of model results
        experiment_id: Experiment identifier
        output_dir: Directory to save plot
        metrics: List of metrics to plot (default: ['r_squared', 'mse', 'l1_loss'])
        
    Returns:
        Path to saved plot
    """
    if metrics is None:
        metrics = ['r_squared', 'mse', 'l1_loss']
    successful_results = {k: v for k, v in results.items() if v.get('success', False)}
    
    if not successful_results:
        print("No successful results to plot")
        return None
    
    n_metrics = len(metrics)
    
    _, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        values = []
        labels = []
        
        for model_name, result in successful_results.items():
            if 'fit_metrics' in result and metric in result['fit_metrics']:
                values.append(result['fit_metrics'][metric])
                labels.append(model_name)
        
        if values:
            bars = axes[i].bar(range(len(values)), values)
            axes[i].set_xticks(range(len(values)))
            axes[i].set_xticklabels(labels, rotation=45, ha='right')
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}', ha='center', va='bottom')
    
    plt.suptitle(f'Model Performance Comparison - {experiment_id}', fontsize=16)
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%d%b%Y_%H_%M")
    filename = f"{experiment_id}_metrics_summary_{timestamp}.png"
    plot_path = Path(output_dir) / filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Metrics summary plot saved to {plot_path}")
    return plot_path


def plot_batch_mape_distribution(batch_results, layer_count=1, output_dir=".", save=True):
    """
    Create MAPE distribution plot showing how experiments are distributed across MAPE ranges.
    
    Args:
        batch_results: Dictionary of batch results from BatchInferencePipeline
        layer_count: Number of layers for plot title
        output_dir: Directory to save plot
        save: Whether to save the plot
        
    Returns:
        Figure path if saved, None otherwise
    """
    # Collect MAPE values
    mape_values = []
    
    for exp_result in batch_results.values():
        if exp_result.get('success', False) and 'param_metrics' in exp_result:
            param_metrics = exp_result['param_metrics']
            if param_metrics and 'overall_mape' in param_metrics:
                mape_values.append(param_metrics['overall_mape'])
    
    if not mape_values:
        print("No MAPE data available for plotting")
        return None
    
    # Create distribution plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle(f'MAPE Distribution Analysis - {len(batch_results)} {layer_count}-Layer Experiments', 
                fontsize=16, fontweight='bold')
    
    # Define MAPE ranges
    mape_ranges = [0, 5, 10, 15, 20, 25, 30, 40, 50, 100]
    range_labels = ['0-5%', '5-10%', '10-15%', '15-20%', '20-25%', '25-30%', '30-40%', '40-50%', '50%+']
    
    # Count experiments in each MAPE range
    counts = []
    for i in range(len(mape_ranges) - 1):
        count = sum(1 for mape in mape_values if mape_ranges[i] <= mape < mape_ranges[i+1])
        counts.append(count)
    
    # Add count for 50%+ range
    counts.append(sum(1 for mape in mape_values if mape >= 50))
    
    # Create bar chart with proper colormap
    colors = plt.cm.get_cmap('RdYlGn_r')(np.linspace(0.2, 0.8, len(counts)))
    bars = ax.bar(range(len(counts)), counts, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        if count > 0:
            percentage = (count / len(mape_values)) * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
    
    ax.set_xlabel('MAPE Range')
    ax.set_ylabel('Number of Experiments')
    ax.set_title('MAPE Distribution')
    ax.set_xticks(range(len(range_labels)))
    ax.set_xticklabels(range_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = f'Total: {len(mape_values)} experiments\n'
    stats_text += f'Mean MAPE: {np.mean(mape_values):.1f}%\n'
    stats_text += f'Median MAPE: {np.median(mape_values):.1f}%\n'
    stats_text += f'Std Dev: {np.std(mape_values):.1f}%\n'
    stats_text += f'Min MAPE: {np.min(mape_values):.1f}%\n'
    stats_text += f'Max MAPE: {np.max(mape_values):.1f}%'
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
           ha='right', va='top', fontsize=10, 
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save:
        timestamp = datetime.now().strftime("%d%b%Y_%H_%M")
        filename = f"mape_distribution_{layer_count}layer_{timestamp}.png"
        plot_path = Path(output_dir) / filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"MAPE distribution plot saved to: {plot_path}")
        return plot_path
    else:
        plt.show()
        return None


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
    # Collect experiment data
    exp_data = {}
    
    for exp_id, exp_result in batch_results.items():
        if exp_result.get('success', False) and 'param_metrics' in exp_result:
            param_metrics = exp_result['param_metrics']
            if param_metrics and 'overall_mape' in param_metrics:
                exp_data[exp_id] = param_metrics['overall_mape']
    
    if not exp_data:
        print("No data available for edge case detection")
        return None
    
    # Create edge case detection plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    fig.suptitle(f'Edge Case Detection - {len(batch_results)} {layer_count}-Layer Experiments', 
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


def plot_batch_parameter_breakdown(batch_results, layer_count=1, output_dir=".", save=True):
    """
    Create parameter-specific MAPE breakdown plot.
    
    Args:
        batch_results: Dictionary of batch results from BatchInferencePipeline
        layer_count: Number of layers for plot title
        output_dir: Directory to save plot
        save: Whether to save the plot
        
    Returns:
        Figure path if saved, None otherwise
    """
    # Collect parameter-specific MAPE data
    param_data = defaultdict(list)
    
    for exp_result in batch_results.values():
        if exp_result.get('success', False) and 'param_metrics' in exp_result:
            param_metrics = exp_result['param_metrics']
            if param_metrics and 'by_type' in param_metrics:
                by_type = param_metrics['by_type']
                for param_type in ['thickness', 'roughness', 'sld']:
                    mape_key = f'{param_type}_mape'
                    if mape_key in by_type:
                        param_data[param_type].append(by_type[mape_key])
    
    if not param_data:
        print("No parameter-specific MAPE data available for plotting")
        return None
    
    # Create parameter breakdown plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle(f'Parameter-Specific MAPE Breakdown - {len(batch_results)} {layer_count}-Layer Experiments', 
                fontsize=16, fontweight='bold')
    
    param_types = list(param_data.keys())
    param_colors = {'thickness': '#FF6B6B', 'roughness': '#4ECDC4', 'sld': '#45B7D1'}
    
    # Create box plot
    box_data = [param_data[param_type] for param_type in param_types]
    box_colors = [param_colors.get(param_type, 'gray') for param_type in param_types]
    
    bp = ax.boxplot(box_data, labels=[param_type.title() for param_type in param_types],
                    patch_artist=True, showmeans=True, meanline=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('MAPE (%)')
    ax.set_title('Parameter-Specific MAPE Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = ""
    for param_type in param_types:
        values = param_data[param_type]
        if values:
            stats_text += f'{param_type.title()}:\n'
            stats_text += f'  Mean: {np.mean(values):.1f}% ± {np.std(values):.1f}%\n'
            stats_text += f'  Median: {np.median(values):.1f}%\n\n'
    
    ax.text(0.98, 0.98, stats_text.strip(), transform=ax.transAxes, 
           ha='right', va='top', fontsize=10, 
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save:
        timestamp = datetime.now().strftime("%d%b%Y_%H_%M")
        filename = f"parameter_breakdown_{layer_count}layer_{timestamp}.png"
        plot_path = Path(output_dir) / filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Parameter breakdown plot saved to: {plot_path}")
        return plot_path
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
