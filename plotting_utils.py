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
                             metrics=['r_squared', 'mse', 'l1_loss']):
    """
    Create a summary plot showing key metrics for all models.
    
    Args:
        results: Dictionary of model results
        experiment_id: Experiment identifier
        output_dir: Directory to save plot
        metrics: List of metrics to plot
        
    Returns:
        Path to saved plot
    """
    successful_results = {k: v for k, v in results.items() if v.get('success', False)}
    
    if not successful_results:
        print("No successful results to plot")
        return None
    
    model_names = list(successful_results.keys())
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
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
