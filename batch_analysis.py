#!/usr/bin/env python3
"""
Batch analysis utilities for reflectometry experiments.

This module contains functions for analyzing batch processing results,
calculating statistics, and detecting edge cases.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def create_summary_statistics(successful_results, layer_count, enable_preprocessing=True, 
                            priors_type="narrow", narrow_priors_deviation=None):
    """Create simplified summary statistics focused on MAPE with detailed debugging."""
    print("\nGenerating summary statistics...")
    
    # Collect MAPE values with detailed debugging
    mape_values = []
    debug_info = []
    
    for exp_id, result in successful_results.items():
        if 'param_metrics' in result and result['param_metrics']:
            param_metrics = result['param_metrics']
            
            # Debug: Show what data structure we have
            print(f"\nDEBUG - Experiment {exp_id}:")
            print(f"  param_metrics keys: {list(param_metrics.keys())}")
            
            # Check for overall MAPE in different formats
            overall_mape = None
            if 'overall_mape' in param_metrics:
                overall_mape = param_metrics['overall_mape']
                print(f"  Found overall_mape: {overall_mape:.2f}%")
            elif 'overall' in param_metrics and isinstance(param_metrics['overall'], dict):
                if 'mape' in param_metrics['overall']:
                    overall_mape = param_metrics['overall']['mape']
                    print(f"  Found overall.mape: {overall_mape:.2f}%")
            
            if overall_mape is not None:
                mape_values.append(overall_mape)
                
                # Debug: Show by_type breakdown if available
                if 'by_type' in param_metrics:
                    print(f"  Parameter breakdown:")
                    by_type = param_metrics['by_type']
                    for param_type, metrics in by_type.items():
                        if isinstance(metrics, dict) and 'mape' in metrics:
                            print(f"    {param_type}: {metrics['mape']:.2f}%")
                
                # Debug: Show individual parameter details if available
                if 'by_parameter' in param_metrics:
                    print(f"  Individual parameters:")
                    for param_name, metrics in param_metrics['by_parameter'].items():
                        if isinstance(metrics, dict):
                            pred = metrics.get('predicted', 'N/A')
                            true = metrics.get('true', 'N/A')
                            rel_err = metrics.get('relative_error_percent', 'N/A')
                            print(f"    {param_name}: pred={pred}, true={true}, error={rel_err}%")
            
            debug_info.append({
                'exp_id': exp_id,
                'overall_mape': overall_mape,
                'param_metrics': param_metrics
            })
    
    print(f"\nCollected {len(mape_values)} MAPE values from {len(successful_results)} successful experiments")
    
    summary = {
        'total_experiments': len(successful_results),
        'layer_count': layer_count,
        'preprocessing_enabled': enable_preprocessing,
        'priors_type': priors_type,
        'narrow_priors_deviation': narrow_priors_deviation if priors_type == "narrow" else None,
        'debug_info': debug_info  # Add debug info to summary
    }
    
    if mape_values:
        summary['parameter_accuracy'] = {
            'overall_mape': {
                'median': float(np.median(mape_values)),
                'mean': float(np.mean(mape_values)),
                'std': float(np.std(mape_values)),
                'min': float(np.min(mape_values)),
                'max': float(np.max(mape_values)),
                'count': len(mape_values)
            }
        }
    
    return summary


def print_summary_statistics(summary):
    """Print simplified summary statistics focusing on MAPE."""
    print("\nBATCH PROCESSING SUMMARY")
    print("="*60)
    print(f"Total successful experiments: {summary['total_experiments']}")
    print(f"Layer count: {summary['layer_count']}")
    print(f"Preprocessing enabled: {summary['preprocessing_enabled']}")
    print(f"Prior bounds: {summary.get('priors_type', 'unknown')}")
    
    if summary.get('narrow_priors_deviation'):
        deviation_percent = summary['narrow_priors_deviation'] * 100
        print(f"Narrow priors deviation: ±{deviation_percent:.1f}%")
    
    if 'parameter_accuracy' in summary and summary['parameter_accuracy']:
        print("\nParameter Accuracy (MAPE):")
        mape_stats = summary['parameter_accuracy']['overall_mape']
        print(f"  Median: {mape_stats['median']:.2f}%")
        print(f"  Mean: {mape_stats['mean']:.2f}% ± {mape_stats['std']:.2f}%")
        print(f"  Range: {mape_stats['min']:.2f}% - {mape_stats['max']:.2f}%")
        print(f"  Experiments: {mape_stats['count']}")
    else:
        print("\nNo MAPE data available")


def print_mape_distribution(successful_results):
    """Print MAPE distribution summary using real overall MAPE values."""
    # Collect real overall MAPE values
    mape_values = []
    
    for result in successful_results.values():
        if 'param_metrics' in result and result['param_metrics']:
            param_metrics = result['param_metrics']
            
            # Get real overall MAPE
            overall_mape = None
            if 'overall_mape' in param_metrics:
                overall_mape = param_metrics['overall_mape']
            elif 'overall' in param_metrics and isinstance(param_metrics['overall'], dict):
                if 'mape' in param_metrics['overall']:
                    overall_mape = param_metrics['overall']['mape']
            
            if overall_mape is not None:
                mape_values.append(overall_mape)
    
    if not mape_values:
        print("\nNo MAPE data available")
        return
    
    print("\nREAL MAPE DISTRIBUTION:")
    print("-" * 35)
    
    # Count experiments in quality ranges
    excellent = sum(1 for mape in mape_values if mape < 5)
    good = sum(1 for mape in mape_values if 5 <= mape < 10)
    acceptable = sum(1 for mape in mape_values if 10 <= mape < 20)
    poor = sum(1 for mape in mape_values if mape >= 20)
    
    total = len(mape_values)
    print(f"Excellent (< 5%): {excellent} ({100*excellent/total:.1f}%)")
    print(f"Good (5-10%): {good} ({100*good/total:.1f}%)")
    print(f"Acceptable (10-20%): {acceptable} ({100*acceptable/total:.1f}%)")
    print(f"Poor (≥ 20%): {poor} ({100*poor/total:.1f}%)")
    
    print(f"\nStatistics:")
    print(f"Mean: {np.mean(mape_values):.1f}% ± {np.std(mape_values):.1f}%")
    print(f"Median: {np.median(mape_values):.1f}%")
    print(f"Range: {np.min(mape_values):.1f}% - {np.max(mape_values):.1f}%")


def create_mape_distribution_plot(successful_results, layer_count, plots_dir, narrow_priors_deviation=0.99):
    """Create MAPE distribution plot showing real overall MAPE values with debugging."""
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
    sld_mode_text = f" (SLD: {fix_sld_mode})" if fix_sld_mode != 'none' else ""
    
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
    
    # Save plot
    plot_file = plots_dir / f"mape_distribution_{layer_count}layer.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"MAPE distribution plot saved to: {plot_file}")
    
    return plot_file


def create_parameter_breakdown_plot(successful_results, layer_count, plots_dir, narrow_priors_deviation=0.99):
    """Create parameter-specific MAPE breakdown plot with proper debugging."""
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
    param_values = [param_mapes[name] for name in param_names]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95E1D3'][:len(param_names)]
    
    # Create box plot
    box_plot = ax.boxplot(param_values, tick_labels=param_names, patch_artist=True,
                         showfliers=True, flierprops=dict(marker='o', markerfacecolor='red', 
                         markersize=5, alpha=0.5))
    
    # Color the boxes
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Parameter Type', fontsize=12)
    ax.set_ylabel('MAPE (%)', fontsize=12)
    # Create SLD mode text for title
    sld_mode_text = ""
    if fix_sld_mode != "none":
        sld_mode_text = f" (SLD: {fix_sld_mode})"
    
    ax.set_title(f'Parameter-Specific MAPE Distribution - {len(successful_results)} {layer_count}-Layer Experiments{sld_mode_text}\n'
                f'(Narrow Priors ±{int(narrow_priors_deviation * 100)}%)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistical annotations
    for i, (name, values) in enumerate(zip(param_names, param_values)):
        if values:  # Only add annotation if there are values
            median_val = np.median(values)
            mean_val = np.mean(values)
            ax.text(i + 1, ax.get_ylim()[1] * 0.95, 
                   f'Med: {median_val:.1f}%\nMean: {mean_val:.1f}%', 
                   ha='center', va='top', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    plot_file = plots_dir / f"parameter_breakdown_{layer_count}layer.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Parameter breakdown plot saved to: {plot_file}")
    
    return plot_file


def detect_edge_cases(successful_results):
    """Detect edge cases with poor performance using real MAPE values."""
    edge_cases = []
    
    print("\nDEBUG - Edge case detection:")
    
    for exp_name, result in successful_results.items():
        if 'param_metrics' not in result or not result['param_metrics']:
            continue
            
        param_metrics = result['param_metrics']
        
        # Get real overall MAPE
        overall_mape = None
        if 'overall_mape' in param_metrics:
            overall_mape = param_metrics['overall_mape']
        elif 'overall' in param_metrics and isinstance(param_metrics['overall'], dict):
            if 'mape' in param_metrics['overall']:
                overall_mape = param_metrics['overall']['mape']
        
        if overall_mape is not None:
            print(f"  {exp_name}: {overall_mape:.1f}% MAPE")
            
            # Flag as edge case if MAPE > 50%
            if overall_mape > 50:
                # Get individual parameter details if available
                thickness_mape = None
                roughness_mape = None
                sld_mape = None
                
                if 'by_type' in param_metrics:
                    by_type = param_metrics['by_type']
                    thickness_mape = by_type.get('thickness', {}).get('mape', 0)
                    roughness_mape = by_type.get('roughness', {}).get('mape', 0)
                    sld_mape = by_type.get('sld', {}).get('mape', 0)
                
                edge_cases.append({
                    'experiment': exp_name,
                    'overall_mape': overall_mape,
                    'thickness_mape': thickness_mape,
                    'roughness_mape': roughness_mape,
                    'sld_mape': sld_mape
                })
    
    # Sort by worst performance
    edge_cases.sort(key=lambda x: x['overall_mape'], reverse=True)
    
    if edge_cases:
        print(f"\n🚨 Edge Cases Detected ({len(edge_cases)} experiments with MAPE > 50%):")
        print("-" * 80)
        for i, case in enumerate(edge_cases[:5], 1):  # Show top 5 worst
            print(f"{i}. {case['experiment']}")
            print(f"   Overall MAPE: {case['overall_mape']:.1f}%")
            if case['thickness_mape'] is not None:
                print(f"   Thickness: {case['thickness_mape']:.1f}%")
            if case['roughness_mape'] is not None:
                print(f"   Roughness: {case['roughness_mape']:.1f}%")
            if case['sld_mape'] is not None:
                print(f"   SLD: {case['sld_mape']:.1f}%")
            print()
    else:
        print("\n✅ No edge cases detected (all experiments < 50% MAPE)")
    
    return edge_cases


if __name__ == "__main__":
    print("Batch analysis module loaded successfully.")
    print("Available functions:")
    print("  - create_summary_statistics()")
    print("  - print_summary_statistics()")
    print("  - print_mape_distribution()")
    print("  - create_mape_distribution_plot()")
    print("  - create_parameter_breakdown_plot()")
    print("  - detect_edge_cases()")
