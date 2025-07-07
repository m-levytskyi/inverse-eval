#!/usr/bin/env python3
"""
Batch Inference Pipeline for ReflecTorch Models

This script runs the inference pipeline on multiple experiments
with appropriate models for each layer configuration.

Usage:
    python batch_inference_pipeline.py [--num-experiments 25] [--layer-count 2]
"""

import argparse
import json
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import subprocess

# Import the existing inference pipeline
from inference_pipeline import InferencePipeline

class BatchInferencePipeline:
    """Batch pipeline using the new parameterized inference pipeline."""
    
    def __init__(self, num_experiments=25, layer_count=2, data_directory="data"):
        self.num_experiments = num_experiments
        self.layer_count = layer_count
        self.data_directory = Path(data_directory)
        self.maria_dataset_path = Path(data_directory) / "MARIA_VIPR_dataset"
        self.output_dir = Path("batch_inference_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Model sets for different layer counts
        self.model_sets = {
            1: [
                "b_mc_point_neutron_conv_standard_L1_comp",
                "b_mc_point_neutron_conv_standard_L1_InputQDq",
                "b_mc_point_xray_conv_standard_L2" # this xray model is designed for 2 layers. however, there are no xray models for 1 layer.
            ],
            2: [
                "b_mc_point_neutron_conv_standard_L2_comp",
                "b_mc_point_neutron_conv_standard_L2_InputQDq", 
                "b_mc_point_xray_conv_standard_L2"
            ]
        }
        
        # Get models for current layer count
        if self.layer_count not in self.model_sets:
            raise ValueError(f"Layer count {self.layer_count} not supported. Available: {list(self.model_sets.keys())}")
            
        self.models = self.model_sets[self.layer_count]
        self.batch_results = {}

    def discover_experiments(self):
        """Find experiments using terminal commands to avoid reading large directories."""
        layer_dir = self.maria_dataset_path / str(self.layer_count)
        
        if not layer_dir.exists():
            print(f"Warning: MARIA dataset directory not found: {layer_dir}")
            return []
        
        print(f"Searching for {self.layer_count}-layer experiments in: {layer_dir}")
        
        # Use find command to locate experiment files
        try:
            result = subprocess.run([
                'find', str(layer_dir), 
                '-name', '*_experimental_curve.dat', 
                '-type', 'f'
            ], capture_output=True, text=True, check=True)
            
            experiment_files = result.stdout.strip().split('\n')
            if not experiment_files or experiment_files == ['']:
                print(f"No experiment files found in {layer_dir}")
                return []
            
            experiments = []
            for exp_file in experiment_files:
                if not exp_file:
                    continue
                    
                exp_path = Path(exp_file)
                exp_id = exp_path.name.replace('_experimental_curve.dat', '')
                if exp_id.startswith('s'):
                    # Check if model file exists
                    model_file = exp_path.parent / f"{exp_id}_model.txt"
                    if model_file.exists():
                        experiments.append(exp_id)
                        print(f"  Found: {exp_id}")
            
            print(f"Found {len(experiments)} experiments")
            
            # Sample if we have more than requested
            if len(experiments) > self.num_experiments:
                experiments = random.sample(experiments, self.num_experiments)
                print(f"Randomly selected {len(experiments)} experiments for processing")
            
            return experiments
            
        except subprocess.CalledProcessError as e:
            print(f"Error finding experiments: {e}")
            return []

    def run_experiment_inference(self, exp_id, priors_type="broad"):
        """Run inference for a single experiment using the new parameterized interface."""
        try:
            result = InferencePipeline.run_experiment_inference(
                experiment_id=exp_id,
                models_list=self.models,
                data_directory=str(self.data_directory),
                priors_type=priors_type,
                output_dir=str(self.output_dir),
                layer_count=self.layer_count
            )
            
            if result['success']:
                print(f"  ✓ {exp_id} completed successfully")
                return result
            else:
                print(f"  ✗ {exp_id} failed: {result['error']}")
                return result
            
        except Exception as e:
            print(f"  ✗ {exp_id} error: {e}")
            return {
                'exp_id': exp_id,
                'success': False,
                'error': str(e),
                'models_results': {}
            }

    def run(self):
        """Run batch inference on all discovered experiments."""
        print(f"Starting Batch Inference Pipeline")
        print("=" * 50)
        print(f"Target experiments: {self.num_experiments}")
        print(f"Layer count: {self.layer_count}")
        print(f"Models: {self.models}")
        print(f"MARIA dataset path: {self.maria_dataset_path}")
        
        # Discover experiments
        experiments = self.discover_experiments()
        
        if not experiments:
            print("No experiments found. Exiting.")
            return
        
        print(f"Processing {len(experiments)} experiments...")
        
        # Process each experiment with both broad and narrow priors
        all_results = {}
        
        for i, exp_id in enumerate(experiments, 1):
            print(f"\n[{i}/{len(experiments)}] Processing {exp_id}...")
            
            exp_results = {
                'experiment_id': exp_id,
                'layer_count': self.layer_count,
                'priors': {}
            }
            
            # Run with both broad and narrow priors
            for priors_type in ['broad', 'narrow']:
                print(f"  Running with {priors_type} priors...")
                result = self.run_experiment_inference(exp_id, priors_type)
                exp_results['priors'][priors_type] = result
            
            all_results[exp_id] = exp_results
            
            # Save individual experiment results
            exp_file = self.output_dir / f"{exp_id}_results.json"
            with open(exp_file, 'w') as f:
                json.dump(exp_results, f, indent=2, default=str)
        
        # Create batch summary
        self.create_batch_summary(all_results)
        
        # Create performance plots  
        self.create_performance_plots(all_results)

    def create_batch_summary(self, all_results):
        """Create summary of batch results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n{'='*80}")
        print(f"BATCH INFERENCE SUMMARY - {self.layer_count}-LAYER EXPERIMENTS")
        print(f"{'='*80}")
        
        # Calculate statistics
        total_experiments = len(all_results)
        successful_broad = 0
        successful_narrow = 0
        
        # Aggregate performance by model and priors type
        model_performance = defaultdict(lambda: defaultdict(list))
        
        for exp_id, exp_result in all_results.items():
            for priors_type in ['broad', 'narrow']:
                priors_result = exp_result['priors'][priors_type]
                if priors_result['success']:
                    if priors_type == 'broad':
                        successful_broad += 1
                    else:
                        successful_narrow += 1
                    
                    # Collect metrics for each model
                    for model_name, model_result in priors_result['models_results'].items():
                        if model_result['success']:
                            # Fit metrics
                            if 'fit_metrics' in model_result:
                                fit_metrics = model_result['fit_metrics']
                                model_performance[model_name][f'{priors_type}_r2'].append(fit_metrics.get('r_squared', 0))
                                model_performance[model_name][f'{priors_type}_mse'].append(fit_metrics.get('mse', float('inf')))
                            
                            # Parameter metrics
                            if 'parameter_metrics' in model_result and model_result['parameter_metrics']:
                                param_metrics = model_result['parameter_metrics']
                                overall_mape = param_metrics['overall']['mape']
                                model_performance[model_name][f'{priors_type}_param_mape'].append(overall_mape)
        
        print(f"Total experiments: {total_experiments}")
        print(f"Successful with broad priors: {successful_broad}")
        print(f"Successful with narrow priors: {successful_narrow}")
        print(f"Success rate (broad): {100*successful_broad/total_experiments:.1f}%")
        print(f"Success rate (narrow): {100*successful_narrow/total_experiments:.1f}%")
        
        # Print model performance summary
        print(f"\nMODEL PERFORMANCE SUMMARY:")
        print("-" * 80)
        
        for model_name in self.models:
            if model_name in model_performance:
                perf = model_performance[model_name]
                print(f"\n{model_name}:")
                
                for priors_type in ['broad', 'narrow']:
                    r2_key = f'{priors_type}_r2'
                    mse_key = f'{priors_type}_mse'
                    mape_key = f'{priors_type}_param_mape'
                    
                    if r2_key in perf and perf[r2_key]:
                        print(f"  {priors_type.title()} priors:")
                        print(f"    R² - Mean: {np.mean(perf[r2_key]):.3f}, Std: {np.std(perf[r2_key]):.3f}")
                        
                        mse_vals = [x for x in perf[mse_key] if x != float('inf')]
                        if mse_vals:
                            print(f"    MSE - Mean: {np.mean(mse_vals):.6f}, Std: {np.std(mse_vals):.6f}")
                        
                        if mape_key in perf and perf[mape_key]:
                            print(f"    Param MAPE - Mean: {np.mean(perf[mape_key]):.1f}%, Std: {np.std(perf[mape_key]):.1f}%")
        
        # Save batch summary
        summary = {
            'timestamp': timestamp,
            'layer_count': self.layer_count,
            'total_experiments': total_experiments,
            'successful_broad': successful_broad,
            'successful_narrow': successful_narrow,
            'models_tested': self.models,
            'model_performance': dict(model_performance),
            'all_results': all_results
        }
        
        summary_file = self.output_dir / f"batch_summary_{self.layer_count}layer_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nBatch summary saved to: {summary_file}")
        return summary

    def create_performance_plots(self, all_results):
        """Create comprehensive 2-column performance visualization plots."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Collect data for plotting
        model_mapes = defaultdict(lambda: defaultdict(list))
        model_param_mapes = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # model -> priors -> param_type -> values
        experiment_mapes = defaultdict(lambda: defaultdict(list))  # priors_type -> exp_id -> mape
        
        for exp_id, exp_result in all_results.items():
            for priors_type in ['broad', 'narrow']:
                priors_result = exp_result['priors'][priors_type]
                if priors_result['success']:
                    exp_mapes_for_priors = []
                    
                    for model_name, model_result in priors_result['models_results'].items():
                        if model_result['success'] and 'parameter_metrics' in model_result:
                            param_metrics = model_result['parameter_metrics']
                            if param_metrics and 'overall' in param_metrics:
                                # Overall MAPE
                                overall_mape = param_metrics['overall']['mape']
                                model_mapes[model_name][priors_type].append(overall_mape)
                                exp_mapes_for_priors.append(overall_mape)
                                
                                # Parameter-specific MAPEs
                                by_type = param_metrics.get('by_type', {})
                                for param_type in ['thickness', 'roughness', 'sld']:
                                    mape_key = f'{param_type}_mape'
                                    if mape_key in by_type:
                                        model_param_mapes[model_name][priors_type][param_type].append(by_type[mape_key])
                    
                    # Calculate average MAPE for this experiment with this priors type
                    if exp_mapes_for_priors:
                        experiment_mapes[priors_type][exp_id] = np.mean(exp_mapes_for_priors)
        
        # Create 2-column, 3-row layout
        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        fig.suptitle(f'Batch Inference Performance Analysis - {len(all_results)} {self.layer_count}-Layer Experiments', fontsize=16, fontweight='bold')
        
        # Column titles
        axes[0, 0].text(0.5, 1.15, 'BROAD PRIORS', transform=axes[0, 0].transAxes, 
                       ha='center', va='center', fontsize=14, fontweight='bold')
        axes[0, 1].text(0.5, 1.15, 'NARROW PRIORS', transform=axes[0, 1].transAxes, 
                       ha='center', va='center', fontsize=14, fontweight='bold')
        
        models = list(model_mapes.keys())
        model_labels = [model.replace('b_mc_point_', '').replace('_conv_standard', '') for model in models]
        
        # Row 1: Average MAPE by Model
        for col, priors_type in enumerate(['broad', 'narrow']):
            ax = axes[0, col]
            
            means = []
            stds = []
            for model in models:
                vals = model_mapes[model][priors_type]
                means.append(np.mean(vals) if vals else 0)
                stds.append(np.std(vals) if vals else 0)
            
            x = np.arange(len(models))
            color = 'skyblue' if priors_type == 'broad' else 'lightcoral'
            bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, color=color, 
                         edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Model')
            ax.set_ylabel('Average MAPE (%)')
            ax.set_title(f'Overall MAPE by Model ({priors_type.title()} Priors)')
            ax.set_xticks(x)
            ax.set_xticklabels(model_labels, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (mean, std) in enumerate(zip(means, stds)):
                if mean > 0:
                    ax.text(i, mean + std + max(means) * 0.02, f'{mean:.1f}%', 
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Row 2: Parameter-specific MAPE by Model
        param_types = ['thickness', 'roughness', 'sld']
        param_colors = {'thickness': '#FF6B6B', 'roughness': '#4ECDC4', 'sld': '#45B7D1'}
        
        for col, priors_type in enumerate(['broad', 'narrow']):
            ax = axes[1, col]
            
            x = np.arange(len(models))
            width = 0.25
            
            for i, param_type in enumerate(param_types):
                medians = []
                for model in models:
                    vals = model_param_mapes[model][priors_type][param_type]
                    medians.append(np.median(vals) if vals else 0)
                
                ax.bar(x + i*width - width, medians, width, 
                      label=param_type.title(), alpha=0.8, color=param_colors[param_type],
                      edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Model')
            ax.set_ylabel('Median MAPE (%)')
            ax.set_title(f'Parameter-Specific MAPE ({priors_type.title()} Priors)')
            ax.set_xticks(x)
            ax.set_xticklabels(model_labels, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        # Row 3: Edge Case Detection
        for col, priors_type in enumerate(['broad', 'narrow']):
            ax = axes[2, col]
            
            exp_data = experiment_mapes[priors_type]
            if exp_data:
                exp_ids = list(exp_data.keys())
                exp_vals = list(exp_data.values())
                exp_indices = range(len(exp_ids))
                
                # Plot all experiments
                ax.plot(exp_indices, exp_vals, 'o-', alpha=0.7, linewidth=1, markersize=4,
                       color='darkblue', label='Experiments')
                
                # Calculate threshold for edge cases
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
                    
                    # Annotate worst edge cases
                    worst_cases = sorted(edge_cases, key=lambda x: x[2], reverse=True)[:3]
                    for i, exp_id, mape in worst_cases:
                        ax.annotate(f'{exp_id}\n{mape:.1f}%', 
                                  xy=(i, mape), xytext=(10, 10), 
                                  textcoords='offset points', fontsize=8,
                                  bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                                  arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                
                # Add threshold line
                ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
                          label=f'Threshold (μ+2σ)')
                
                ax.set_xlabel('Experiment Index')
                ax.set_ylabel('Average MAPE (%)')
                ax.set_title(f'Edge Case Detection ({priors_type.title()} Priors)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Print edge cases for this priors type
                if edge_cases:
                    print(f"\nEdge Cases for {priors_type.title()} Priors (MAPE > {threshold:.1f}%):")
                    for i, exp_id, mape in sorted(edge_cases, key=lambda x: x[2], reverse=True):
                        print(f"  {exp_id}: {mape:.1f}% MAPE")
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # Make room for main title
        
        # Save plot
        plot_file = self.output_dir / f"batch_inference_results_{self.layer_count}layer_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance plots saved to: {plot_file}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run batch inference on MARIA experiments")
    parser.add_argument('--num-experiments', type=int, default=25,
                       help='Number of experiments to process (default: 25)')
    parser.add_argument('--layer-count', type=int, choices=[1, 2], default=2,
                       help='Number of layers to process (1 or 2, default: 2)')
    parser.add_argument('--data-directory', type=str, default='data',
                       help='Data directory path (default: data)')
    return parser.parse_args()


def main():
    """Main function to run the batch inference pipeline."""
    args = parse_arguments()
    
    print(f"Processing {args.layer_count}-layer experiments from MARIA_VIPR_dataset/{args.layer_count}/")
    
    # Run batch inference pipeline
    batch_pipeline = BatchInferencePipeline(
        num_experiments=args.num_experiments,
        layer_count=args.layer_count,
        data_directory=args.data_directory
    )
    
    batch_pipeline.run()


if __name__ == "__main__":
    main()
