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
                "b_mc_point_neutron_conv_standard_L1_InputQDq"
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
        """Create performance visualization plots according to architecture.md."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Collect data for plotting
        model_mapes = defaultdict(lambda: defaultdict(list))
        experiment_mapes = defaultdict(list)  # For edge case detection
        experiment_ids = []
        
        for exp_id, exp_result in all_results.items():
            experiment_ids.append(exp_id)
            exp_avg_mapes = []
            
            for priors_type in ['broad', 'narrow']:
                priors_result = exp_result['priors'][priors_type]
                if priors_result['success']:
                    for model_name, model_result in priors_result['models_results'].items():
                        if model_result['success'] and 'parameter_metrics' in model_result:
                            param_metrics = model_result['parameter_metrics']
                            if param_metrics and 'overall' in param_metrics:
                                mape = param_metrics['overall']['mape']
                                model_mapes[model_name][priors_type].append(mape)
                                exp_avg_mapes.append(mape)
            
            # Calculate average MAPE for this experiment (across all models)
            if exp_avg_mapes:
                experiment_mapes[exp_id] = np.mean(exp_avg_mapes)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot 1: MAPE by model (bar chart as specified in architecture)
        models = list(model_mapes.keys())
        x = np.arange(len(models))
        width = 0.35
        
        broad_means = []
        narrow_means = []
        
        for model in models:
            broad_vals = model_mapes[model]['broad']
            narrow_vals = model_mapes[model]['narrow']
            
            broad_means.append(np.mean(broad_vals) if broad_vals else 0)
            narrow_means.append(np.mean(narrow_vals) if narrow_vals else 0)
        
        ax1.bar(x - width/2, broad_means, width, label='Broad Priors', alpha=0.8, color='skyblue')
        ax1.bar(x + width/2, narrow_means, width, label='Narrow Priors', alpha=0.8, color='lightcoral')
        
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Average MAPE (%)')
        ax1.set_title(f'Parameter Prediction MAPE by Model ({self.layer_count}-layer experiments)')
        ax1.set_xticks(x)
        ax1.set_xticklabels([model.replace('b_mc_point_', '').replace('_conv_standard', '') for model in models], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (broad, narrow) in enumerate(zip(broad_means, narrow_means)):
            if broad > 0:
                ax1.text(i - width/2, broad + max(broad_means) * 0.01, f'{broad:.1f}%', 
                        ha='center', va='bottom', fontsize=8)
            if narrow > 0:
                ax1.text(i + width/2, narrow + max(narrow_means) * 0.01, f'{narrow:.1f}%', 
                        ha='center', va='bottom', fontsize=8)
        
        # Plot 2: MAPE by experiment (line plot to identify edge cases)
        exp_indices = range(len(experiment_ids))
        exp_mapes = [experiment_mapes.get(exp_id, 0) for exp_id in experiment_ids]
        
        ax2.plot(exp_indices, exp_mapes, 'o-', alpha=0.7, linewidth=1, markersize=4)
        ax2.set_xlabel('Experiment Index')
        ax2.set_ylabel('Average MAPE (%)')
        ax2.set_title(f'Average MAPE by Experiment - Edge Case Detection ({self.layer_count}-layer)')
        ax2.grid(True, alpha=0.3)
        
        # Highlight potential edge cases (experiments with high MAPE)
        if exp_mapes:
            threshold = np.mean(exp_mapes) + 2 * np.std(exp_mapes)
            edge_cases = [(i, exp_id, mape) for i, (exp_id, mape) in enumerate(zip(experiment_ids, exp_mapes)) if mape > threshold]
            
            if edge_cases:
                edge_indices = [i for i, _, _ in edge_cases]
                edge_mapes = [mape for _, _, mape in edge_cases]
                ax2.scatter(edge_indices, edge_mapes, color='red', s=50, alpha=0.8, label='Potential Edge Cases')
                ax2.legend()
                
                # Print edge cases
                print(f"\nPotential Edge Cases (MAPE > {threshold:.1f}%):")
                for i, exp_id, mape in edge_cases:
                    print(f"  {exp_id}: {mape:.1f}% MAPE")
        
        plt.tight_layout()
        
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
