#!/usr/bin/env python3
"""
Simple Batch Inference Pipeline for ReflecTorch Models

This script runs the existing inference pipeline on multiple experiments
with appropriate models for each layer configuration.

Usage:
    python batch_inference_pipeline_simple.py [--num-experiments 25] [--layer-counts 1,2]
"""

import numpy as np
import json
import random
from pathlib import Path
import argparse
import sys
from datetime import datetime
from collections import defaultdict

# Import the existing inference pipeline
from inference_pipeline import InferencePipeline

class SimpleBatchInferencePipeline:
    """Simple batch pipeline using the existing inference pipeline."""
    
    def __init__(self, num_experiments=25, layer_counts=[1, 2], data_directory="data"):
        self.num_experiments = num_experiments
        self.layer_counts = layer_counts
        self.data_directory = Path(data_directory)
        self.output_dir = Path("batch_inference_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load MARIA bounds
        self.load_maria_bounds()
        
        # Model sets for different layer counts
        self.model_sets = {
            1: [
                "b_mc_point_neutron_conv_standard_L1_comp",
                "b_mc_point_neutron_conv_standard_L1_InputQDq"
            ],
            2: [
                "mc25",
                "b_mc_point_neutron_conv_standard_L2_comp", 
                "b_mc_point_neutron_conv_standard_L2_InputQDq"
            ]
        }
        
        self.batch_results = {}
        
    def load_maria_bounds(self):
        """Load MARIA dataset prior bounds."""
        bounds_file = Path("maria_dataset_prior_bounds.json")
        if not bounds_file.exists():
            print("Warning: MARIA bounds file not found. Using default bounds.")
            self.maria_bounds = None
            return
            
        with open(bounds_file, 'r') as f:
            self.maria_bounds = json.load(f)
    
    def get_prior_bounds_for_layer_count(self, layer_count):
        """Get prior bounds for specific layer count from MARIA dataset."""
        if not self.maria_bounds:
            # Default bounds if MARIA bounds not available
            if layer_count == 1:
                return [
                    (50, 1500),    # L1 thickness
                    (1, 100),      # ambient/L1 roughness
                    (1, 100),      # L1/substrate roughness
                    (-5, 20),      # L1 SLD
                    (0, 10)        # substrate SLD
                ]
            else:  # layer_count == 2
                return [
                    (10, 800),     # L1 thickness
                    (10, 800),     # L2 thickness
                    (1, 100),      # ambient/L1 roughness
                    (1, 100),      # L1/L2 roughness
                    (1, 100),      # L2/substrate roughness
                    (-5, 20),      # L1 SLD
                    (-5, 20),      # L2 SLD
                    (0, 10)        # substrate SLD
                ]
        
        # Use MARIA bounds
        layer_key = f'{layer_count}_layers'
        if layer_key not in self.maria_bounds:
            raise ValueError(f"No MARIA bounds for {layer_count} layers")
            
        bounds_data = self.maria_bounds[layer_key]
        
        if layer_count == 1:
            layer_1 = bounds_data['parameters']['by_position']['layer_1']
            interface = bounds_data['parameters']['interface']
            
            return [
                (layer_1['thickness']['min'], layer_1['thickness']['max']),
                (interface['fronting_roughness']['min'], interface['fronting_roughness']['max']),
                (layer_1['roughness']['min'], layer_1['roughness']['max']),
                (layer_1['sld']['min'] * 1e6, layer_1['sld']['max'] * 1e6),
                (interface['backing_sld']['min'] * 1e6, interface['backing_sld']['max'] * 1e6)
            ]
        else:  # layer_count == 2
            layer_1 = bounds_data['parameters']['by_position']['layer_1']
            layer_2 = bounds_data['parameters']['by_position']['layer_2']
            interface = bounds_data['parameters']['interface']
            
            return [
                (layer_1['thickness']['min'], layer_1['thickness']['max']),
                (layer_2['thickness']['min'], layer_2['thickness']['max']),
                (interface['fronting_roughness']['min'], interface['fronting_roughness']['max']),
                (layer_1['roughness']['min'], layer_1['roughness']['max']),
                (layer_2['roughness']['min'], layer_2['roughness']['max']),
                (layer_1['sld']['min'] * 1e6, layer_1['sld']['max'] * 1e6),
                (layer_2['sld']['min'] * 1e6, layer_2['sld']['max'] * 1e6),
                (interface['backing_sld']['min'] * 1e6, interface['backing_sld']['max'] * 1e6)
            ]
    
    def get_layer_count_from_model_file(self, model_file_path):
        """Determine layer count from model file."""
        try:
            with open(model_file_path, 'r') as f:
                lines = f.readlines()
            
            layer_count = 0
            for line in lines:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                    
                parts = line.split()
                if len(parts) >= 4:
                    layer_name = parts[0]
                    thickness = parts[2]
                    
                    if (layer_name not in ['fronting', 'backing'] and 
                        thickness != 'inf' and thickness != 'none'):
                        layer_count += 1
            
            return layer_count
        except Exception as e:
            print(f"Error parsing {model_file_path}: {e}")
            return 0
    
    def discover_experiments(self):
        """Find experiments in the data directory."""
        experiments = []
        
        # Look for experiment directories
        for item in self.data_directory.iterdir():
            if item.is_dir() and item.name.startswith('s'):
                exp_curve = item / f"{item.name}_experimental_curve.dat"
                model_file = item / f"{item.name}_model.txt"
                
                if exp_curve.exists() and model_file.exists():
                    layer_count = self.get_layer_count_from_model_file(model_file)
                    if layer_count in self.layer_counts:
                        experiments.append((item.name, layer_count))
        
        # Also check for files directly in data directory
        for file in self.data_directory.glob("s*_experimental_curve.dat"):
            exp_id = file.stem.replace("_experimental_curve", "")
            model_file = self.data_directory / f"{exp_id}_model.txt"
            
            if model_file.exists():
                layer_count = self.get_layer_count_from_model_file(model_file)
                if layer_count in self.layer_counts and exp_id not in [e[0] for e in experiments]:
                    experiments.append((exp_id, layer_count))
        
        print(f"Found {len(experiments)} experiments matching layer criteria")
        
        # Sample if we have more than requested
        if len(experiments) > self.num_experiments:
            experiments = random.sample(experiments, self.num_experiments)
        
        return experiments
    
    def create_config_for_experiment(self, experiment_id, layer_count):
        """Create a temporary config file for an experiment."""
        # Get prior bounds and models for this layer count
        prior_bounds = self.get_prior_bounds_for_layer_count(layer_count)
        models = self.model_sets[layer_count]
        
        # Find data file
        data_paths = [
            self.data_directory / experiment_id / f"{experiment_id}_experimental_curve.dat",
            self.data_directory / f"{experiment_id}_experimental_curve.dat"
        ]
        
        data_path = None
        for path in data_paths:
            if path.exists():
                data_path = str(path)
                break
        
        if not data_path:
            raise FileNotFoundError(f"Data file not found for {experiment_id}")
        
        # Find model file for true parameters
        model_paths = [
            self.data_directory / experiment_id / f"{experiment_id}_model.txt",
            self.data_directory / f"{experiment_id}_model.txt"
        ]
        
        model_path = None
        for path in model_paths:
            if path.exists():
                model_path = str(path)
                break
        
        # Create config
        config = {
            "data_config": {
                "description": f"Batch inference for {experiment_id} ({layer_count}-layer)",
                "data_path": data_path,
                "data_format": "4_column",
                "dq_over_q": 0.1,
                "max_points": None
            },
            "model_configurations": {}
        }
        
        # Add true parameters file if available
        if model_path:
            config["data_config"]["true_parameters_file"] = model_path
        
        # Add model configurations
        for model_name in models:
            config["model_configurations"][model_name] = {
                "config_name": model_name,
                "description": f"{model_name} for {layer_count}-layer system",
                "prior_bounds": prior_bounds,
                "weights_format": "safetensors"
            }
        
        return config
    
    def run_batch_inference(self):
        """Run batch inference on all discovered experiments."""
        print("Starting Simple Batch Inference Pipeline")
        print("=" * 50)
        print(f"Target experiments: {self.num_experiments}")
        print(f"Layer counts: {self.layer_counts}")
        
        # Discover experiments
        experiments = self.discover_experiments()
        
        if not experiments:
            print("No experiments found!")
            return
        
        print(f"Processing {len(experiments)} experiments...")
        
        # Process each experiment
        for i, (experiment_id, layer_count) in enumerate(experiments):
            print(f"\n{'='*60}")
            print(f"EXPERIMENT {i+1}/{len(experiments)}: {experiment_id} ({layer_count}-layer)")
            print(f"{'='*60}")
            
            try:
                # Create config for this experiment
                config = self.create_config_for_experiment(experiment_id, layer_count)
                
                # Create temporary config file
                temp_config_file = self.output_dir / f"temp_config_{experiment_id}.json"
                with open(temp_config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                
                # Run inference pipeline
                pipeline = InferencePipeline(str(temp_config_file))
                pipeline.run_all_models()
                
                # Store results
                self.batch_results[experiment_id] = {
                    'layer_count': layer_count,
                    'results': pipeline.results,
                    'success': True
                }
                
                # Clean up temp config
                temp_config_file.unlink()
                
            except Exception as e:
                print(f"ERROR processing {experiment_id}: {e}")
                self.batch_results[experiment_id] = {
                    'layer_count': layer_count,
                    'success': False,
                    'error': str(e)
                }
        
        # Generate batch summary
        self.create_batch_summary()
    
    def create_batch_summary(self):
        """Create a summary of batch results."""
        print(f"\n{'='*80}")
        print("BATCH INFERENCE SUMMARY")
        print(f"{'='*80}")
        
        successful_experiments = {k: v for k, v in self.batch_results.items() if v['success']}
        failed_experiments = {k: v for k, v in self.batch_results.items() if not v['success']}
        
        print(f"Total experiments: {len(self.batch_results)}")
        print(f"Successful: {len(successful_experiments)}")
        print(f"Failed: {len(failed_experiments)}")
        
        if failed_experiments:
            print(f"\nFailed experiments:")
            for exp_id, result in failed_experiments.items():
                print(f"  - {exp_id}: {result.get('error', 'Unknown error')}")
        
        if successful_experiments:
            # Collect model performance across all experiments
            model_stats = defaultdict(list)
            
            for exp_id, exp_result in successful_experiments.items():
                for model_name, model_result in exp_result['results'].items():
                    if model_result['success']:
                        mse = model_result['fit_metrics']['mse']
                        r2 = model_result['fit_metrics']['r_squared']
                        model_stats[model_name].append({'mse': mse, 'r2': r2, 'exp_id': exp_id})
            
            print(f"\nModel Performance Summary:")
            print("-" * 80)
            print(f"{'Model':<30} {'Tests':<8} {'Avg MSE':<12} {'Avg R²':<10}")
            print("-" * 80)
            
            for model_name, results in model_stats.items():
                num_tests = len(results)
                avg_mse = np.mean([r['mse'] for r in results])
                avg_r2 = np.mean([r['r2'] for r in results])
                
                print(f"{model_name:<30} {num_tests:<8} {avg_mse:<12.6f} {avg_r2:<10.4f}")
            
            # Best model overall
            if model_stats:
                best_model = min(model_stats.items(), key=lambda x: np.mean([r['mse'] for r in x[1]]))
                print(f"\nBest model overall: {best_model[0]}")
                print(f"Average MSE: {np.mean([r['mse'] for r in best_model[1]]):.6f}")
        
        # Save batch results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"batch_summary_{timestamp}.json"
        
        # Prepare serializable results
        serializable_results = {}
        for exp_id, exp_result in self.batch_results.items():
            if exp_result['success']:
                serializable_results[exp_id] = {
                    'layer_count': exp_result['layer_count'],
                    'success': True,
                    'models': {}
                }
                
                for model_name, model_result in exp_result['results'].items():
                    if model_result['success']:
                        serializable_results[exp_id]['models'][model_name] = {
                            'mse': float(model_result['fit_metrics']['mse']),
                            'r_squared': float(model_result['fit_metrics']['r_squared']),
                            'l1_loss': float(model_result['fit_metrics']['l1_loss']),
                            'predicted_params': model_result['predicted_params'].tolist()
                        }
            else:
                serializable_results[exp_id] = {
                    'success': False,
                    'error': exp_result['error']
                }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nBatch results saved to: {results_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Simple Batch Inference Pipeline")
    parser.add_argument('--num-experiments', type=int, default=25,
                       help='Number of experiments to process (default: 25)')
    parser.add_argument('--layer-counts', type=str, default='1,2',
                       help='Comma-separated layer counts (default: 1,2)')
    parser.add_argument('--data-directory', default='data',
                       help='Data directory (default: data)')
    
    args = parser.parse_args()
    
    layer_counts = [int(x.strip()) for x in args.layer_counts.split(',')]
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    pipeline = SimpleBatchInferencePipeline(
        num_experiments=args.num_experiments,
        layer_counts=layer_counts,
        data_directory=args.data_directory
    )
    
    pipeline.run_batch_inference()


if __name__ == "__main__":
    main()
