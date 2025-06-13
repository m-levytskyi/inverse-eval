#!/usr/bin/env python3
"""
Simple Batch Inference Pipeline for ReflecTorch Models

This script runs the existing inference pipeline on multiple experiments
with appropriate models for each layer configuration.

Usage:
    python batch_inference_pipeline_simple.py [--num-experiments 25] [--layer-count 1]
"""

import numpy as np
import json
import random
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
from datetime import datetime
from collections import defaultdict

# Import the existing inference pipeline
from inference_pipeline import InferencePipeline

class SimpleBatchInferencePipeline:
    """Simple batch pipeline using the existing inference pipeline."""
    
    def __init__(self, num_experiments=25, layer_count=2, data_directory="data"):
        self.num_experiments = num_experiments
        self.layer_count = layer_count
        self.data_directory = Path(data_directory)
        self.maria_dataset_path = Path(data_directory) / "MARIA_VIPR_dataset"
        self.output_dir = Path("batch_inference_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load MARIA bounds
        self.load_maria_bounds()
        
        # Model sets for different layer counts - remove .yaml extensions for config names
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
            raise ValueError(f"No models configured for {self.layer_count} layers")
        self.models = self.model_sets[self.layer_count]
        
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
        # Default bounds
        if layer_count == 1:
            default_bounds = [
                [1.0, 1000.0],  # L1 thickness
                [0.0, 60.0],    # ambient/L1 roughness
                [0.0, 60.0],    # L1/substrate roughness
                [-8.0, 16.0],   # L1 SLD
                [-8.0, 16.0]    # substrate SLD
            ]
        elif layer_count == 2:
            default_bounds = [
                [1.0, 500.0],   # L1 thickness
                [1.0, 500.0],   # L2 thickness
                [0.0, 60.0],    # ambient/L1 roughness
                [0.0, 60.0],    # L1/L2 roughness
                [0.0, 60.0],    # L2/substrate roughness
                [-8.0, 16.0],   # L1 SLD
                [-8.0, 16.0],   # L2 SLD
                [-8.0, 16.0]    # substrate SLD
            ]
        else:
            raise ValueError(f"Unsupported layer count: {layer_count}")
        
        # Use MARIA bounds if available, otherwise use defaults
        if self.maria_bounds:
            key = f"{layer_count}_layers"
            if key in self.maria_bounds:
                maria_data = self.maria_bounds[key]
                return self.convert_maria_bounds_to_prior_bounds(maria_data, layer_count)
        
        # Warn once per layer count and return defaults
        if not hasattr(self, '_warned_missing_bounds'):
            self._warned_missing_bounds = set()
        if layer_count not in self._warned_missing_bounds:
            print(f"Warning: No MARIA bounds found for {layer_count} layers. Using defaults.")
            self._warned_missing_bounds.add(layer_count)
        return default_bounds
    
    def convert_maria_bounds_to_prior_bounds(self, maria_data, layer_count):
        """Convert MARIA bounds data to prior bounds format matching the working models."""
        prior_bounds = []
        
        if layer_count == 1:
            # For 1-layer: use mc25 (2-layer) format but with combined parameters
            # mc25 expects: [L1_thick, L2_thick, amb/L1_rough, L1/L2_rough, L2/sub_rough, L1_SLD, L2_SLD, sub_SLD]
            
            # Get combined thickness for both layers
            combined_thickness = maria_data["parameters"]["overall"]["thickness"]["max"]
            thickness_bounds = [
                maria_data["parameters"]["overall"]["thickness"]["min"],
                combined_thickness
            ]
            
            # Split the thickness between two layers for mc25
            prior_bounds.append([thickness_bounds[0] * 0.3, thickness_bounds[1] * 0.7])  # L1 thickness
            prior_bounds.append([thickness_bounds[0] * 0.3, thickness_bounds[1] * 0.7])  # L2 thickness
            
            # Roughness bounds
            roughness_bounds = [
                maria_data["parameters"]["overall"]["roughness"]["min"],
                maria_data["parameters"]["overall"]["roughness"]["max"]
            ]
            prior_bounds.append(roughness_bounds)  # ambient/L1 roughness
            prior_bounds.append(roughness_bounds)  # L1/L2 roughness
            prior_bounds.append(roughness_bounds)  # L2/substrate roughness
            
            # SLD bounds
            sld_bounds = [
                maria_data["parameters"]["overall"]["sld"]["min"] * 1e6,
                maria_data["parameters"]["overall"]["sld"]["max"] * 1e6
            ]
            prior_bounds.append(sld_bounds)  # L1 SLD
            prior_bounds.append(sld_bounds)  # L2 SLD
            prior_bounds.append(sld_bounds)  # substrate SLD
            
        elif layer_count == 2:
            # For 2-layer: Generate both mc25 (8 params) and L3_comp (11 params) formats
            # For mc25: [L1_thick, L2_thick, amb/L1_rough, L1/L2_rough, L2/sub_rough, L1_SLD, L2_SLD, sub_SLD]
            
            # Layer-specific thickness bounds
            l1_thickness_bounds = [
                maria_data["parameters"]["by_position"]["layer_1"]["thickness"]["min"],
                maria_data["parameters"]["by_position"]["layer_1"]["thickness"]["max"]
            ]
            l2_thickness_bounds = [
                maria_data["parameters"]["by_position"]["layer_2"]["thickness"]["min"],
                maria_data["parameters"]["by_position"]["layer_2"]["thickness"]["max"]
            ]
            
            # For b_mc_point_neutron_conv_standard_L3_comp (11 parameters):
            # [L1_thick, L2_thick, L3_thick, amb/L1_rough, L1/L2_rough, L2/L3_rough, L3/sub_rough, L1_SLD, L2_SLD, L3_SLD, sub_SLD]
            
            # Split L2 into L2 and L3 for the 3-layer model
            l2_split = l2_thickness_bounds[1] * 0.6
            l3_split = l2_thickness_bounds[1] * 0.4
            
            prior_bounds.append(l1_thickness_bounds)  # L1 thickness
            prior_bounds.append([l2_thickness_bounds[0], l2_split])  # L2 thickness (reduced)
            prior_bounds.append([l2_thickness_bounds[0], l3_split])  # L3 thickness (new layer)
            
            # Roughness bounds
            roughness_bounds = [
                maria_data["parameters"]["overall"]["roughness"]["min"],
                maria_data["parameters"]["overall"]["roughness"]["max"]
            ]
            prior_bounds.append(roughness_bounds)  # ambient/L1 roughness
            prior_bounds.append(roughness_bounds)  # L1/L2 roughness
            prior_bounds.append(roughness_bounds)  # L2/L3 roughness
            prior_bounds.append(roughness_bounds)  # L3/substrate roughness
            
            # SLD bounds
            l1_sld_bounds = [
                maria_data["parameters"]["by_position"]["layer_1"]["sld"]["min"] * 1e6,
                maria_data["parameters"]["by_position"]["layer_1"]["sld"]["max"] * 1e6
            ]
            l2_sld_bounds = [
                maria_data["parameters"]["by_position"]["layer_2"]["sld"]["min"] * 1e6,
                maria_data["parameters"]["by_position"]["layer_2"]["sld"]["max"] * 1e6
            ]
            substrate_sld_bounds = [
                maria_data["parameters"]["overall"]["sld"]["min"] * 1e6,
                maria_data["parameters"]["overall"]["sld"]["max"] * 1e6
            ]
            
            prior_bounds.append(l1_sld_bounds)      # L1 SLD
            prior_bounds.append(l2_sld_bounds)      # L2 SLD
            prior_bounds.append(l2_sld_bounds)      # L3 SLD (same as L2)
            prior_bounds.append(substrate_sld_bounds)  # substrate SLD
        
        return prior_bounds
    
    def get_layer_count_from_model_file(self, model_file_path):
        """Extract layer count from model file."""
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
        """Find experiments in the MARIA dataset directory for the specific layer count."""
        layer_dir = self.maria_dataset_path / str(self.layer_count)
        
        if not layer_dir.exists():
            print(f"Warning: MARIA dataset directory not found: {layer_dir}")
            print("Falling back to data directory search...")
            return self.discover_experiments_fallback()
        
        print(f"Searching for {self.layer_count}-layer experiments in: {layer_dir}")
        experiments = []
        
        # Look for experiment files in MARIA dataset
        experiment_files = {}
        for item in layer_dir.iterdir():
            if item.is_file() and item.name.endswith('_experimental_curve.dat'):
                exp_id = item.name.replace('_experimental_curve.dat', '')
                if exp_id.startswith('s'):
                    experiment_files[exp_id] = item.parent
        
        for exp_id, exp_dir in experiment_files.items():
            exp_curve = exp_dir / f"{exp_id}_experimental_curve.dat"
            model_file = exp_dir / f"{exp_id}_model.txt"
            
            if exp_curve.exists() and model_file.exists():
                # Verify layer count matches
                layer_count = self.get_layer_count_from_model_file(model_file)
                if layer_count == self.layer_count:
                    experiments.append((exp_id, layer_count, str(exp_dir)))
                    print(f"  Found: {exp_id}")
                else:
                    print(f"  Skipped {exp_id}: layer count mismatch ({layer_count} != {self.layer_count})")
        
        if not experiments:
            print(f"No experiments found in {layer_dir}. Checking fallback locations...")
            return self.discover_experiments_fallback()
        
        print(f"Found {len(experiments)} experiments matching {self.layer_count}-layer criteria")
        
        # Sample if we have more than requested
        if len(experiments) > self.num_experiments:
            experiments = random.sample(experiments, self.num_experiments)
            print(f"Randomly selected {len(experiments)} experiments for processing")
        
        return experiments
    
    def discover_experiments_fallback(self):
        """Fallback method to find experiments in the main data directory."""
        experiments = []
        
        # Look for experiment directories in main data folder
        for item in self.data_directory.iterdir():
            if item.is_dir() and item.name.startswith('s'):
                exp_curve = item / f"{item.name}_experimental_curve.dat"
                model_file = item / f"{item.name}_model.txt"
                
                if exp_curve.exists() and model_file.exists():
                    layer_count = self.get_layer_count_from_model_file(model_file)
                    if layer_count == self.layer_count:
                        experiments.append((item.name, layer_count, str(item)))
        
        # Also check for files directly in data directory
        for file in self.data_directory.glob("s*_experimental_curve.dat"):
            exp_id = file.stem.replace("_experimental_curve", "")
            model_file = self.data_directory / f"{exp_id}_model.txt"
            
            if model_file.exists():
                layer_count = self.get_layer_count_from_model_file(model_file)
                if layer_count == self.layer_count and exp_id not in [e[0] for e in experiments]:
                    experiments.append((exp_id, layer_count, str(self.data_directory)))
        
        print(f"Found {len(experiments)} experiments matching {self.layer_count}-layer criteria in fallback locations")
        
        # Sample if we have more than requested
        if len(experiments) > self.num_experiments:
            experiments = random.sample(experiments, self.num_experiments)
            print(f"Randomly selected {len(experiments)} experiments for processing")
        
        return experiments
    
    def create_experiment_config(self, exp_id, exp_path, layer_count):
        """Create a temporary configuration file for the experiment."""
        config = {
            "data_config": {
                "description": f"Batch inference for {exp_id} ({layer_count}-layer)",
                "data_format": "4_column",
                "data_path": f"{exp_path}/{exp_id}_experimental_curve.dat" if exp_path.endswith(exp_id) else f"{exp_path}/{exp_id}_experimental_curve.dat",
                "true_parameters_file": f"{exp_path}/{exp_id}_model.txt" if exp_path.endswith(exp_id) else f"{exp_path}/{exp_id}_model.txt",
                "dq_over_q": 0.1,
                "max_points": None
            },
            "model_configurations": {}
        }
        
        # Add models for the appropriate layer count with dynamic configuration
        for model_name in self.models:
            # Determine model type and parameter count from the model name
            if "L1" in model_name:
                expected_layers = 1
                param_count = 5  # Standard 1-layer: [thickness, amb_rough, sub_rough, layer_sld, sub_sld]
            elif "L2" in model_name:
                expected_layers = 2
                param_count = 8  # Standard 2-layer: [L1_thick, L2_thick, amb_rough, L1L2_rough, L2sub_rough, L1_sld, L2_sld, sub_sld]
            elif model_name == "mc25":
                expected_layers = 2
                param_count = 8  # mc25 is always 8 parameters
            else:
                # Default to 2-layer if unclear
                expected_layers = 2
                param_count = 8
            
            # Generate appropriate bounds based on parameter count
            if param_count == 5:
                prior_bounds = self.get_bounds_for_param_count(5, layer_count)
                parameter_names = [
                    "L1 thickness (Å)",
                    "ambient/L1 roughness (Å)",
                    "L1/substrate roughness (Å)",
                    "L1 SLD (×10⁻⁶ Å⁻²)",
                    "substrate SLD (×10⁻⁶ Å⁻²)"
                ]
            elif param_count == 8:
                prior_bounds = self.get_bounds_for_param_count(8, layer_count)
                parameter_names = [
                    "L1 thickness (Å)",
                    "L2 thickness (Å)",
                    "ambient/L1 roughness (Å)",
                    "L1/L2 roughness (Å)",
                    "L2/substrate roughness (Å)",
                    "L1 SLD (×10⁻⁶ Å⁻²)",
                    "L2 SLD (×10⁻⁶ Å⁻²)",
                    "substrate SLD (×10⁻⁶ Å⁻²)"
                ]
            elif param_count == 11:
                # Absorption model: model_with_absorption 
                # Based on config: 2-layer model with real + imaginary SLDs
                # Parameters: [L1_thick, L2_thick, amb_rough, L1L2_rough, L2sub_rough, L1_sld, L2_sld, sub_sld, L1_isld, L2_isld, sub_isld]
                prior_bounds = self.get_bounds_for_param_count_absorption(layer_count)
                parameter_names = [
                    "L1 thickness (Å)",
                    "L2 thickness (Å)",
                    "ambient/L1 roughness (Å)", 
                    "L1/L2 roughness (Å)",
                    "L2/substrate roughness (Å)",
                    "L1 SLD (×10⁻⁶ Å⁻²)",
                    "L2 SLD (×10⁻⁶ Å⁻²)",
                    "substrate SLD (×10⁻⁶ Å⁻²)",
                    "L1 iSLD (×10⁻⁶ Å⁻²)",
                    "L2 iSLD (×10⁻⁶ Å⁻²)",
                    "substrate iSLD (×10⁻⁶ Å⁻²)"
                ]
            else:
                # Fallback - assume 8 parameters
                prior_bounds = self.get_bounds_for_param_count(8, layer_count)
                parameter_names = [f"Parameter_{i+1}" for i in range(param_count)]
            
            config["model_configurations"][model_name] = {
                "config_name": model_name,
                "description": f"{model_name} model for {layer_count}-layer system",
                "weights_format": "safetensors",
                "prior_bounds": prior_bounds,
                "parameter_names": parameter_names
            }
        
        # Save the configuration to a temporary file
        config_dir = self.output_dir
        config_file = config_dir / f"temp_config_{exp_id}.json"
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config_file
    
    def run_experiment_inference(self, exp_id, exp_path, layer_count):
        """Run inference on a single experiment with all appropriate models."""
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {exp_id} ({layer_count}-layer)")
        print(f"Path: {exp_path}")
        print(f"{'='*60}")
        
        try:
            # Create temporary config for this experiment
            config_file = self.create_experiment_config(exp_id, exp_path, layer_count)
            
            # Initialize and run the inference pipeline
            pipeline = InferencePipeline(config_file)
            pipeline.run_all_models()
            
            # Extract and store results
            results = {
                'exp_id': exp_id,
                'layer_count': layer_count,
                'results': {},
                'parameter_comparisons': {},
                'success': True
            }
            
            # Extract results from the pipeline
            for model_name, model_result in pipeline.results.items():
                if model_result['success']:
                    # Store basic results
                    results['results'][model_name] = {
                        'success': True,
                        'fit_metrics': model_result['fit_metrics'],
                        'predicted_params': model_result['predicted_params'].tolist(),
                        'polished_params': model_result['polished_params'].tolist(),
                        'param_names': model_result['param_names']
                    }
                    
                    # Add parameter metrics if available
                    if 'parameter_metrics' in model_result and model_result['parameter_metrics'] is not None:
                        results['results'][model_name]['parameter_metrics'] = model_result['parameter_metrics']
                else:
                    results['results'][model_name] = {
                        'success': False,
                        'error': model_result['error']
                    }
            
            return results
            
        except Exception as e:
            print(f"Error processing experiment {exp_id}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'exp_id': exp_id,
                'layer_count': layer_count,
                'success': False,
                'error': str(e)
            }
    
    def run(self):
        """Run the batch inference pipeline."""
        print(f"Starting Simple Batch Inference Pipeline")
        print("=" * 50)
        print(f"Target experiments: {self.num_experiments}")
        print(f"Layer count: {self.layer_count}")
        print(f"MARIA dataset path: {self.maria_dataset_path}")
        
        # Discover experiments
        experiments = self.discover_experiments()
        
        if not experiments:
            print("No experiments found. Exiting.")
            return
        
        print(f"Processing {len(experiments)} experiments...")
        
        # Run inference on each experiment
        successful_experiments = {}
        failed_experiments = {}
        
        for i, (exp_id, layer_count, exp_path) in enumerate(experiments):
            print(f"\n{'='*60}")
            print(f"EXPERIMENT {i+1}/{len(experiments)}: {exp_id} ({layer_count}-layer)")
            print(f"Path: {exp_path}")
            print(f"{'='*60}")
            
            result = self.run_experiment_inference(exp_id, exp_path, layer_count)
            
            if result['success']:
                successful_experiments[exp_id] = result
            else:
                failed_experiments[exp_id] = result
        
        # Create batch summary
        self.batch_results = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'num_experiments': len(experiments),
            'successful': len(successful_experiments),
            'failed': len(failed_experiments),
            'layer_count': self.layer_count,
            'models': self.models,
            'experiments': {
                'successful': successful_experiments,
                'failed': failed_experiments
            }
        }
        
        # Generate batch summary
        self.create_batch_summary()
    
    def create_batch_summary(self):
        """Create summary of batch results with detailed parameter metrics."""
        print(f"\n{'='*80}")
        print(f"BATCH INFERENCE SUMMARY - {self.layer_count}-LAYER EXPERIMENTS")
        print(f"{'='*80}")
        
        successful_experiments = self.batch_results['experiments']['successful']
        failed_experiments = self.batch_results['experiments']['failed']
        
        print(f"Total experiments: {self.batch_results['num_experiments']}")
        print(f"Successful: {self.batch_results['successful']}")
        print(f"Failed: {self.batch_results['failed']}")
        print(f"Layer count: {self.layer_count}")
        print(f"Models tested: {', '.join(self.models)}")
        
        if failed_experiments:
            print(f"\nFailed experiments:")
            for exp_id, result in failed_experiments.items():
                print(f"  - {exp_id}: {result.get('error', 'Unknown error')}")
        
        if successful_experiments:
            # Collect model performance across all experiments
            model_stats = defaultdict(list)
            parameter_stats = defaultdict(list)
            
            for exp_id, exp_result in successful_experiments.items():
                for model_name, model_result in exp_result['results'].items():
                    if model_result['success']:
                        # Collect fit metrics
                        mse = model_result['fit_metrics']['mse']
                        r2 = model_result['fit_metrics']['r_squared']
                        l1_loss = model_result['fit_metrics']['l1_loss']
                        model_stats[model_name].append({
                            'mse': mse, 
                            'r2': r2, 
                            'l1_loss': l1_loss,
                            'exp_id': exp_id
                        })
                        
                        # Collect parameter metrics if available
                        if 'parameter_metrics' in model_result:
                            param_metrics = model_result['parameter_metrics']
                            parameter_stats[model_name].append({
                                'overall_mape': param_metrics['overall']['mape'],
                                'overall_mse': param_metrics['overall']['mse'],
                                'thickness_mape': param_metrics['by_type']['thickness_mape'],
                                'thickness_mse': param_metrics['by_type']['thickness_mse'],
                                'roughness_mape': param_metrics['by_type']['roughness_mape'],
                                'roughness_mse': param_metrics['by_type']['roughness_mse'],
                                'sld_mape': param_metrics['by_type']['sld_mape'],
                                'sld_mse': param_metrics['by_type']['sld_mse'],
                                'exp_id': exp_id
                            })
            
            print(f"\nModel Performance Summary for {self.layer_count}-Layer Systems:")
            print("=" * 100)
            print(f"{'Model':<40} {'Tests':<8} {'Avg MSE':<12} {'Std MSE':<12} {'Avg R²':<10} {'Avg L1':<12}")
            print("=" * 100)
            
            model_performance = []
            for model_name, results in model_stats.items():
                num_tests = len(results)
                mse_values = [r['mse'] for r in results]
                r2_values = [r['r2'] for r in results]
                l1_values = [r['l1_loss'] for r in results]
                
                avg_mse = np.mean(mse_values)
                std_mse = np.std(mse_values)
                avg_r2 = np.mean(r2_values)
                avg_l1 = np.mean(l1_values)
                
                model_performance.append({
                    'model': model_name,
                    'avg_mse': avg_mse,
                    'std_mse': std_mse,
                    'avg_r2': avg_r2,
                    'avg_l1': avg_l1,
                    'num_tests': num_tests
                })
                
                print(f"{model_name:<40} {num_tests:<8} {avg_mse:<12.6f} {std_mse:<12.6f} {avg_r2:<10.4f} {avg_l1:<12.6f}")
            
            # Calculate MSE ranking
            mse_ranking = sorted(model_performance, key=lambda x: x['avg_mse'])
            r2_ranking = sorted(model_performance, key=lambda x: x['avg_r2'], reverse=True)
            
            print(f"\n{self.layer_count}-Layer Model Rankings:")
            print("=" * 60)
            
            print("By MSE (lower is better):")
            for i, model in enumerate(mse_ranking):
                print(f"  {i+1}. {model['model']}: {model['avg_mse']:.6f} ± {model['std_mse']:.6f}")
            
            print("\nBy R² (higher is better):")
            for i, model in enumerate(r2_ranking):
                print(f"  {i+1}. {model['model']}: {model['avg_r2']:.4f}")
            
            # Parameter performance if available
            if parameter_stats:
                print(f"\nParameter Prediction Performance:")
                print("=" * 100)
                print(f"{'Model':<40} {'Tests':<8} {'MAPE %':<10} {'Thick MAPE':<12} {'Rough MAPE':<12} {'SLD MAPE':<12}")
                print("=" * 100)
                
                param_performance = []
                for model_name, results in parameter_stats.items():
                    num_tests = len(results)
                    
                    if num_tests > 0:
                        overall_mape = np.mean([r['overall_mape'] for r in results])
                        overall_mse = np.mean([r['overall_mse'] for r in results])
                        thickness_mape = np.mean([r['thickness_mape'] for r in results])
                        roughness_mape = np.mean([r['roughness_mape'] for r in results])
                        sld_mape = np.mean([r['sld_mape'] for r in results])
                        
                        param_performance.append({
                            'model': model_name,
                            'overall_mape': overall_mape,
                            'overall_mse': overall_mse,
                            'thickness_mape': thickness_mape,
                            'roughness_mape': roughness_mape,
                            'sld_mape': sld_mape,
                            'num_tests': num_tests
                        })
                        
                        print(f"{model_name:<40} {num_tests:<8} {overall_mape:<10.2f} {thickness_mape:<12.2f} {roughness_mape:<12.2f} {sld_mape:<12.2f}")
                
                # Parameter MAPE ranking
                if param_performance:
                    param_ranking = sorted(param_performance, key=lambda x: x['overall_mape'])
                    
                    print(f"\nParameter Accuracy Ranking (by MAPE, lower is better):")
                    for i, model in enumerate(param_ranking):
                        print(f"  {i+1}. {model['model']}: {model['overall_mape']:.2f}%")
            
            # Find best model overall
            if mse_ranking:
                best_model = mse_ranking[0]['model']
                best_mse = mse_ranking[0]['avg_mse']
                best_r2 = next(m['avg_r2'] for m in model_performance if m['model'] == best_model)
                
                print(f"\nBEST {self.layer_count}-LAYER MODEL: {best_model}")
                print(f"   Average MSE: {best_mse:.6f} ± {mse_ranking[0]['std_mse']:.6f}")
                print(f"   Average R²: {best_r2:.4f}")
                print(f"   Tests: {mse_ranking[0]['num_tests']}")
                
                # Create summary statistics for batch results
                summary_stats = {
                    'layer_count': self.layer_count,
                    'total_experiments': len(successful_experiments) + len(failed_experiments),
                    'successful_experiments': len(successful_experiments),
                    'failed_experiments': len(failed_experiments),
                    'model_performance': model_performance,
                    'mse_ranking': [{'model': m['model'], 'avg_mse': m['avg_mse'], 'std_mse': m['std_mse']} for m in mse_ranking],
                    'r2_ranking': [{'model': m['model'], 'avg_r2': m['avg_r2']} for m in r2_ranking],
                    'best_model': {
                        'name': best_model,
                        'avg_mse': best_mse,
                        'avg_r2': best_r2,
                        'num_tests': mse_ranking[0]['num_tests']
                    }
                }
            else:
                print(f"\nNo successful models for {self.layer_count}-layer experiments.")
                summary_stats = {
                    'layer_count': self.layer_count,
                    'total_experiments': len(successful_experiments) + len(failed_experiments),
                    'successful_experiments': len(successful_experiments),
                    'failed_experiments': len(failed_experiments),
                    'model_performance': [],
                    'mse_ranking': [],
                    'r2_ranking': [],
                    'best_model': None
                }
            
            if parameter_stats:
                summary_stats['parameter_performance'] = param_performance
            
            # Save detailed batch results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_file = self.output_dir / f"batch_summary_{self.layer_count}layer_{timestamp}.json"
            
            with open(batch_file, 'w') as f:
                json.dump({
                    'summary': summary_stats,
                    'detailed_results': self.batch_results
                }, f, indent=2)
            
            print(f"\nDetailed batch results saved to: {batch_file}")
            
            # Create visualization of model performance across experiments
            self.create_batch_visualization(model_stats, parameter_stats)
    def create_batch_visualization(self, model_stats, parameter_stats):
        """Create visualizations for model performance across all experiments."""
        # Skip if we have fewer than 2 successful experiments
        if sum(len(results) for results in model_stats.values()) < 2:
            return
        
        # Create a plot for model MSE comparison
        fig, axes = plt.subplots(2, 1, figsize=(12, 12))
        
        # Sort models by average MSE
        model_names = list(model_stats.keys())
        avg_mse_by_model = {model: np.mean([r['mse'] for r in results]) 
                          for model, results in model_stats.items()}
        model_names.sort(key=lambda x: avg_mse_by_model[x])
        
        # Plot MSE for each model
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
        bar_width = 0.8 / len(model_names)
        
        # Plot 1: Average MSE by model
        avg_mse_values = [avg_mse_by_model[model] for model in model_names]
        std_mse_values = [np.std([r['mse'] for r in model_stats[model]]) for model in model_names]
        
        x = np.arange(1)
        for i, model in enumerate(model_names):
            axes[0].bar(x + i*bar_width, avg_mse_values[i], bar_width, 
                      label=model, color=colors[i], yerr=std_mse_values[i])
        
        axes[0].set_xticks([])
        axes[0].set_title(f"Average MSE by Model ({self.layer_count}-layer systems)")
        axes[0].set_ylabel("Mean Squared Error (MSE)")
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, model in enumerate(model_names):
            axes[0].text(x + i*bar_width, avg_mse_values[i], 
                       f"{avg_mse_values[i]:.6f}", ha='center', va='bottom', 
                       fontsize=8, rotation=90)
        
        # Plot 2: R² by model
        avg_r2_values = [np.mean([r['r2'] for r in model_stats[model]]) for model in model_names]
        std_r2_values = [np.std([r['r2'] for r in model_stats[model]]) for model in model_names]
        
        for i, model in enumerate(model_names):
            axes[1].bar(x + i*bar_width, avg_r2_values[i], bar_width, 
                      label=model, color=colors[i], yerr=std_r2_values[i])
        
        axes[1].set_xticks([])
        axes[1].set_title(f"Average R² by Model ({self.layer_count}-layer systems)")
        axes[1].set_ylabel("R² Score")
        axes[1].grid(True, alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, model in enumerate(model_names):
            axes[1].text(x + i*bar_width, max(0, avg_r2_values[i]), 
                       f"{avg_r2_values[i]:.4f}", ha='center', va='bottom', 
                       fontsize=8, rotation=90)
        
        # Add legend
        fig.legend(model_names, loc='upper center', bbox_to_anchor=(0.5, 0.08), 
                 ncol=min(3, len(model_names)))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.output_dir / f"model_performance_{self.layer_count}layer_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        
        print(f"Model performance visualization saved to: {plot_file}")
        
        # Plot parameter metrics if available
        if parameter_stats and sum(len(results) for results in parameter_stats.values()) >= 2:
            self.create_parameter_comparison_plot(parameter_stats)
    
    def create_parameter_comparison_plot(self, parameter_stats):
        """Create visualization for parameter prediction accuracy across models."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        # Sort models by overall MAPE
        model_names = list(parameter_stats.keys())
        avg_mape_by_model = {model: np.mean([r['overall_mape'] for r in results]) 
                           for model, results in parameter_stats.items()}
        model_names.sort(key=lambda x: avg_mape_by_model[x])
        
        # Colors for consistent model representation
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
        bar_width = 0.8 / len(model_names)
        
        # Plot 1: Overall Parameter MAPE
        avg_mape_values = [avg_mape_by_model[model] for model in model_names]
        x = np.arange(1)
        for i, model in enumerate(model_names):
            axes[0].bar(x + i*bar_width, avg_mape_values[i], bar_width, 
                      label=model, color=colors[i])
        
        axes[0].set_xticks([])
        axes[0].set_title(f"Overall Parameter MAPE ({self.layer_count}-layer systems)")
        axes[0].set_ylabel("MAPE (%)")
        axes[0].grid(True, alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, model in enumerate(model_names):
            axes[0].text(x + i*bar_width, avg_mape_values[i], 
                       f"{avg_mape_values[i]:.2f}%", ha='center', va='bottom', 
                       fontsize=8, rotation=90)
        
        # Plot 2: Thickness MAPE
        avg_thickness_mape = [np.mean([r['thickness_mape'] for r in parameter_stats[model]]) 
                            for model in model_names]
        for i, model in enumerate(model_names):
            axes[1].bar(x + i*bar_width, avg_thickness_mape[i], bar_width, 
                      label=model, color=colors[i])
        
        axes[1].set_xticks([])
        axes[1].set_title(f"Thickness Parameter MAPE ({self.layer_count}-layer systems)")
        axes[1].set_ylabel("MAPE (%)")
        axes[1].grid(True, alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, model in enumerate(model_names):
            axes[1].text(x + i*bar_width, avg_thickness_mape[i], 
                       f"{avg_thickness_mape[i]:.2f}%", ha='center', va='bottom', 
                       fontsize=8, rotation=90)
        
        # Plot 3: Roughness MAPE
        avg_roughness_mape = [np.mean([r['roughness_mape'] for r in parameter_stats[model]]) 
                            for model in model_names]
        for i, model in enumerate(model_names):
            axes[2].bar(x + i*bar_width, avg_roughness_mape[i], bar_width, 
                      label=model, color=colors[i])
        
        axes[2].set_xticks([])
        axes[2].set_title(f"Roughness Parameter MAPE ({self.layer_count}-layer systems)")
        axes[2].set_ylabel("MAPE (%)")
        axes[2].grid(True, alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, model in enumerate(model_names):
            axes[2].text(x + i*bar_width, avg_roughness_mape[i], 
                       f"{avg_roughness_mape[i]:.2f}%", ha='center', va='bottom', 
                       fontsize=8, rotation=90)
        
        # Plot 4: SLD MAPE
        avg_sld_mape = [np.mean([r['sld_mape'] for r in parameter_stats[model]]) 
                      for model in model_names]
        for i, model in enumerate(model_names):
            axes[3].bar(x + i*bar_width, avg_sld_mape[i], bar_width, 
                      label=model, color=colors[i])
        
        axes[3].set_xticks([])
        axes[3].set_title(f"SLD Parameter MAPE ({self.layer_count}-layer systems)")
        axes[3].set_ylabel("MAPE (%)")
        axes[3].grid(True, alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, model in enumerate(model_names):
            axes[3].text(x + i*bar_width, avg_sld_mape[i], 
                       f"{avg_sld_mape[i]:.2f}%", ha='center', va='bottom', 
                       fontsize=8, rotation=90)
        
        # Add legend
        fig.legend(model_names, loc='upper center', bbox_to_anchor=(0.5, 0.08), 
                 ncol=min(3, len(model_names)))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.output_dir / f"parameter_accuracy_{self.layer_count}layer_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        
        print(f"Parameter accuracy visualization saved to: {plot_file}")
    
    def get_bounds_for_param_count(self, param_count, layer_count):
        """Generate prior bounds for models with different parameter counts using MARIA data."""
        if not self.maria_bounds:
            # Return default bounds based on parameter count
            if param_count == 5:
                return [
                    [1.0, 1000.0],   # L1 thickness
                    [0.0, 60.0],     # ambient/L1 roughness  
                    [0.0, 60.0],     # L1/substrate roughness
                    [-8.0, 16.0],    # L1 SLD
                    [-8.0, 16.0]     # substrate SLD
                ]
            elif param_count == 8:
                return [
                    [1.0, 500.0],    # L1 thickness
                    [1.0, 500.0],    # L2 thickness
                    [0.0, 60.0],     # ambient/L1 roughness
                    [0.0, 60.0],     # L1/L2 roughness
                    [0.0, 60.0],     # L2/substrate roughness
                    [-8.0, 16.0],    # L1 SLD
                    [-8.0, 16.0],    # L2 SLD
                    [-8.0, 16.0]     # substrate SLD
                ]
            elif param_count == 7:
                # 1-layer model with nuisance parameters: [thickness, amb_rough, sub_rough, layer_sld, sub_sld, r_scale, log10_background]
                return [
                    [1.0, 1000.0],   # L1 thickness
                    [0.0, 60.0],     # ambient/L1 roughness  
                    [0.0, 60.0],     # L1/substrate roughness
                    [-8.0, 16.0],    # L1 SLD
                    [-8.0, 16.0],    # substrate SLD
                    [0.9, 1.1],      # r_scale
                    [-10.0, -4.0]    # log10_background
                ]
            elif param_count == 9:
                # 2-layer model with nuisance parameters: [L1_thick, L2_thick, amb_rough, L1L2_rough, L2sub_rough, L1_sld, L2_sld, sub_sld, r_scale, log10_background] 
                return [
                    [1.0, 500.0],    # L1 thickness
                    [1.0, 500.0],    # L2 thickness
                    [0.0, 60.0],     # ambient/L1 roughness
                    [0.0, 60.0],     # L1/L2 roughness
                    [0.0, 60.0],     # L2/substrate roughness
                    [-8.0, 16.0],    # L1 SLD
                    [-8.0, 16.0],    # L2 SLD
                    [-8.0, 16.0],    # substrate SLD
                    [0.9, 1.1]       # r_scale
                ]
            else:
                # Generic bounds for unknown parameter counts
                return [[0.0, 100.0] for _ in range(param_count)]
        
        # Use MARIA bounds
        key = f"{layer_count}_layers"
        if key not in self.maria_bounds:
            key = "2_layers"  # fallback to 2-layer bounds
        
        maria_data = self.maria_bounds[key]
        
        # Generate bounds based on parameter count and MARIA data
        if param_count == 5:
            # 1-layer model: [thickness, amb_rough, sub_rough, layer_sld, sub_sld]
            thickness_bounds = [
                maria_data["parameters"]["overall"]["thickness"]["min"],
                maria_data["parameters"]["overall"]["thickness"]["max"]
            ]
            roughness_bounds = [
                maria_data["parameters"]["overall"]["roughness"]["min"],
                maria_data["parameters"]["overall"]["roughness"]["max"]
            ]
            sld_bounds = [
                maria_data["parameters"]["overall"]["sld"]["min"] * 1e6,
                maria_data["parameters"]["overall"]["sld"]["max"] * 1e6
            ]
            
            return [
                thickness_bounds,    # L1 thickness
                roughness_bounds,    # ambient/L1 roughness
                roughness_bounds,    # L1/substrate roughness
                sld_bounds,          # L1 SLD
                sld_bounds           # substrate SLD
            ]
            
        elif param_count == 8:
            # 2-layer model: [L1_thick, L2_thick, amb_rough, L1L2_rough, L2sub_rough, L1_sld, L2_sld, sub_sld]
            if layer_count == 1:
                # For 1-layer data, split into two layers
                total_thickness = maria_data["parameters"]["overall"]["thickness"]["max"]
                thickness_min = maria_data["parameters"]["overall"]["thickness"]["min"]
                l1_bounds = [thickness_min * 0.3, total_thickness * 0.7]
                l2_bounds = [thickness_min * 0.3, total_thickness * 0.7]
                
                sld_bounds = [
                    maria_data["parameters"]["overall"]["sld"]["min"] * 1e6,
                    maria_data["parameters"]["overall"]["sld"]["max"] * 1e6
                ]
                l1_sld = l2_sld = sub_sld = sld_bounds
            else:
                # For 2-layer data, use layer-specific bounds
                l1_bounds = [
                    maria_data["parameters"]["by_position"]["layer_1"]["thickness"]["min"],
                    maria_data["parameters"]["by_position"]["layer_1"]["thickness"]["max"]
                ]
                l2_bounds = [
                    maria_data["parameters"]["by_position"]["layer_2"]["thickness"]["min"],
                    maria_data["parameters"]["by_position"]["layer_2"]["thickness"]["max"]
                ]
                
                l1_sld = [
                    maria_data["parameters"]["by_position"]["layer_1"]["sld"]["min"] * 1e6,
                    maria_data["parameters"]["by_position"]["layer_1"]["sld"]["max"] * 1e6
                ]
                l2_sld = [
                    maria_data["parameters"]["by_position"]["layer_2"]["sld"]["min"] * 1e6,
                    maria_data["parameters"]["by_position"]["layer_2"]["sld"]["max"] * 1e6
                ]
                sub_sld = [
                    maria_data["parameters"]["overall"]["sld"]["min"] * 1e6,
                    maria_data["parameters"]["overall"]["sld"]["max"] * 1e6
                ]
            
            roughness_bounds = [
                maria_data["parameters"]["overall"]["roughness"]["min"],
                maria_data["parameters"]["overall"]["roughness"]["max"]
            ]
            
            return [
                l1_bounds,           # L1 thickness
                l2_bounds,           # L2 thickness
                roughness_bounds,    # ambient/L1 roughness
                roughness_bounds,    # L1/L2 roughness
                roughness_bounds,    # L2/substrate roughness
                l1_sld,              # L1 SLD
                l2_sld,              # L2 SLD
                sub_sld              # substrate SLD
            ]
            
        elif param_count == 11:
            # 3-layer model: [L1_thick, L2_thick, L3_thick, amb_rough, L1L2_rough, L2L3_rough, L3sub_rough, L1_sld, L2_sld, L3_sld, sub_sld]
            if layer_count == 1:
                # For 1-layer data, split into three layers
                total_thickness = maria_data["parameters"]["overall"]["thickness"]["max"]
                thickness_min = maria_data["parameters"]["overall"]["thickness"]["min"]
                l1_bounds = [thickness_min * 0.2, total_thickness * 0.5]
                l2_bounds = [thickness_min * 0.2, total_thickness * 0.4]
                l3_bounds = [thickness_min * 0.1, total_thickness * 0.3]
                
                sld_bounds = [
                    maria_data["parameters"]["overall"]["sld"]["min"] * 1e6,
                    maria_data["parameters"]["overall"]["sld"]["max"] * 1e6
                ]
                l1_sld = l2_sld = l3_sld = sub_sld = sld_bounds
            else:
                # For 2-layer data, split layer 2 into L2 and L3
                l1_bounds = [
                    maria_data["parameters"]["by_position"]["layer_1"]["thickness"]["min"],
                    maria_data["parameters"]["by_position"]["layer_1"]["thickness"]["max"]
                ]
                l2_total_max = maria_data["parameters"]["by_position"]["layer_2"]["thickness"]["max"]
                l2_total_min = maria_data["parameters"]["by_position"]["layer_2"]["thickness"]["min"]
                l2_bounds = [l2_total_min, l2_total_max * 0.6]
                l3_bounds = [l2_total_min, l2_total_max * 0.4]
                
                l1_sld = [
                    maria_data["parameters"]["by_position"]["layer_1"]["sld"]["min"] * 1e6,
                    maria_data["parameters"]["by_position"]["layer_1"]["sld"]["max"] * 1e6
                ]
                l2_sld = [
                    maria_data["parameters"]["by_position"]["layer_2"]["sld"]["min"] * 1e6,
                    maria_data["parameters"]["by_position"]["layer_2"]["sld"]["max"] * 1e6
                ]
                l3_sld = l2_sld  # Same as L2
                sub_sld = [
                    maria_data["parameters"]["overall"]["sld"]["min"] * 1e6,
                    maria_data["parameters"]["overall"]["sld"]["max"] * 1e6
                ]
            
            roughness_bounds = [
                maria_data["parameters"]["overall"]["roughness"]["min"],
                maria_data["parameters"]["overall"]["roughness"]["max"]
            ]
            
            return [
                l1_bounds,           # L1 thickness
                l2_bounds,           # L2 thickness
                l3_bounds,           # L3 thickness
                roughness_bounds,    # ambient/L1 roughness
                roughness_bounds,    # L1/L2 roughness
                roughness_bounds,    # L2/L3 roughness
                roughness_bounds,    # L3/substrate roughness
                l1_sld,              # L1 SLD
                l2_sld,              # L2 SLD
                l3_sld,              # L3 SLD
                sub_sld              # substrate SLD
            ]
        
        else:
            # For unknown parameter counts, generate generic bounds
            thickness_bounds = [
                maria_data["parameters"]["overall"]["thickness"]["min"],
                maria_data["parameters"]["overall"]["thickness"]["max"]
            ]
            return [thickness_bounds for _ in range(param_count)]
    
    def get_bounds_for_param_count_absorption(self, layer_count):
        """Generate prior bounds for absorption model (11 parameters) using MARIA data."""
        if not self.maria_bounds:
            # Default bounds for absorption model (11 parameters: real SLDs + imaginary SLDs)
            return [
                [1.0, 500.0],    # L1 thickness  
                [1.0, 500.0],    # L2 thickness
                [0.0, 60.0],     # ambient/L1 roughness
                [0.0, 60.0],     # L1/L2 roughness
                [0.0, 60.0],     # L2/substrate roughness
                [0.0, 150.0],    # L1 real SLD (from config: slds: [0., 150.])
                [0.0, 150.0],    # L2 real SLD
                [0.0, 150.0],    # substrate real SLD  
                [0.0, 30.0],     # L1 imaginary SLD (from config: islds: [0., 30.])
                [0.0, 30.0],     # L2 imaginary SLD
                [0.0, 30.0]      # substrate imaginary SLD
            ]
        
        # Use MARIA bounds adapted for absorption model
        key = f"{layer_count}_layers"
        if key not in self.maria_bounds:
            key = "2_layers"  # fallback to 2-layer bounds
        
        maria_data = self.maria_bounds[key]
        
        # Thickness bounds
        if layer_count == 1:
            # For 1-layer data, split into two layers
            total_thickness = maria_data["parameters"]["overall"]["thickness"]["max"]
            thickness_min = maria_data["parameters"]["overall"]["thickness"]["min"]
            l1_bounds = [thickness_min * 0.3, total_thickness * 0.7]
            l2_bounds = [thickness_min * 0.3, total_thickness * 0.7]
        else:
            # For 2-layer data, use layer-specific bounds
            l1_bounds = [
                maria_data["parameters"]["by_position"]["layer_1"]["thickness"]["min"],
                maria_data["parameters"]["by_position"]["layer_1"]["thickness"]["max"]
            ]
            l2_bounds = [
                maria_data["parameters"]["by_position"]["layer_2"]["thickness"]["min"],
                maria_data["parameters"]["by_position"]["layer_2"]["thickness"]["max"]
            ]
        
        # Roughness bounds
        roughness_bounds = [
            maria_data["parameters"]["overall"]["roughness"]["min"],
            maria_data["parameters"]["overall"]["roughness"]["max"]
        ]
        
        # Real SLD bounds (scale to 0-150 range for absorption model)
        if layer_count == 1:
            sld_bounds = [
                0.0,  # Absorption model uses 0-150 range
                150.0
            ]
            l1_sld = l2_sld = sub_sld = sld_bounds
        else:
            # Scale MARIA SLD bounds to absorption model range (0-150)
            maria_sld_range = (maria_data["parameters"]["overall"]["sld"]["max"] - 
                             maria_data["parameters"]["overall"]["sld"]["min"]) * 1e6
            
            l1_sld = [0.0, min(150.0, maria_sld_range)]
            l2_sld = [0.0, min(150.0, maria_sld_range)]
            sub_sld = [0.0, min(150.0, maria_sld_range)]
        
        # Imaginary SLD bounds (0-30 range from config)
        isld_bounds = [0.0, 30.0]
        
        return [
            l1_bounds,           # L1 thickness
            l2_bounds,           # L2 thickness
            roughness_bounds,    # ambient/L1 roughness
            roughness_bounds,    # L1/L2 roughness
            roughness_bounds,    # L2/substrate roughness
            l1_sld,              # L1 real SLD
            l2_sld,              # L2 real SLD
            sub_sld,             # substrate real SLD
            isld_bounds,         # L1 imaginary SLD
            isld_bounds,         # L2 imaginary SLD
            isld_bounds          # substrate imaginary SLD
        ]
    
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simple Batch Inference Pipeline")
    parser.add_argument("--num-experiments", type=int, default=25,
                        help="Number of experiments to process (default: 25)")
    parser.add_argument("--layer-count", type=int, default=2,
                        help="Layer count to process (1 or 2, default: 2)")
    parser.add_argument("--data-directory", type=str, default="data",
                        help="Data directory (default: data)")
    
    return parser.parse_args()


def main():
    """Main function to run the batch inference pipeline."""
    args = parse_arguments()
    
    print(f"Processing {args.layer_count}-layer experiments from MARIA_VIPR_dataset/{args.layer_count}/")
    
    # Run batch inference pipeline
    batch_pipeline = SimpleBatchInferencePipeline(
        num_experiments=args.num_experiments,
        layer_count=args.layer_count,
        data_directory=args.data_directory
    )
    
    batch_pipeline.run()


if __name__ == "__main__":
    main()
