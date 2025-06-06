#!/usr/bin/env python3
"""
Inference Pipeline for Testing Multiple ReflecTorch Models

This script tests different trained models on experimental neutron reflectometry data
and compares their performance and parameter predictions. The pipeline is configurable
through JSON configuration files that specify data paths, formats, and model parameters.

Usage:
    python inference_pipeline.py [config_file]

Configuration files:
    - configs/membrane_config.json: For membrane analysis with dQ/Q = 0.1
    - configs/s000000_config.json: For s000000 data analysis
    
The pipeline handles both 3-column (Q, R, dR) and 4-column (Q, R, dR, dQ) data formats
automatically based on the configuration.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import json
from datetime import datetime
import warnings
import sys
warnings.filterwarnings('ignore')

from reflectorch import EasyInferenceModel

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class InferencePipeline:
    """Pipeline for testing multiple models on experimental data."""
    
    def __init__(self, config_file, output_dir="inference_results"):
        """
        Initialize the inference pipeline.
        
        Args:
            config_file: Path to JSON configuration file
            output_dir: Directory to save results
        """
        self.config_file = Path(config_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load configuration
        self.load_configuration()
        
        # Load experimental data
        self.load_experimental_data()
        
        # Initialize results storage
        self.results = {}
        
    def load_configuration(self):
        """Load configuration from JSON file."""
        print(f"Loading configuration from: {self.config_file}")
        
        if not self.config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)
        
        # Extract data and model configurations
        self.data_config = self.config['data_config']
        self.model_configs = self.config['model_configurations']
        
        print(f"Loaded configuration for: {self.data_config['description']}")
        print(f"Data format: {self.data_config['data_format']}")
        print(f"Available models: {len(self.model_configs)}")
        
    def load_experimental_data(self):
        """Load experimental data from file based on configuration."""
        data_path = Path(self.data_config['data_path'])
        print(f"Loading experimental data from: {data_path}")
        
        # Load data
        data = np.loadtxt(data_path, skiprows=1)
        
        # Handle data format based on configuration
        if self.data_config['data_format'] == '3_column':
            # 3-column format: Q, R, dR
            print("Processing 3-column format (Q, R, dR)")
            self.q_exp = data[:, 0]
            self.curve_exp = data[:, 1]
            self.sigmas_exp = data[:, 2]
            
            # Calculate Q-resolution using configured dQ/Q ratio
            dq_over_q = self.data_config['dq_over_q']
            if dq_over_q is None:
                raise ValueError("dq_over_q must be specified for 3-column data format")
            
            self.q_res_exp = self.q_exp * dq_over_q
            print(f"Calculated Q-resolution using dQ/Q = {dq_over_q}")
            
        elif self.data_config['data_format'] == '4_column':
            # 4-column format: Q, R, dR, dQ
            print("Processing 4-column format (Q, R, dR, dQ)")
            
            # Apply max_points limit if specified
            max_points = self.data_config.get('max_points')
            if max_points is not None:
                max_points = min(max_points, len(data))
                data = data[:max_points]
                print(f"Trimmed data to {max_points} points")
            
            self.q_exp = data[:, 0]
            self.curve_exp = data[:, 1]
            self.sigmas_exp = data[:, 2]
            self.q_res_exp = data[:, 3]
        else:
            raise ValueError(f"Unsupported data format: {self.data_config['data_format']}")
        
        print(f"Loaded {len(self.q_exp)} data points")
        print(f"Q range: {self.q_exp.min():.4f} - {self.q_exp.max():.4f} Å⁻¹")
        print(f"dQ range: {self.q_res_exp.min():.6f} - {self.q_res_exp.max():.6f} Å⁻¹")
        
    def run_inference(self, model_config, model_name):
        """
        Run inference with a specific model configuration.
        
        Args:
            model_config: Dictionary containing model configuration
            model_name: Name identifier for the model
            
        Returns:
            Dictionary containing inference results
        """
        print(f"\n{'='*60}")
        print(f"Testing model: {model_name}")
        print(f"Description: {model_config['description']}")
        print(f"{'='*60}")
        
        try:
            # Initialize model
            inference_model = EasyInferenceModel(
                config_name=model_config['config_name'],
                device='cpu',
                weights_format=model_config.get('weights_format', 'pth')
            )
            
            # Interpolate data to model grid
            q_model, exp_curve_interp = inference_model.interpolate_data_to_model_q(
                self.q_exp, self.curve_exp
            )
            
            # Interpolate resolution data
            q_res_interp = np.interp(q_model, self.q_exp, self.q_res_exp)
            
            print(f"Model Q grid: {len(q_model)} points, range: {q_model.min():.4f} - {q_model.max():.4f}")
            
            # Convert prior bounds from list format to tuple format
            prior_bounds = [tuple(bound) for bound in model_config['prior_bounds']]
            
            # Run prediction
            prediction_dict = inference_model.predict(
                reflectivity_curve=exp_curve_interp,
                prior_bounds=prior_bounds,
                q_values=q_model,
                q_resolution=q_res_interp,
                clip_prediction=True,
                polish_prediction=True,
                calc_pred_curve=True,
                calc_pred_sld_profile=True,
                calc_polished_sld_profile=True,
            )
            
            # Extract results
            result = {
                'model_name': model_name,
                'config_name': model_config['config_name'],
                'description': model_config['description'],
                'q_model': q_model,
                'predicted_params': prediction_dict['predicted_params_array'],
                'polished_params': prediction_dict['polished_params_array'],
                'param_names': prediction_dict['param_names'],
                'predicted_curve': prediction_dict['predicted_curve'],
                'polished_curve': prediction_dict['polished_curve'],
                'sld_profile_x': prediction_dict['predicted_sld_xaxis'],
                'sld_profile_predicted': prediction_dict['predicted_sld_profile'],
                'sld_profile_polished': prediction_dict['sld_profile_polished'],
                'success': True,
                'error': None
            }
            
            # Print parameter results using parameter names from config
            print(f"\nParameter Results for {model_name}:")
            print("-" * 50)
            param_names = model_config.get('parameter_names', prediction_dict["param_names"])
            for param_name, pred_val, polish_val in zip(
                param_names, 
                prediction_dict['predicted_params_array'],
                prediction_dict["polished_params_array"]
            ):
                print(f'{param_name.ljust(25)} -> Predicted: {pred_val:8.2f}   Polished: {polish_val:8.2f}')
            
            # Calculate fit quality metrics
            result['fit_metrics'] = self.calculate_fit_metrics(
                self.curve_exp, prediction_dict['polished_curve'], 
                self.sigmas_exp, q_model
            )
            
            print(f"\nFit Quality Metrics:")
            print(f"R-squared: {result['fit_metrics']['r_squared']:.4f}")
            print(f"MSE: {result['fit_metrics']['mse']:.6f}")
            print(f"L1 Loss: {result['fit_metrics']['l1_loss']:.6f}")
            
            return result
            
        except Exception as e:
            print(f"ERROR with model {model_name}: {str(e)}")
            return {
                'model_name': model_name,
                'success': False,
                'error': str(e)
            }
    
    def calculate_fit_metrics(self, y_exp, y_pred, sigma_exp, q_model):
        """Calculate fit quality metrics including MSE, R-squared, and L1 loss."""
        # Interpolate predicted curve to experimental Q points for comparison
        y_pred_interp = np.interp(self.q_exp, q_model, y_pred)
        
        # R-squared
        ss_res = np.sum((y_exp - y_pred_interp) ** 2)
        ss_tot = np.sum((y_exp - np.mean(y_exp)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # MSE (Mean Squared Error)
        mse = np.mean((y_exp - y_pred_interp) ** 2)
        
        # L1 Loss (Mean Absolute Error)
        l1_loss = np.mean(np.abs(y_exp - y_pred_interp))
        
        return {
            'r_squared': r_squared,
            'mse': mse,
            'l1_loss': l1_loss
        }
    
    def run_all_models(self):
        """Run inference on all models defined in the configuration."""
        print(f"Starting inference pipeline with {len(self.model_configs)} models...")
        print(f"Results will be saved to: {self.output_dir}")
        
        for model_name, model_config in self.model_configs.items():
            result = self.run_inference(model_config, model_name)
            self.results[model_name] = result
        
        # Save results
        self.save_results()
        
        # Generate comparison plots
        self.create_comparison_plots()
        
        # Print summary
        self.print_summary()
    
    def save_results(self):
        """Save results to JSON file."""
        # Prepare serializable results
        serializable_results = {}
        
        for model_name, result in self.results.items():
            if result['success']:
                serializable_results[model_name] = {
                    'model_name': result['model_name'],
                    'config_name': result['config_name'],
                    'description': result['description'],
                    'predicted_params': result['predicted_params'].tolist(),
                    'polished_params': result['polished_params'].tolist(),
                    'param_names': result['param_names'],
                    'fit_metrics': result['fit_metrics'],
                    'success': result['success']
                }
            else:
                serializable_results[model_name] = {
                    'model_name': result['model_name'],
                    'success': result['success'],
                    'error': result['error']
                }
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"inference_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
    
    def create_comparison_plots(self):
        """Create comparison plots for all successful models."""
        successful_models = {k: v for k, v in self.results.items() if v['success']}
        
        if not successful_models:
            print("No successful models to plot")
            return
        
        # Create reflectivity comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot experimental data
        ax1.errorbar(self.q_exp, self.curve_exp, yerr=self.sigmas_exp, 
                    xerr=self.q_res_exp, fmt='o', markersize=2, alpha=0.7,
                    color='black', label='Experimental', zorder=1)
        
        ax1.set_yscale('log')
        ax1.set_xlabel('q [Å⁻¹]', fontsize=12)
        ax1.set_ylabel('R(q)', fontsize=12)
        ax1.set_title('Reflectivity Curves Comparison', fontsize=14)
        
        # Plot model predictions with metrics in labels
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        for i, (model_name, result) in enumerate(successful_models.items()):
            color = colors[i % len(colors)]
            metrics = result['fit_metrics']
            label = f"{model_name} (R²={metrics['r_squared']:.3f}, MSE={metrics['mse']:.4f})"
            ax1.plot(result['q_model'], result['polished_curve'], 
                    color=color, linewidth=2, label=label, 
                    alpha=0.8, zorder=2+i)
        
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot SLD profiles
        for i, (model_name, result) in enumerate(successful_models.items()):
            color = colors[i % len(colors)]
            ax2.plot(result['sld_profile_x'], result['sld_profile_polished'], 
                    color=color, linewidth=2, label=f"{model_name}", alpha=0.8)
        
        ax2.set_xlabel('z [Å]', fontsize=12)
        ax2.set_ylabel('SLD [10⁻⁶ Å⁻²]', fontsize=12)
        ax2.set_title('SLD Profiles Comparison', fontsize=14)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Add metrics summary text box on the plot
        if successful_models:
            # Sort models by MSE for the summary
            sorted_models = sorted(
                successful_models.items(), 
                key=lambda x: x[1]['fit_metrics']['mse']
            )
            
            # Create metrics summary text
            summary_text = "Best Models (by MSE):\n"
            for i, (model_name, result) in enumerate(sorted_models[:3]):  # Top 3
                metrics = result['fit_metrics']
                summary_text += f"{i+1}. {model_name[:15]}...\n"
                summary_text += f"   R²={metrics['r_squared']:.3f}, MSE={metrics['mse']:.4f}\n"
            
            # Add text box to the reflectivity plot
            ax1.text(0.02, 0.02, summary_text, transform=ax1.transAxes, 
                    fontsize=8, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.output_dir / f"model_comparison_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Comparison plot saved to: {plot_file}")
    
    def print_summary(self):
        """Print summary of all model results."""
        print(f"\n{'='*80}")
        print("INFERENCE PIPELINE SUMMARY")
        print(f"{'='*80}")
        
        successful_models = {k: v for k, v in self.results.items() if v['success']}
        failed_models = {k: v for k, v in self.results.items() if not v['success']}
        
        print(f"Total models tested: {len(self.results)}")
        print(f"Successful: {len(successful_models)}")
        print(f"Failed: {len(failed_models)}")
        
        if failed_models:
            print(f"\nFailed models:")
            for model_name, result in failed_models.items():
                print(f"  - {model_name}: {result['error']}")
        
        if successful_models:
            print(f"\nFit Quality Comparison:")
            print("-" * 80)
            print(f"{'Model':<20} {'R²':<10} {'MSE':<12} {'L1 Loss':<12}")
            print("-" * 80)
            
            # Sort by MSE (lower is better)
            sorted_models = sorted(
                successful_models.items(), 
                key=lambda x: x[1]['fit_metrics']['mse']
            )
            
            for model_name, result in sorted_models:
                metrics = result['fit_metrics']
                print(f"{model_name:<20} {metrics['r_squared']:<10.4f} "
                      f"{metrics['mse']:<12.6f} {metrics['l1_loss']:<12.6f}")
            
            print("\nBest performing model (lowest MSE):")
            best_model_name, best_result = sorted_models[0]
            print(f"  {best_model_name}: {best_result['description']}")
            print(f"  MSE = {best_result['fit_metrics']['mse']:.6f}")


def main():
    """Main execution function."""
    # Parse command line arguments
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        # Default to membrane configuration
        config_file = "configs/membrane_config.json"
    
    print("ReflecTorch Multi-Model Inference Pipeline")
    print(f"Using configuration: {config_file}")
    print("=" * 50)
    
    try:
        # Initialize and run pipeline
        pipeline = InferencePipeline(config_file)
        pipeline.run_all_models()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nAvailable configurations:")
        print("  - configs/membrane_config.json (membrane analysis with dQ/Q=0.1)")
        print("  - configs/s000000_config.json (s000000 data analysis)")
        print("  - configs/s000004_config.json (s000004 data analysis - single layer)")
        print("\nUsage: python inference_pipeline.py [config_file]")
        sys.exit(1)
    except Exception as e:
        print(f"Error running pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
