#!/usr/bin/env python3
"""
Inference Pipeline for Testing Multiple ReflecTorch Models

This script tests different trained models on the s000000 experimental data
and compares their performance and parameter predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from reflectorch import EasyInferenceModel

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class InferencePipeline:
    """Pipeline for testing multiple models on experimental data."""
    
    def __init__(self, data_path, output_dir="inference_results"):
        """
        Initialize the inference pipeline.
        
        Args:
            data_path: Path to experimental data file
            output_dir: Directory to save results
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load experimental data
        self.load_experimental_data()
        
        # Initialize results storage
        self.results = {}
        
    def load_experimental_data(self):
        """Load experimental data from file."""
        print(f"Loading experimental data from: {self.data_path}")
        
        # Load data (assuming 4-column format: q, R, dR, dQ)
        data = np.loadtxt(self.data_path, skiprows=1)
        
        # Trim data to remove high error regions
        max_points = min(2050, len(data))
        
        self.q_exp = data[:max_points, 0]
        self.curve_exp = data[:max_points, 1]
        self.sigmas_exp = data[:max_points, 2]
        self.q_res_exp = data[:max_points, 3]
        
        print(f"Loaded {len(self.q_exp)} data points")
        print(f"Q range: {self.q_exp.min():.4f} - {self.q_exp.max():.4f} Å⁻¹")
        
    def define_model_configurations(self):
        """Define the models and their configurations to test."""
        
        models = {
            # Add new models here following the format:
            # 'model_name': {
            #     'config_name': 'config_file_name',
            #     'weights_format': 'pt' or 'safetensors',
            #     'prior_bounds': [(min, max), ...],  # parameter bounds
            #     'description': 'model description'
            # }
            
            'neutron_L3_comp': {
                'config_name': 'b_mc_point_neutron_conv_standard_L3_comp',
                'weights_format': 'safetensors',
                'prior_bounds': [
                    # 3-layer model: broader bounds
                    (200.0, 300.0),   # L1 thickness
                    (850.0, 1100.0),  # L2 thickness  
                    (50.0, 200.0),    # L3 thickness
                    (1.0, 30.0),      # ambient/L1 roughness
                    (5.0, 50.0),      # L1/L2 roughness
                    (10.0, 80.0),     # L2/L3 roughness
                    (30.0, 100.0),    # L3/substrate roughness
                    (8.0, 15.0),      # L1 SLD
                    (7.0, 12.0),      # L2 SLD
                    (5.0, 10.0),      # L3 SLD
                    (4.0, 8.0)        # substrate SLD
                ],
                'description': 'Pre-trained 3-layer neutron model'
            },
            
            'xray_mc25': {
                'config_name': 'mc25',
                'weights_format': 'safetensors',
                'prior_bounds': [
                    # 2-layer X-ray model: adapted for s000000 data structure
                    (200.0, 400.0),   # L1 thickness
                    (800.0, 1200.0),  # L2 thickness
                    (1.0, 30.0),      # ambient/L1 roughness
                    (5.0, 50.0),      # L1/L2 roughness
                    (30.0, 100.0),    # L2/substrate roughness
                    (6.0, 15.0),      # L1 SLD (X-ray)
                    (6.0, 12.0),      # L2 SLD (X-ray)
                    (4.0, 8.0)        # substrate SLD (X-ray)
                ],
                'description': 'Pre-trained X-ray model (mc25)'
            },
            
            # L3 models with different input configurations
            'neutron_L3_InputDq': {
                'config_name': 'b_mc_point_neutron_conv_standard_L3_InputDq',
                'weights_format': 'safetensors',
                'prior_bounds': [
                    # 3-layer model with InputDq
                    (200.0, 300.0),   # L1 thickness
                    (850.0, 1100.0),  # L2 thickness  
                    (50.0, 200.0),    # L3 thickness
                    (1.0, 30.0),      # ambient/L1 roughness
                    (5.0, 50.0),      # L1/L2 roughness
                    (10.0, 80.0),     # L2/L3 roughness
                    (30.0, 100.0),    # L3/substrate roughness
                    (8.0, 15.0),      # L1 SLD
                    (7.0, 12.0),      # L2 SLD
                    (5.0, 10.0),      # L3 SLD
                    (4.0, 8.0)        # substrate SLD
                ],
                'description': 'Pre-trained 3-layer neutron model with InputDq'
            },
            
            'neutron_L3_InputQDq': {
                'config_name': 'b_mc_point_neutron_conv_standard_L3_InputQDq',
                'weights_format': 'safetensors',
                'prior_bounds': [
                    # 3-layer model with InputQDq
                    (200.0, 300.0),   # L1 thickness
                    (850.0, 1100.0),  # L2 thickness  
                    (50.0, 200.0),    # L3 thickness
                    (1.0, 30.0),      # ambient/L1 roughness
                    (5.0, 50.0),      # L1/L2 roughness
                    (10.0, 80.0),     # L2/L3 roughness
                    (30.0, 100.0),    # L3/substrate roughness
                    (8.0, 15.0),      # L1 SLD
                    (7.0, 12.0),      # L2 SLD
                    (5.0, 10.0),      # L3 SLD
                    (4.0, 8.0)        # substrate SLD
                ],
                'description': 'Pre-trained 3-layer neutron model with InputQDq'
            },
            
            # L2 models with different input configurations
            'neutron_L2_InputDq': {
                'config_name': 'b_mc_point_neutron_conv_standard_L2_InputDq',
                'weights_format': 'safetensors',
                'prior_bounds': [
                    # 2-layer model with InputDq
                    (200.0, 400.0),   # L1 thickness
                    (800.0, 1200.0),  # L2 thickness
                    (1.0, 30.0),      # ambient/L1 roughness
                    (5.0, 50.0),      # L1/L2 roughness
                    (30.0, 100.0),    # L2/substrate roughness
                    (8.0, 15.0),      # L1 SLD
                    (7.0, 12.0),      # L2 SLD
                    (4.0, 8.0)        # substrate SLD
                ],
                'description': 'Pre-trained 2-layer neutron model with InputDq'
            },
            
            'neutron_L2_InputQDq': {
                'config_name': 'b_mc_point_neutron_conv_standard_L2_InputQDq',
                'weights_format': 'safetensors',
                'prior_bounds': [
                    # 2-layer model with InputQDq
                    (200.0, 400.0),   # L1 thickness
                    (800.0, 1200.0),  # L2 thickness
                    (1.0, 30.0),      # ambient/L1 roughness
                    (5.0, 50.0),      # L1/L2 roughness
                    (30.0, 100.0),    # L2/substrate roughness
                    (8.0, 15.0),      # L1 SLD
                    (7.0, 12.0),      # L2 SLD
                    (4.0, 8.0)        # substrate SLD
                ],
                'description': 'Pre-trained 2-layer neutron model with InputQDq'
            },
            
            # L1 models with different input configurations
            'neutron_L1_InputDq': {
                'config_name': 'b_mc_point_neutron_conv_standard_L1_InputDq',
                'weights_format': 'safetensors',
                'prior_bounds': [
                    # 1-layer model with InputDq
                    (200.0, 1200.0),  # L1 thickness
                    (1.0, 30.0),      # ambient/L1 roughness
                    (30.0, 100.0),    # L1/substrate roughness
                    (8.0, 15.0),      # L1 SLD
                    (4.0, 8.0)        # substrate SLD
                ],
                'description': 'Pre-trained 1-layer neutron model with InputDq'
            },
            
            'neutron_L1_InputQDq': {
                'config_name': 'b_mc_point_neutron_conv_standard_L1_InputQDq',
                'weights_format': 'safetensors',
                'prior_bounds': [
                    # 1-layer model with InputQDq
                    (200.0, 1200.0),  # L1 thickness
                    (1.0, 30.0),      # ambient/L1 roughness
                    (30.0, 100.0),    # L1/substrate roughness
                    (8.0, 15.0),      # L1 SLD
                    (4.0, 8.0)        # substrate SLD
                ],
                'description': 'Pre-trained 1-layer neutron model with InputQDq'
            }
        }
        
        return models
    
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
            
            # Run prediction
            prediction_dict = inference_model.predict(
                reflectivity_curve=exp_curve_interp,
                prior_bounds=model_config['prior_bounds'],
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
            
            # Print parameter results
            print(f"\nParameter Results for {model_name}:")
            print("-" * 50)
            for param_name, pred_val, polish_val in zip(
                prediction_dict["param_names"], 
                prediction_dict['predicted_params_array'],
                prediction_dict["polished_params_array"]
            ):
                print(f'{param_name.ljust(16)} -> Predicted: {pred_val:8.2f}   Polished: {polish_val:8.2f}')
            
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
        """Run inference on all defined models."""
        models = self.define_model_configurations()
        
        print(f"Starting inference pipeline with {len(models)} models...")
        print(f"Results will be saved to: {self.output_dir}")
        
        for model_name, model_config in models.items():
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
    # Define paths
    data_path = "data/s000000_experimental_curve.dat"
    
    print("ReflecTorch Multi-Model Inference Pipeline")
    print("=" * 50)
    
    # Initialize and run pipeline
    pipeline = InferencePipeline(data_path)
    pipeline.run_all_models()


if __name__ == "__main__":
    main()
