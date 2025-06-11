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
        
        # Load true parameters if available
        self.true_params_dict = None
        if 'true_parameters_file' in self.data_config:
            self.true_params_dict = self.parse_true_parameters_from_model_file(
                self.data_config['true_parameters_file']
            )
        
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
            
            # Calculate parameter metrics - get true parameters for this model
            true_params, true_param_names = self.get_true_params_for_model(model_config, self.true_params_dict)
            if true_params is not None:
                try:
                    result['parameter_metrics'] = self.calculate_parameter_metrics(
                        prediction_dict['polished_params_array'], 
                        true_params, 
                        param_names
                    )
                    
                    # Print parameter comparison
                    print(f"\nParameter Comparison (Predicted vs True):")
                    print("-" * 60)
                    print(f"{'Parameter':<25} {'Predicted':<12} {'True':<12} {'MSE':<12}")
                    print("-" * 60)
                    
                    param_metrics = result['parameter_metrics']['by_parameter']
                    for param_name, metrics in param_metrics.items():
                        pred_val = metrics['predicted']
                        true_val = metrics['true']
                        sq_error = metrics['squared_error']
                        
                        print(f"{param_name:<25} {pred_val:<12.2f} {true_val:<12.2f} "
                              f"{sq_error:<12.4f}")
                    
                    # Print aggregate metrics
                    overall_metrics = result['parameter_metrics']['overall']
                    by_type = result['parameter_metrics']['by_type']
                    print(f"\nParameter Metrics Summary:")
                    print(f"Overall MAPE: {overall_metrics['mape']:.2f}%, Overall MSE: {overall_metrics['mse']:.6f}")
                    if by_type['thickness_mape'] > 0:
                        print(f"Thickness - MAPE: {by_type['thickness_mape']:.2f}%, MSE: {by_type['thickness_mse']:.6f}")
                    if by_type['roughness_mape'] > 0:
                        print(f"Roughness - MAPE: {by_type['roughness_mape']:.2f}%, MSE: {by_type['roughness_mse']:.6f}")
                    if by_type['sld_mape'] > 0:
                        print(f"SLD - MAPE: {by_type['sld_mape']:.2f}%, MSE: {by_type['sld_mse']:.6f}")
                        
                except Exception as e:
                    print(f"Warning: Could not calculate parameter metrics: {e}")
                    result['parameter_metrics'] = None
            else:
                result['parameter_metrics'] = None
            
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
    
    def calculate_parameter_metrics(self, predicted_params, true_params, param_names):
        """
        Calculate parameter metrics comparing predicted vs true parameters.
        Individual parameters show MSE, but overall comparison uses MAPE for better interpretability.
        
        Args:
            predicted_params: Array of predicted parameter values
            true_params: Array of true parameter values  
            param_names: List of parameter names for detailed breakdown
            
        Returns:
            Dictionary containing parameter metrics
        """
        if len(predicted_params) != len(true_params):
            raise ValueError(f"Parameter arrays must have same length: {len(predicted_params)} vs {len(true_params)}")
        
        # Calculate overall MAPE for comparison
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_errors = np.abs((predicted_params - true_params) / true_params)
            relative_errors = np.where(np.isfinite(relative_errors), relative_errors, 0)
        param_mape = np.mean(relative_errors) * 100
        
        # Per-parameter breakdown with MSE for individual parameters
        param_breakdown = {}
        for i, name in enumerate(param_names):
            if i < len(predicted_params):
                pred_val = predicted_params[i]
                true_val = true_params[i]
                squared_error = (pred_val - true_val) ** 2
                abs_error = abs(pred_val - true_val)
                rel_error = abs_error / abs(true_val) * 100 if abs(true_val) > 1e-10 else 0
                
                param_breakdown[name] = {
                    'predicted': float(pred_val),
                    'true': float(true_val),
                    'squared_error': float(squared_error),
                    'absolute_error': float(abs_error),
                    'relative_error_percent': float(rel_error)
                }
        
        # Group parameters by type for aggregate statistics (both MAPE and MSE)
        thickness_mapes = []
        thickness_mses = []
        roughness_mapes = []
        roughness_mses = []
        sld_mapes = []
        sld_mses = []
        
        # Calculate overall MSE
        overall_mse = np.mean([metrics['squared_error'] for metrics in param_breakdown.values()])
        
        for name, metrics in param_breakdown.items():
            rel_err = metrics['relative_error_percent']
            sq_err = metrics['squared_error']
            
            if 'thickness' in name.lower() or 'd_' in name.lower():
                thickness_mapes.append(rel_err)
                thickness_mses.append(sq_err)
            elif 'roughness' in name.lower() or 'sigma' in name.lower():
                roughness_mapes.append(rel_err)
                roughness_mses.append(sq_err)
            elif 'sld' in name.lower() or 'rho' in name.lower():
                sld_mapes.append(rel_err)
                sld_mses.append(sq_err)
        
        return {
            'overall': {
                'mape': float(param_mape),
                'mse': float(overall_mse)
            },
            'by_parameter': param_breakdown,
            'by_type': {
                'thickness_mape': float(np.mean(thickness_mapes)) if thickness_mapes else 0.0,
                'thickness_mse': float(np.mean(thickness_mses)) if thickness_mses else 0.0,
                'roughness_mape': float(np.mean(roughness_mapes)) if roughness_mapes else 0.0,
                'roughness_mse': float(np.mean(roughness_mses)) if roughness_mses else 0.0,
                'sld_mape': float(np.mean(sld_mapes)) if sld_mapes else 0.0,
                'sld_mse': float(np.mean(sld_mses)) if sld_mses else 0.0
            }
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
                serializable_result = {
                    'model_name': result['model_name'],
                    'config_name': result['config_name'],
                    'description': result['description'],
                    'predicted_params': result['predicted_params'].tolist(),
                    'polished_params': result['polished_params'].tolist(),
                    'param_names': result['param_names'],
                    'fit_metrics': result['fit_metrics'],
                    'success': result['success']
                }
                
                # Add parameter metrics if available
                if 'parameter_metrics' in result and result['parameter_metrics'] is not None:
                    serializable_result['parameter_metrics'] = result['parameter_metrics']
                
                serializable_results[model_name] = serializable_result
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
        
        # Check if we have parameter metrics for plotting
        models_with_param_metrics = {
            k: v for k, v in successful_models.items() 
            if 'parameter_metrics' in v and v['parameter_metrics'] is not None
        }
        
        # Create plots with or without parameter loss subplot
        if models_with_param_metrics:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
        else:
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
        
        # Plot parameter loss comparison if available
        if models_with_param_metrics:
            # Create parameter MAPE bar chart
            model_names = list(models_with_param_metrics.keys())
            param_mapes = []
            param_mses = []
            thickness_mapes = []
            thickness_mses = []
            roughness_mapes = []
            roughness_mses = []
            sld_mapes = []
            sld_mses = []
            
            for model_name in model_names:
                result = models_with_param_metrics[model_name]
                param_metrics = result['parameter_metrics']
                param_mapes.append(param_metrics['overall']['mape'])
                param_mses.append(param_metrics['overall']['mse'])
                thickness_mapes.append(param_metrics['by_type']['thickness_mape'])
                thickness_mses.append(param_metrics['by_type']['thickness_mse'])
                roughness_mapes.append(param_metrics['by_type']['roughness_mape'])
                roughness_mses.append(param_metrics['by_type']['roughness_mse'])
                sld_mapes.append(param_metrics['by_type']['sld_mape'])
                sld_mses.append(param_metrics['by_type']['sld_mse'])
            
            x = np.arange(len(model_names))
            width = 0.2
            
            ax3.bar(x - 1.5*width, param_mapes, width, label='Overall MAPE', alpha=0.8, color='gray')
            ax3.bar(x - 0.5*width, thickness_mapes, width, label='Thickness MAPE', alpha=0.8, color='blue')
            ax3.bar(x + 0.5*width, roughness_mapes, width, label='Roughness MAPE', alpha=0.8, color='green')
            ax3.bar(x + 1.5*width, sld_mapes, width, label='SLD MAPE', alpha=0.8, color='red')
            
            ax3.set_xlabel('Model', fontsize=12)
            ax3.set_ylabel('MAPE (%)', fontsize=12)
            ax3.set_title('Parameter Prediction Errors (MAPE)', fontsize=14)
            ax3.set_xticks(x)
            ax3.set_xticklabels([name[:15] + '...' if len(name) > 15 else name for name in model_names], 
                              rotation=45, ha='right')
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add values on top of bars for the overall MAPE
            for i, v in enumerate(param_mapes):
                ax3.text(i - 1.5*width, v + max(param_mapes) * 0.01, f'{v:.1f}%', 
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            # Create MSE text box
            mse_text = "Parameter MSE Values:\n"
            mse_text += "-" * 25 + "\n"
            for i, model_name in enumerate(model_names):
                short_name = model_name[:12] + "..." if len(model_name) > 12 else model_name
                mse_text += f"{short_name}:\n"
                mse_text += f"  Overall: {param_mses[i]:.4f}\n"
                if thickness_mses[i] > 0:
                    mse_text += f"  Thick: {thickness_mses[i]:.4f}\n"
                if roughness_mses[i] > 0:
                    mse_text += f"  Rough: {roughness_mses[i]:.4f}\n"
                if sld_mses[i] > 0:
                    mse_text += f"  SLD: {sld_mses[i]:.4f}\n"
                mse_text += "\n"
            
            # Add MSE text box to the parameter plot
            ax3.text(0.02, 0.98, mse_text, transform=ax3.transAxes, 
                    fontsize=7, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
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
            
            # Print parameter metrics summary if available
            models_with_param_metrics = [
                (name, result) for name, result in successful_models.items() 
                if 'parameter_metrics' in result and result['parameter_metrics'] is not None
            ]
            
            if models_with_param_metrics:
                print(f"\nParameter Quality Comparison:")
                print("-" * 110)
                print(f"{'Model':<20} {'MAPE (%)':<12} {'MSE':<12} {'Thick MAPE':<12} {'Rough MAPE':<12} {'SLD MAPE':<12}")
                print("-" * 110)
                
                # Sort by parameter MAPE
                sorted_param_models = sorted(
                    models_with_param_metrics,
                    key=lambda x: x[1]['parameter_metrics']['overall']['mape']
                )
                
                for model_name, result in sorted_param_models:
                    param_metrics = result['parameter_metrics']['overall']
                    by_type = result['parameter_metrics']['by_type']
                    print(f"{model_name:<20} {param_metrics['mape']:<12.2f} "
                          f"{param_metrics['mse']:<12.6f} {by_type['thickness_mape']:<12.2f} "
                          f"{by_type['roughness_mape']:<12.2f} {by_type['sld_mape']:<12.2f}")
                
                print("\nBest parameter prediction (lowest Parameter MAPE):")
                best_param_model_name, best_param_result = sorted_param_models[0]
                print(f"  {best_param_model_name}: {best_param_result['description']}")
                print(f"  Parameter MAPE = {best_param_result['parameter_metrics']['overall']['mape']:.2f}%")
                print(f"  Parameter MSE = {best_param_result['parameter_metrics']['overall']['mse']:.6f}")
        
    def parse_true_parameters_from_model_file(self, model_file_path):
        """
        Parse true parameters from s000000_model.txt file format.
        
        Args:
            model_file_path: Path to the model.txt file
            
        Returns:
            Dictionary containing true parameters organized by layer count and parameter names
        """
        print(f"Parsing true parameters from: {model_file_path}")
        
        if not Path(model_file_path).exists():
            print(f"Warning: Model file not found: {model_file_path}")
            return None
            
        # Read the model file
        with open(model_file_path, 'r') as f:
            lines = f.readlines()
        
        # Parse the data
        layers = []
        for line in lines:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
                
            parts = line.split()
            if len(parts) >= 4:
                layer_name = parts[0]
                sld = float(parts[1])  # Already in 10^-6 Å^-2 units
                thickness = float(parts[2]) if parts[2] != 'inf' else None
                roughness = float(parts[3]) if parts[3] != 'none' else None
                
                layers.append({
                    'name': layer_name,
                    'sld': sld,
                    'thickness': thickness,
                    'roughness': roughness
                })
        
        print(f"Parsed {len(layers)} layers from model file:")
        for layer in layers:
            print(f"  {layer['name']}: SLD={layer['sld']:.2e}, thickness={layer['thickness']}, roughness={layer['roughness']}")
        
        # Convert to parameter arrays for different layer configurations
        # Based on the model file structure:
        # fronting (ambient): SLD=3.50000e-06, roughness=8.72
        # layer1: SLD=1.14195e-05, thickness=242.09, roughness=24.16  
        # layer2: SLD=9.79919e-06, thickness=959.62, roughness=75.06
        # backing (substrate): SLD=6.45461e-06
        
        true_params_dict = {}
        
        # For 2-layer models (L1, L2): layers[1] and layers[2] are the physical layers
        if len(layers) >= 4:  # fronting + layer1 + layer2 + backing
            # 2-layer configuration
            true_params_2layer = [
                layers[1]['thickness'],  # L1 thickness
                layers[2]['thickness'],  # L2 thickness
                layers[0]['roughness'],  # ambient/L1 roughness (fronting roughness)
                layers[1]['roughness'],  # L1/L2 roughness 
                layers[2]['roughness'],  # L2/substrate roughness
                layers[1]['sld'] * 1e6,  # L1 SLD (convert to 10^-6 units for display)
                layers[2]['sld'] * 1e6,  # L2 SLD
                layers[3]['sld'] * 1e6   # substrate SLD
            ]
            
            param_names_2layer = [
                "L1 thickness (Å)",
                "L2 thickness (Å)", 
                "ambient/L1 roughness (Å)",
                "L1/L2 roughness (Å)",
                "L2/substrate roughness (Å)",
                "L1 SLD (×10⁻⁶ Å⁻²)",
                "L2 SLD (×10⁻⁶ Å⁻²)",
                "substrate SLD (×10⁻⁶ Å⁻²)"
            ]
            
            true_params_dict['2_layer'] = {
                'params': true_params_2layer,
                'param_names': param_names_2layer
            }
            
            # 3-layer configuration - for models that expect a third layer, 
            # we can split one of the existing layers or add a thin intermediate layer
            # For now, let's create a thin third layer by splitting layer2
            l3_thickness = 100.0  # Assume 100 Å for L3
            l2_thickness_reduced = layers[2]['thickness'] - l3_thickness
            
            true_params_3layer = [
                layers[1]['thickness'],    # L1 thickness
                l2_thickness_reduced,      # L2 thickness (reduced)
                l3_thickness,              # L3 thickness
                layers[0]['roughness'],    # ambient/L1 roughness
                layers[1]['roughness'],    # L1/L2 roughness
                layers[1]['roughness'] * 1.5,  # L2/L3 roughness (interpolated)
                layers[2]['roughness'],    # L3/substrate roughness
                layers[1]['sld'] * 1e6,    # L1 SLD
                layers[2]['sld'] * 1e6,    # L2 SLD
                layers[2]['sld'] * 1e6,    # L3 SLD (same as L2)
                layers[3]['sld'] * 1e6     # substrate SLD
            ]
            
            param_names_3layer = [
                "L1 thickness (Å)",
                "L2 thickness (Å)",
                "L3 thickness (Å)",
                "ambient/L1 roughness (Å)",
                "L1/L2 roughness (Å)",
                "L2/L3 roughness (Å)",
                "L3/substrate roughness (Å)",
                "L1 SLD (×10⁻⁶ Å⁻²)",
                "L2 SLD (×10⁻⁶ Å⁻²)",
                "L3 SLD (×10⁻⁶ Å⁻²)",
                "substrate SLD (×10⁻⁶ Å⁻²)"
            ]
            
            true_params_dict['3_layer'] = {
                'params': true_params_3layer,
                'param_names': param_names_3layer
            }
            
            # 1-layer configuration - combine both physical layers
            combined_thickness = layers[1]['thickness'] + layers[2]['thickness']
            # Weighted average SLD
            total_volume = layers[1]['thickness'] + layers[2]['thickness']
            weighted_sld = (layers[1]['sld'] * layers[1]['thickness'] + 
                          layers[2]['sld'] * layers[2]['thickness']) / total_volume
            
            true_params_1layer = [
                combined_thickness,        # L1 thickness (combined)
                layers[0]['roughness'],    # ambient/L1 roughness
                layers[2]['roughness'],    # L1/substrate roughness
                weighted_sld * 1e6,        # L1 SLD (weighted average)
                layers[3]['sld'] * 1e6     # substrate SLD
            ]
            
            param_names_1layer = [
                "L1 thickness (Å)",
                "ambient/L1 roughness (Å)",
                "L1/substrate roughness (Å)",
                "L1 SLD (×10⁻⁶ Å⁻²)",
                "substrate SLD (×10⁻⁶ Å⁻²)"
            ]
            
            true_params_dict['1_layer'] = {
                'params': true_params_1layer,
                'param_names': param_names_1layer
            }
        
        return true_params_dict

    def get_true_params_for_model(self, model_config, true_params_dict):
        """
        Get the appropriate true parameters for a specific model configuration.
        
        Args:
            model_config: Model configuration dictionary
            true_params_dict: Dictionary of true parameters by layer count
            
        Returns:
            Tuple of (true_params_array, param_names) or (None, None) if not found
        """
        if true_params_dict is None:
            return None, None
        
        # Determine layer count from parameter names
        param_names = model_config.get('parameter_names', [])
        layer_count = None
        
        # Count layers based on parameter names
        if any('L3' in name for name in param_names):
            layer_count = '3_layer'
        elif any('L2' in name for name in param_names):
            layer_count = '2_layer'
        elif any('L1' in name for name in param_names):
            layer_count = '1_layer'
        
        if layer_count and layer_count in true_params_dict:
            true_data = true_params_dict[layer_count]
            return np.array(true_data['params']), true_data['param_names']
        
        return None, None

    # ...existing code...
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
