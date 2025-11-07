#!/usr/bin/env python3
"""
Single experiment processing pipeline for reflectometry analysis.

This module contains functions to process individual experiments, focusing on 
the core workflow for analyzing a single reflectometry experiment.
"""

import torch
import numpy as np
from reflectorch import EasyInferenceModel

# Import our modular utilities
from plotting_utils import plot_simple_comparison
from parameter_discovery import (
    discover_experiment_files, 
    parse_true_parameters_from_model_file,
    generate_true_sld_profile,
    get_prior_bounds_for_experiment,
    get_parameter_names_for_layer_count
)
from error_calculation import (
    calculate_fit_metrics, 
    calculate_parameter_metrics,
    print_metrics_report
)
from data_preprocessing import preprocess_experimental_data
from parameter_constraints import apply_physical_constraints

# Set seed for reproducibility
torch.manual_seed(42)


def load_experimental_data(data_file_path, enable_preprocessing=True, 
                          threshold=0.5, consecutive=3, remove_singles=False):
    """Load and parse experimental data from file with optional preprocessing."""
    print(f"Loading experimental data from: {data_file_path}")
    
    data = np.loadtxt(data_file_path, skiprows=1)
    print(f"Data shape: {data.shape}")
    
    q_exp = data[..., 0]
    curve_exp = data[..., 1]
    
    # Handle both theoretical (3 columns) and experimental (4 columns) data
    if data.shape[1] == 3:
        # Theoretical data: create minimal dummy error bars
        sigmas_exp = np.full_like(curve_exp, 1e-6)
        print("Detected theoretical data (3 columns) - using minimal dummy errors")
    else:
        # Experimental data: use actual error bars
        sigmas_exp = data[..., 2]
        print("Detected experimental data (4 columns) - using actual errors")
    
    print(f"Raw Q range: {q_exp.min():.4f} - {q_exp.max():.4f} Å⁻¹")
    print(f"Raw curve shape: {curve_exp.shape}")
    print(f"Raw relative error range: {(sigmas_exp / curve_exp).min():.4f} - {(sigmas_exp / curve_exp).max():.4f}")
    
    # Apply preprocessing
    q_exp, curve_exp, sigmas_exp = preprocess_experimental_data(
        q_exp, curve_exp, sigmas_exp,
        error_threshold=threshold,
        consecutive_threshold=consecutive,
        remove_singles=remove_singles
    ) if enable_preprocessing else (q_exp, curve_exp, sigmas_exp)
    
    print(f"Final Q range: {q_exp.min():.4f} - {q_exp.max():.4f} Å⁻¹")
    print(f"Final curve shape: {curve_exp.shape}")
    print(f"Final relative error range: {(sigmas_exp / curve_exp).min():.4f} - {(sigmas_exp / curve_exp).max():.4f}")
    
    return q_exp, curve_exp, sigmas_exp

def run_inference(inference_model, q_exp, curve_exp, prior_bounds, q_resolution=0.1, apply_constraints=True):
    """Run the inference prediction."""
    print("Performing inference prediction...")
    
    # Interpolate data to model grid
    q_model, exp_curve_interp = inference_model.interpolate_data_to_model_q(q_exp, curve_exp)
    print(f"Model Q range: {q_model.min():.4f} - {q_model.max():.4f} Å⁻¹")
    print(f"Interpolated curve shape: {exp_curve_interp.shape}")
    
    # Perform prediction
    prediction_dict = inference_model.predict(
        reflectivity_curve=exp_curve_interp,
        prior_bounds=prior_bounds,
        q_values=q_model,
        q_resolution=q_resolution,
        polish_prediction=True,
        calc_pred_curve=True,
        calc_pred_sld_profile=True,
        calc_polished_sld_profile=True,
    )
    
    # Apply physical constraints to prevent negative thickness/roughness (if enabled)
    if apply_constraints:
        print("Applying physical constraints...")
        prediction_dict = apply_physical_constraints(prediction_dict)
    else:
        print("Physical constraints disabled - skipping constraint application")
    
    return q_model, prediction_dict

def display_results(prediction_dict):
    """Display prediction results in a formatted way."""
    print("\nPrediction Results:")
    print("-" * 50)
    
    pred_params = prediction_dict['predicted_params_array']
    polished_params = prediction_dict['polished_params_array']
    param_names = prediction_dict["param_names"]
    
    for param_name, pred_val, polished_val in zip(param_names, pred_params, polished_params):
        print(f'{param_name.ljust(18)} -> Predicted: {pred_val:.3f}    Polished: {polished_val:.3f}')

def run_single_experiment(experiment_id, layer_count=1, enable_preprocessing=True,
                         preprocessing_threshold=0.5, preprocessing_consecutive=3,
                         preprocessing_remove_singles=False, apply_constraints=True,
                         priors_type="broad", priors_deviation=0.5, fix_sld_mode="none",
                         use_theoretical=False):
    """
    Run a single experiment inference with configurable options.
    
    Args:
        experiment_id: ID of the experiment to analyze
        layer_count: Number of layers (1 or 2)
        enable_preprocessing: Whether to enable data preprocessing
        preprocessing_threshold: Error threshold for preprocessing
        preprocessing_consecutive: Consecutive points threshold
        preprocessing_remove_singles: Remove isolated high-error points
        apply_constraints: Whether to apply physical constraints to parameters
        priors_type: Type of priors to use ("broad", "narrow", or "constraint_based")
        priors_deviation: Deviation for narrow priors (e.g., 0.3 for 30%) or 
                         constraint percentage for constraint_based priors
        fix_sld_mode: SLD fixing mode - "none", "backing", or "all"
        use_theoretical: If True, use theoretical curves; if False (default), use experimental curves
    
    Returns:
        Dictionary with results including parameters and metrics
    """
    data_directory = "data"
    
    # Discover experiment files
    data_file, model_file, detected_layer_count = discover_experiment_files(
        experiment_id, data_directory, layer_count, use_theoretical=use_theoretical
    )
    if not data_file:
        raise FileNotFoundError(f"Could not find data file for {experiment_id}")
    
    # Use detected layer count if available
    final_layer_count = detected_layer_count if detected_layer_count else layer_count
    
    # Load experimental data with preprocessing
    q_exp, curve_exp, sigmas_exp = load_experimental_data(
        data_file,
        enable_preprocessing=enable_preprocessing,
        threshold=preprocessing_threshold,
        consecutive=preprocessing_consecutive,
        remove_singles=preprocessing_remove_singles
    )
    
    # Load true parameters if available
    true_params_dict = None
    if model_file:
        true_params_dict = parse_true_parameters_from_model_file(str(model_file))
    
    # Get prior bounds
    prior_bounds = get_prior_bounds_for_experiment(
        experiment_id, 
        true_params_dict, 
        priors_type=priors_type,
        deviation=priors_deviation,
        layer_count=final_layer_count,
        fix_sld_mode=fix_sld_mode
    )
    
    # Initialize inference model
    config_name = 'b_mc_point_neutron_conv_standard_L1_InputQDq'
    inference_model = EasyInferenceModel(config_name=config_name, device='cpu')
    
    # Run inference
    q_model, prediction_dict = run_inference(
        inference_model, q_exp, curve_exp, prior_bounds, q_resolution=0.1, apply_constraints=apply_constraints
    )
    
    # Calculate metrics
    fit_metrics = calculate_fit_metrics(
        curve_exp, 
        prediction_dict['polished_curve'], 
        sigmas_exp, 
        q_exp, 
        q_model
    )
    
    param_metrics = None
    if true_params_dict and f'{final_layer_count}_layer' in true_params_dict:
        param_metrics = calculate_parameter_metrics(
            prediction_dict['polished_params_array'],
            true_params_dict[f'{final_layer_count}_layer']['params'],
            true_params_dict[f'{final_layer_count}_layer']['param_names'],
            prior_bounds=prior_bounds,
            priors_type=priors_type
        )
    
    # Prepare results dictionary
    results = {
        'experiment_id': experiment_id,
        'layer_count': final_layer_count,
        'prediction_dict': prediction_dict,
        'fit_metrics': fit_metrics,
        'param_metrics': param_metrics,
        'true_params_dict': true_params_dict,
        'q_exp': q_exp,
        'curve_exp': curve_exp,
        'sigmas_exp': sigmas_exp,
        'q_model': q_model,
        'prior_bounds': prior_bounds,
        'priors_config': {
            'priors_type': priors_type,
            'priors_deviation': priors_deviation,
            'fix_sld_mode': fix_sld_mode
        }
    }
    
    return results

def main():
    """Main function to run a single experiment with specific settings."""
    # Experiment configuration
    experiment_name = "s007384"
    layer_count = 1
    
    print(f"Running inference for experiment: {experiment_name}")
    print(f"Layer count: {layer_count}")
    print("Running with 99% constraint-based priors and preprocessing OFF.")
    print("="*60)

    # Run the experiment with specified settings
    results = run_single_experiment(
        experiment_id=experiment_name,
        layer_count=layer_count,
        enable_preprocessing=False,
        priors_type="constraint_based",
        priors_deviation=0.99,  # 99% constraint
        use_theoretical=False
    )

    # Unpack results for clarity
    prediction_dict = results['prediction_dict']
    fit_metrics = results['fit_metrics']
    param_metrics = results['param_metrics']
    true_params_dict = results['true_params_dict']
    q_exp = results['q_exp']
    curve_exp = results['curve_exp']
    sigmas_exp = results['sigmas_exp']
    q_model = results['q_model']

    # Display results
    display_results(prediction_dict)
    
    # Print metrics report
    print_metrics_report(fit_metrics, param_metrics, "b_mc_point_neutron_conv_standard_L1_InputQDq")
    
    # Generate true SLD profile for plotting
    if true_params_dict:
        generate_true_sld_profile(true_params_dict)
    
    # Create plots
    plot_simple_comparison(
        q_exp, curve_exp, sigmas_exp,
        q_model, 
        prediction_dict['predicted_curve'],
        prediction_dict['polished_curve'],
        prediction_dict['predicted_sld_xaxis'],
        prediction_dict['predicted_sld_profile'],
        prediction_dict['sld_profile_polished'],
        experiment_name=experiment_name,
        show=True
    )
    
    print(f"\nInference completed for experiment {experiment_name}")


if __name__ == "__main__":
    print("Simple pipeline module loaded successfully.")
    print("Available functions:")
    print("  - run_single_experiment() - Main function for processing one experiment")
    print("  - load_experimental_data() - Load and preprocess data from file")
    print("  - run_inference() - Run reflectorch inference")
    print("  - display_results() - Display prediction results")
    print("\nTo run the example:")
    print("  main()")
    
    # Optionally run the main function
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--run-example":
        main()
    else:
        print("\nUse --run-example to run the built-in example")
