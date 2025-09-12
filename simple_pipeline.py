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
    sigmas_exp = data[..., 2]
    
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
                         priors_type="broad", priors_deviation=0.5):
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
        priors_type: Type of priors to use ("broad" or "narrow")
        priors_deviation: Deviation for narrow priors (e.g., 0.3 for 30%)
    
    Returns:
        Dictionary with results including parameters and metrics
    """
    data_directory = "data"
    
    # Discover experiment files
    data_file, model_file, detected_layer_count = discover_experiment_files(
        experiment_id, data_directory, layer_count
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
        layer_count=final_layer_count
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
            true_params_dict[f'{final_layer_count}_layer']['param_names']
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
            'priors_deviation': priors_deviation
        }
    }
    
    return results

def main():
    # Experiment configuration
    experiment_name = "s005063"
    data_directory = "data"
    layer_count = 1  # This experiment has 1 layer
    
    # Preprocessing configuration
    preprocessing_config = {
        'enable_preprocessing': True,
        'threshold': 0.75,          # Relative error threshold (50%)
        'consecutive': 5,          # Number of consecutive bad points to trigger truncation
        'remove_singles': False    # Remove isolated high-error points
    }
    
    print(f"Running inference for experiment: {experiment_name}")
    print(f"Layer count: {layer_count}")
    print(f"Preprocessing enabled: {preprocessing_config['enable_preprocessing']}")
    if preprocessing_config['enable_preprocessing']:
        print(f"  Threshold: {preprocessing_config['threshold']}")
        print(f"  Consecutive points: {preprocessing_config['consecutive']}")
        print(f"  Remove singles: {preprocessing_config['remove_singles']}")
    print("="*60)
    
    # Discover experiment files
    data_file, model_file, detected_layer_count = discover_experiment_files(
        experiment_name, data_directory, layer_count
    )
    if not data_file:
        print(f"ERROR: Could not find data file for {experiment_name}")
        return
    
    # Use detected layer count if available, otherwise use specified
    final_layer_count = detected_layer_count if detected_layer_count else layer_count
    print(f"Using layer count: {final_layer_count}")
    
    # Load experimental data with preprocessing
    q_exp, curve_exp, sigmas_exp = load_experimental_data(
        data_file, 
        enable_preprocessing=preprocessing_config['enable_preprocessing'],
        threshold=preprocessing_config['threshold'],
        consecutive=preprocessing_config['consecutive'],
        remove_singles=preprocessing_config['remove_singles']
    )
    
    # Load true parameters if available
    true_params_dict = None
    if model_file:
        true_params_dict = parse_true_parameters_from_model_file(str(model_file))
        print(f"Loaded true parameters: {true_params_dict}")
    
    # Get prior bounds
    prior_bounds = get_prior_bounds_for_experiment(
        experiment_name, 
        true_params_dict, 
        priors_type="broad",
        layer_count=final_layer_count
    )
    
    print("\nPrior bounds:")
    param_names = get_parameter_names_for_layer_count(final_layer_count)
    for name, bounds in zip(param_names, prior_bounds):
        print(f"  {name}: {bounds}")
    
    # Initialize inference model
    config_name = 'b_mc_point_neutron_conv_standard_L1_InputQDq'
    print(f"\nInitializing model: {config_name}")
    
    inference_model = EasyInferenceModel(config_name=config_name, device='cpu')
    
    # Run inference
    q_model, prediction_dict = run_inference(
        inference_model, q_exp, curve_exp, prior_bounds, q_resolution=0.1, apply_constraints=True
    )
    
    # Display results
    display_results(prediction_dict)
    
    # Calculate error metrics
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
            true_params_dict[f'{final_layer_count}_layer']['param_names']
        )
    
    # Print metrics report
    print_metrics_report(fit_metrics, param_metrics, config_name)
    
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
