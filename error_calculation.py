#!/usr/bin/env python3
"""
Error calculation utilities for reflectometry analysis.

This module contains functions to calculate various error metrics and 
goodness-of-fit measures for comparing predicted and experimental data.
"""

import numpy as np


def calculate_fit_metrics(y_exp, y_pred, sigma_exp, q_exp, q_model):
    """
    Calculate fit quality metrics including R-squared, MSE, and L1 loss.
    
    Args:
        y_exp: Experimental reflectivity values
        y_pred: Predicted reflectivity values
        sigma_exp: Experimental uncertainties  
        q_exp: Experimental Q values
        q_model: Model Q values
        
    Returns:
        Dictionary with calculated metrics
    """
    print("Calculating fit metrics (R-squared, MSE, L1 loss)")
    
    # Interpolate predicted curve to experimental Q points for comparison
    print(f"Interpolating predicted curve ({len(y_pred)} pts) to experimental Q points ({len(q_exp)} pts)")
    y_pred_interp = np.interp(q_exp, q_model, y_pred)
    
    # R-squared
    ss_res = np.sum((y_exp - y_pred_interp) ** 2)
    ss_tot = np.sum((y_exp - np.mean(y_exp)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # MSE (Mean Squared Error)
    mse = np.mean((y_exp - y_pred_interp) ** 2)
    
    # L1 Loss (Mean Absolute Error)
    l1_loss = np.mean(np.abs(y_exp - y_pred_interp))
    
    # Weighted chi-squared
    chi_squared = np.sum(((y_exp - y_pred_interp) / sigma_exp) ** 2)
    reduced_chi_squared = chi_squared / (len(y_exp) - 1) if len(y_exp) > 1 else chi_squared
    
    # Relative errors
    relative_errors = np.abs((y_exp - y_pred_interp) / y_exp)
    mean_relative_error = np.mean(relative_errors)
    max_relative_error = np.max(relative_errors)
    
    metrics = {
        'r_squared': float(r_squared),
        'mse': float(mse),
        'l1_loss': float(l1_loss),
        'chi_squared': float(chi_squared),
        'reduced_chi_squared': float(reduced_chi_squared),
        'mean_relative_error': float(mean_relative_error),
        'max_relative_error': float(max_relative_error)
    }
    
    print(f"Calculated fit metrics: {metrics}")
    return metrics


def calculate_parameter_metrics(pred_params, true_params, param_names):
    """
    Calculate parameter metrics: MAPE and MSE for different parameter types.
    
    Args:
        pred_params: Predicted parameter values
        true_params: True parameter values
        param_names: List of parameter names
        
    Returns:
        Dictionary with calculated parameter metrics
    """
    print("Calculating parameter metrics (MAPE, MSE)")
    print(f"  - Predicted params: {pred_params}")
    print(f"  - True params: {true_params}")
    print(f"  - Param names: {param_names}")

    if len(pred_params) != len(true_params):
        print(f"WARNING: Parameter count mismatch. Predicted: {len(pred_params)}, True: {len(true_params)}")
        return {
            'overall': {'mape': -1, 'mse': -1},
            'by_type': {},
            'by_parameter': {}
        }

    # Convert arrays and handle unit conversions for SLD parameters
    pred_params_converted = []
    true_params_converted = []
    
    for i, param_name in enumerate(param_names):
        pred_val = pred_params[i]
        true_val = true_params[i]
        
        # Convert SLD values to same units for proper comparison
        if 'sld' in param_name.lower():
            # The model predicts in scientific notation, true values are in 10^-6 units
            # Convert both to 10^-6 units for comparison
            pred_converted = pred_val * 1e6
            true_converted = true_val  # Already in 10^-6 units from parsing
            print(f"SLD unit conversion for {param_name} - Pred: {pred_val:.6e} -> {pred_converted:.4f}, True: {true_val:.4f}")
        else:
            pred_converted = pred_val
            true_converted = true_val
            
        pred_params_converted.append(pred_converted)
        true_params_converted.append(true_converted)

    pred_array = np.array(pred_params_converted)
    true_array = np.array(true_params_converted)
    
    errors = pred_array - true_array
    squared_errors = errors ** 2
    
    # Calculate percentage errors, handling true zeros
    true_params_mape = np.array(true_params_converted)
    zero_mask = np.abs(true_params_mape) < 1e-10
    
    # For zero true values, use absolute error instead of percentage
    percentage_errors = np.zeros_like(errors)
    nonzero_mask = ~zero_mask
    if np.any(nonzero_mask):
        percentage_errors[nonzero_mask] = np.abs(errors[nonzero_mask] / true_params_mape[nonzero_mask]) * 100
    if np.any(zero_mask):
        percentage_errors[zero_mask] = np.abs(errors[zero_mask])  # Absolute error for zeros
        print(f"WARNING: Zero true values found for parameters: {[param_names[i] for i in np.where(zero_mask)[0]]}")

    # Overall metrics
    overall_mape = np.mean(percentage_errors)
    overall_mse = np.mean(squared_errors)
    
    # Metrics by parameter type
    by_type = {}
    thickness_indices = [i for i, name in enumerate(param_names) if 'thickness' in name.lower()]
    roughness_indices = [i for i, name in enumerate(param_names) if 'rough' in name.lower()]
    sld_indices = [i for i, name in enumerate(param_names) if 'sld' in name.lower()]
    
    if thickness_indices:
        by_type['thickness'] = {
            'mape': float(np.mean(percentage_errors[thickness_indices])),
            'mse': float(np.mean(squared_errors[thickness_indices]))
        }
    
    if roughness_indices:
        by_type['roughness'] = {
            'mape': float(np.mean(percentage_errors[roughness_indices])),
            'mse': float(np.mean(squared_errors[roughness_indices]))
        }
    
    if sld_indices:
        by_type['sld'] = {
            'mape': float(np.mean(percentage_errors[sld_indices])),
            'mse': float(np.mean(squared_errors[sld_indices]))
        }
    
    # Metrics by individual parameter
    by_parameter = {}
    for i, param_name in enumerate(param_names):
        by_parameter[param_name] = {
            'predicted': float(pred_params_converted[i]),
            'true': float(true_params_converted[i]),
            'error': float(errors[i]),
            'percentage_error': float(percentage_errors[i]),
            'squared_error': float(squared_errors[i])
        }
    
    metrics = {
        'overall': {
            'mape': float(overall_mape),
            'mse': float(overall_mse)
        },
        'by_type': by_type,
        'by_parameter': by_parameter
    }
    
    print(f"Parameter metrics calculated:")
    print(f"  - Overall MAPE: {overall_mape:.2f}%")
    print(f"  - Overall MSE: {overall_mse:.6f}")
    
    return metrics


def calculate_residuals(y_exp, y_pred, sigma_exp, q_exp, q_model):
    """
    Calculate residuals between experimental and predicted data.
    
    Args:
        y_exp: Experimental reflectivity values
        y_pred: Predicted reflectivity values
        sigma_exp: Experimental uncertainties
        q_exp: Experimental Q values
        q_model: Model Q values
        
    Returns:
        Dictionary with residual data
    """
    # Interpolate predicted curve to experimental Q points
    y_pred_interp = np.interp(q_exp, q_model, y_pred)
    
    # Calculate residuals
    residuals = y_exp - y_pred_interp
    standardized_residuals = residuals / sigma_exp
    
    # Calculate statistics
    residual_stats = {
        'mean': float(np.mean(residuals)),
        'std': float(np.std(residuals)),
        'max_abs': float(np.max(np.abs(residuals))),
        'rms': float(np.sqrt(np.mean(residuals**2)))
    }
    
    standardized_stats = {
        'mean': float(np.mean(standardized_residuals)),
        'std': float(np.std(standardized_residuals)),
        'max_abs': float(np.max(np.abs(standardized_residuals))),
        'rms': float(np.sqrt(np.mean(standardized_residuals**2)))
    }
    
    return {
        'q_exp': q_exp,
        'residuals': residuals,
        'standardized_residuals': standardized_residuals,
        'residual_stats': residual_stats,
        'standardized_stats': standardized_stats
    }


def calculate_confidence_intervals(pred_params, true_params=None, confidence_level=0.95):
    """
    Calculate confidence intervals for predicted parameters.
    
    This is a simplified implementation. In practice, you might want to use
    bootstrap methods or uncertainty propagation from the model.
    
    Args:
        pred_params: Predicted parameter values
        true_params: True parameter values (optional, for validation)
        confidence_level: Confidence level (default: 0.95)
        
    Returns:
        Dictionary with confidence interval information
    """
    # This is a placeholder implementation
    # In practice, you would need uncertainty estimates from the model
    
    alpha = 1 - confidence_level
    z_score = 1.96  # For 95% confidence interval
    
    # Assume 10% uncertainty as a rough estimate
    # This should be replaced with actual uncertainty estimates
    uncertainties = np.array(pred_params) * 0.1
    
    lower_bounds = np.array(pred_params) - z_score * uncertainties
    upper_bounds = np.array(pred_params) + z_score * uncertainties
    
    intervals = {
        'confidence_level': confidence_level,
        'lower_bounds': lower_bounds.tolist(),
        'upper_bounds': upper_bounds.tolist(),
        'uncertainties': uncertainties.tolist()
    }
    
    if true_params is not None:
        # Check if true values fall within confidence intervals
        true_array = np.array(true_params)
        within_ci = (true_array >= lower_bounds) & (true_array <= upper_bounds)
        intervals['true_within_ci'] = within_ci.tolist()
        intervals['coverage'] = float(np.mean(within_ci))
    
    return intervals


def summary_statistics(fit_metrics, param_metrics=None):
    """
    Generate summary statistics for model performance.
    
    Args:
        fit_metrics: Dictionary of fit metrics
        param_metrics: Dictionary of parameter metrics (optional)
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'fit_quality': {
            'excellent': fit_metrics.get('r_squared', 0) > 0.95,
            'good': fit_metrics.get('r_squared', 0) > 0.90,
            'acceptable': fit_metrics.get('r_squared', 0) > 0.80,
            'poor': fit_metrics.get('r_squared', 0) <= 0.80
        },
        'primary_metrics': {
            'r_squared': fit_metrics.get('r_squared', 0),
            'reduced_chi_squared': fit_metrics.get('reduced_chi_squared', float('inf')),
            'mean_relative_error': fit_metrics.get('mean_relative_error', float('inf'))
        }
    }
    
    if param_metrics:
        summary['parameter_accuracy'] = {
            'overall_mape': param_metrics.get('overall', {}).get('mape', float('inf')),
            'excellent': param_metrics.get('overall', {}).get('mape', float('inf')) < 5,
            'good': param_metrics.get('overall', {}).get('mape', float('inf')) < 10,
            'acceptable': param_metrics.get('overall', {}).get('mape', float('inf')) < 20,
            'poor': param_metrics.get('overall', {}).get('mape', float('inf')) >= 20
        }
    
    return summary


def print_metrics_report(fit_metrics, param_metrics=None, model_name="Model"):
    """
    Print a formatted report of all calculated metrics.
    
    Args:
        fit_metrics: Dictionary of fit metrics
        param_metrics: Dictionary of parameter metrics (optional)
        model_name: Name of the model for the report
    """
    print(f"\n{'='*60}")
    print(f"METRICS REPORT: {model_name}")
    print(f"{'='*60}")
    
    print(f"\nFIT QUALITY METRICS:")
    print(f"  R-squared:              {fit_metrics.get('r_squared', 'N/A'):.6f}")
    print(f"  MSE:                    {fit_metrics.get('mse', 'N/A'):.6e}")
    print(f"  L1 Loss:                {fit_metrics.get('l1_loss', 'N/A'):.6e}")
    print(f"  Chi-squared:            {fit_metrics.get('chi_squared', 'N/A'):.6f}")
    print(f"  Reduced Chi-squared:    {fit_metrics.get('reduced_chi_squared', 'N/A'):.6f}")
    print(f"  Mean Relative Error:    {fit_metrics.get('mean_relative_error', 'N/A'):.4f}")
    print(f"  Max Relative Error:     {fit_metrics.get('max_relative_error', 'N/A'):.4f}")
    
    if param_metrics:
        print(f"\nPARAMETER ACCURACY:")
        print(f"  Overall MAPE:           {param_metrics.get('overall', {}).get('mape', 'N/A'):.2f}%")
        print(f"  Overall MSE:            {param_metrics.get('overall', {}).get('mse', 'N/A'):.6f}")
        
        if 'by_type' in param_metrics:
            print(f"\n  By Parameter Type:")
            for param_type, metrics in param_metrics['by_type'].items():
                print(f"    {param_type.capitalize()}:")
                print(f"      MAPE: {metrics.get('mape', 'N/A'):.2f}%")
                print(f"      MSE:  {metrics.get('mse', 'N/A'):.6f}")
    
    # Generate quality assessment
    summary = summary_statistics(fit_metrics, param_metrics)
    
    print(f"\nQUALITY ASSESSMENT:")
    if summary['fit_quality']['excellent']:
        print(f"  Fit Quality: EXCELLENT (R² > 0.95)")
    elif summary['fit_quality']['good']:
        print(f"  Fit Quality: GOOD (R² > 0.90)")
    elif summary['fit_quality']['acceptable']:
        print(f"  Fit Quality: ACCEPTABLE (R² > 0.80)")
    else:
        print(f"  Fit Quality: POOR (R² ≤ 0.80)")
    
    if param_metrics:
        if summary['parameter_accuracy']['excellent']:
            print(f"  Parameter Accuracy: EXCELLENT (MAPE < 5%)")
        elif summary['parameter_accuracy']['good']:
            print(f"  Parameter Accuracy: GOOD (MAPE < 10%)")
        elif summary['parameter_accuracy']['acceptable']:
            print(f"  Parameter Accuracy: ACCEPTABLE (MAPE < 20%)")
        else:
            print(f"  Parameter Accuracy: POOR (MAPE ≥ 20%)")
    
    print(f"{'='*60}\n")
