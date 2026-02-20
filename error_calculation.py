#!/usr/bin/env python3
"""
Error calculation utilities for reflectometry analysis.

This module contains functions to calculate various error metrics and
goodness-of-fit measures for comparing predicted and experimental data.
"""

import numpy as np
from constraints_utils import get_constraint_width


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
    print(
        f"Interpolating predicted curve ({len(y_pred)} pts) to experimental Q points ({len(q_exp)} pts)"
    )
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
    reduced_chi_squared = (
        chi_squared / (len(y_exp) - 1) if len(y_exp) > 1 else chi_squared
    )

    # Relative errors
    relative_errors = np.abs((y_exp - y_pred_interp) / y_exp)
    mean_relative_error = np.mean(relative_errors)
    max_relative_error = np.max(relative_errors)

    metrics = {
        "r_squared": float(r_squared),
        "mse": float(mse),
        "l1_loss": float(l1_loss),
        "chi_squared": float(chi_squared),
        "reduced_chi_squared": float(reduced_chi_squared),
        "mean_relative_error": float(mean_relative_error),
        "max_relative_error": float(max_relative_error),
    }

    print(f"Calculated fit metrics: {metrics}")
    return metrics


def calculate_parameter_metrics(
    pred_params, true_params, param_names, prior_bounds=None, priors_type=None
):
    """
    Calculate parameter metrics: MAPE and MSE for different parameter types.

    For constraint-based priors, also calculates constraint-based MAPE which normalizes
    errors by the constraint interval width instead of the prior interval width. This
    provides an error metric that is bounded by the prior deviation percentage (e.g.,
    30% for 30% constraint-based priors, 99% for 99% constraint-based priors).

    Args:
        pred_params: Predicted parameter values
        true_params: True parameter values
        param_names: List of parameter names
        prior_bounds: Optional prior bounds array (for constraint-based MAPE calculation)
        priors_type: Optional priors type string ("constraint_based" or "narrow")

    Returns:
        Dictionary with calculated parameter metrics
    """
    print("Calculating parameter metrics (MAPE, MSE)")
    print(f"  - Predicted params: {pred_params}")
    print(f"  - True params: {true_params}")
    print(f"  - Param names: {param_names}")

    if len(pred_params) != len(true_params):
        print(
            f"WARNING: Parameter count mismatch. Predicted: {len(pred_params)}, True: {len(true_params)}"
        )
        return {"overall": {"mape": -1, "mse": -1}, "by_type": {}, "by_parameter": {}}

    # Convert arrays and handle unit conversions for SLD parameters
    pred_params_converted = []
    true_params_converted = []

    for i, param_name in enumerate(param_names):
        pred_val = pred_params[i]
        true_val = true_params[i]

        # No unit conversion needed - both predicted and true values are already in consistent units
        # The parsing process has already converted true SLD values to the same scale as predictions
        pred_converted = pred_val
        true_converted = true_val

        if "sld" in param_name.lower():
            print(
                f"SLD comparison for {param_name} - Pred: {pred_val:.6f}, True: {true_val:.6f} (no conversion needed)"
            )

        pred_params_converted.append(pred_converted)
        true_params_converted.append(true_converted)

    pred_array = np.array(pred_params_converted)
    true_array = np.array(true_params_converted)

    errors = pred_array - true_array
    squared_errors = errors**2

    # Calculate percentage errors, handling true zeros
    true_params_mape = np.array(true_params_converted)
    zero_mask = np.abs(true_params_mape) < 1e-10

    # For zero true values, use absolute error instead of percentage
    percentage_errors = np.zeros_like(errors)
    nonzero_mask = ~zero_mask
    if np.any(nonzero_mask):
        percentage_errors[nonzero_mask] = (
            np.abs(errors[nonzero_mask] / true_params_mape[nonzero_mask]) * 100
        )
    if np.any(zero_mask):
        percentage_errors[zero_mask] = np.abs(
            errors[zero_mask]
        )  # Absolute error for zeros
        print(
            f"WARNING: Zero true values found for parameters: {[param_names[i] for i in np.where(zero_mask)[0]]}"
        )

    # Overall metrics
    overall_mape = np.mean(percentage_errors)
    overall_mse = np.mean(squared_errors)

    # Calculate constraint-based MAPE if using constraint-based priors
    constraint_based_percentage_errors = None
    overall_constraint_mape = None

    if priors_type == "constraint_based" and prior_bounds is not None:
        print(
            "Calculating constraint-based MAPE (normalized by constraint interval width)"
        )

        constraint_based_percentage_errors = np.zeros_like(errors)

        for i in range(len(errors)):
            param_name = param_names[i]

            # Get constraint width from centralized definition
            try:
                constraint_width = get_constraint_width(param_name)
            except KeyError:
                raise ValueError(
                    f"Unknown parameter type: {param_name}. "
                    f"Please add to model_constraints.json"
                )

            # Constraint-based percentage error = |error| / constraint_width * 100
            # This normalizes error by the full constraint space, so max error is bounded
            # by the prior deviation percentage (e.g., 30% for 30% constraint-based priors)
            constraint_based_percentage_errors[i] = (
                np.abs(errors[i]) / constraint_width * 100
            )

        overall_constraint_mape = np.mean(constraint_based_percentage_errors)
        print(f"  - Overall Constraint-based MAPE: {overall_constraint_mape:.2f}%")

    # Metrics by parameter type
    by_type = {}
    thickness_indices = [
        i for i, name in enumerate(param_names) if "thickness" in name.lower()
    ]
    roughness_indices = [
        i for i, name in enumerate(param_names) if "rough" in name.lower()
    ]
    sld_indices = [i for i, name in enumerate(param_names) if "sld" in name.lower()]

    if thickness_indices:
        type_metrics = {
            "mape": float(np.mean(percentage_errors[thickness_indices])),
            "mse": float(np.mean(squared_errors[thickness_indices])),
        }
        if constraint_based_percentage_errors is not None:
            type_metrics["constraint_mape"] = float(
                np.mean(constraint_based_percentage_errors[thickness_indices])
            )
        by_type["thickness"] = type_metrics

    if roughness_indices:
        type_metrics = {
            "mape": float(np.mean(percentage_errors[roughness_indices])),
            "mse": float(np.mean(squared_errors[roughness_indices])),
        }
        if constraint_based_percentage_errors is not None:
            type_metrics["constraint_mape"] = float(
                np.mean(constraint_based_percentage_errors[roughness_indices])
            )
        by_type["roughness"] = type_metrics

    if sld_indices:
        type_metrics = {
            "mape": float(np.mean(percentage_errors[sld_indices])),
            "mse": float(np.mean(squared_errors[sld_indices])),
        }
        if constraint_based_percentage_errors is not None:
            type_metrics["constraint_mape"] = float(
                np.mean(constraint_based_percentage_errors[sld_indices])
            )
        by_type["sld"] = type_metrics

    # Metrics by individual parameter
    by_parameter = {}
    for i, param_name in enumerate(param_names):
        param_metrics = {
            "predicted": float(pred_params_converted[i]),
            "true": float(true_params_converted[i]),
            "error": float(errors[i]),
            "percentage_error": float(percentage_errors[i]),
            "squared_error": float(squared_errors[i]),
        }

        # Add constraint-based metrics if available
        if constraint_based_percentage_errors is not None:
            param_metrics["constraint_percentage_error"] = float(
                constraint_based_percentage_errors[i]
            )
            param_metrics["prior_bounds"] = [
                float(prior_bounds[i][0]),
                float(prior_bounds[i][1]),
            ]
            param_metrics["prior_width"] = float(
                prior_bounds[i][1] - prior_bounds[i][0]
            )

            # Add constraint width from centralized definition
            try:
                param_metrics["constraint_width"] = get_constraint_width(param_name)
            except KeyError:
                pass  # Skip if parameter not found

        by_parameter[param_name] = param_metrics

    metrics = {
        "overall": {"mape": float(overall_mape), "mse": float(overall_mse)},
        "by_type": by_type,
        "by_parameter": by_parameter,
    }

    # Add constraint-based overall MAPE if calculated
    if overall_constraint_mape is not None:
        metrics["overall"]["constraint_mape"] = float(overall_constraint_mape)
        metrics["priors_type"] = priors_type

    print("Parameter metrics calculated:")
    print(f"  - Overall MAPE: {overall_mape:.2f}%")
    if overall_constraint_mape is not None:
        print(f"  - Overall Constraint-based MAPE: {overall_constraint_mape:.2f}%")
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
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "max_abs": float(np.max(np.abs(residuals))),
        "rms": float(np.sqrt(np.mean(residuals**2))),
    }

    standardized_stats = {
        "mean": float(np.mean(standardized_residuals)),
        "std": float(np.std(standardized_residuals)),
        "max_abs": float(np.max(np.abs(standardized_residuals))),
        "rms": float(np.sqrt(np.mean(standardized_residuals**2))),
    }

    return {
        "q_exp": q_exp,
        "residuals": residuals,
        "standardized_residuals": standardized_residuals,
        "residual_stats": residual_stats,
        "standardized_stats": standardized_stats,
    }


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
        "fit_quality": {
            "excellent": fit_metrics.get("r_squared", 0) > 0.95,
            "good": fit_metrics.get("r_squared", 0) > 0.90,
            "acceptable": fit_metrics.get("r_squared", 0) > 0.80,
            "poor": fit_metrics.get("r_squared", 0) <= 0.80,
        },
        "primary_metrics": {
            "r_squared": fit_metrics.get("r_squared", 0),
            "reduced_chi_squared": fit_metrics.get("reduced_chi_squared", float("inf")),
            "mean_relative_error": fit_metrics.get("mean_relative_error", float("inf")),
        },
    }

    if param_metrics:
        # Use constraint-based MAPE for assessment if available, otherwise traditional MAPE
        mape_for_assessment = param_metrics.get("overall", {}).get(
            "constraint_mape",
            param_metrics.get("overall", {}).get("mape", float("inf")),
        )

        summary["parameter_accuracy"] = {
            "overall_mape": mape_for_assessment,
            "excellent": mape_for_assessment < 5,
            "good": mape_for_assessment < 10,
            "acceptable": mape_for_assessment < 20,
            "poor": mape_for_assessment >= 20,
        }

    return summary


def print_metrics_report(fit_metrics=None, param_metrics=None, model_name="Model"):
    """
    Print a formatted report of all calculated metrics.

    Args:
        fit_metrics: Dictionary of fit metrics (optional)
        param_metrics: Dictionary of parameter metrics (optional)
        model_name: Name of the model for the report
    """
    print(f"\n{'=' * 60}")
    print(f"METRICS REPORT: {model_name}")
    print(f"{'=' * 60}")

    if fit_metrics is not None:
        print("\nFIT QUALITY METRICS:")
        print(f"  R-squared:              {fit_metrics.get('r_squared', 'N/A'):.6f}")
        print(f"  MSE:                    {fit_metrics.get('mse', 'N/A'):.6e}")
        print(f"  L1 Loss:                {fit_metrics.get('l1_loss', 'N/A'):.6e}")
        print(f"  Chi-squared:            {fit_metrics.get('chi_squared', 'N/A'):.6f}")
        print(
            f"  Reduced Chi-squared:    {fit_metrics.get('reduced_chi_squared', 'N/A'):.6f}"
        )
        print(
            f"  Mean Relative Error:    {fit_metrics.get('mean_relative_error', 'N/A'):.4f}"
        )
        print(
            f"  Max Relative Error:     {fit_metrics.get('max_relative_error', 'N/A'):.4f}"
        )

    if param_metrics:
        print("\nPARAMETER ACCURACY:")

        # Use constraint-based MAPE as primary metric when using constraint-based priors
        use_constraint_mape = "constraint_mape" in param_metrics.get("overall", {})

        if use_constraint_mape:
            print(
                f"  Overall Constraint-based MAPE: {param_metrics['overall']['constraint_mape']:.2f}%"
            )
            print(
                f"  (Traditional MAPE:             {param_metrics.get('overall', {}).get('mape', 'N/A'):.2f}%)"
            )
        else:
            print(
                f"  Overall MAPE:           {param_metrics.get('overall', {}).get('mape', 'N/A'):.2f}%"
            )

        print(
            f"  Overall MSE:            {param_metrics.get('overall', {}).get('mse', 'N/A'):.6f}"
        )

        if "by_type" in param_metrics:
            print("\n  By Parameter Type:")
            for param_type, metrics in param_metrics["by_type"].items():
                print(f"    {param_type.capitalize()}:")
                if use_constraint_mape and "constraint_mape" in metrics:
                    print(
                        f"      Constraint-based MAPE: {metrics.get('constraint_mape', 'N/A'):.2f}%"
                    )
                    print(
                        f"      (Traditional MAPE:     {metrics.get('mape', 'N/A'):.2f}%)"
                    )
                else:
                    print(f"      MAPE: {metrics.get('mape', 'N/A'):.2f}%")
                print(f"      MSE:  {metrics.get('mse', 'N/A'):.6f}")

    # Generate quality assessment
    use_constraint_mape = param_metrics and "constraint_mape" in param_metrics.get(
        "overall", {}
    )
    mape_type = "Constraint-based MAPE" if use_constraint_mape else "MAPE"

    if fit_metrics is not None or param_metrics is not None:
        summary = summary_statistics(fit_metrics or {}, param_metrics)
        print("\nQUALITY ASSESSMENT:")
        if fit_metrics is not None:
            if summary["fit_quality"]["excellent"]:
                print("  Fit Quality: EXCELLENT (R² > 0.95)")
            elif summary["fit_quality"]["good"]:
                print("  Fit Quality: GOOD (R² > 0.90)")
            elif summary["fit_quality"]["acceptable"]:
                print("  Fit Quality: ACCEPTABLE (R² > 0.80)")
            else:
                print("  Fit Quality: POOR (R² ≤ 0.80)")

        if param_metrics:
            if summary["parameter_accuracy"]["excellent"]:
                print(f"  Parameter Accuracy: EXCELLENT ({mape_type} < 5%)")
            elif summary["parameter_accuracy"]["good"]:
                print(f"  Parameter Accuracy: GOOD ({mape_type} < 10%)")
            elif summary["parameter_accuracy"]["acceptable"]:
                print(f"  Parameter Accuracy: ACCEPTABLE ({mape_type} < 20%)")
            else:
                print(f"  Parameter Accuracy: POOR ({mape_type} ≥ 20%)")

    print(f"{'=' * 60}\n")
