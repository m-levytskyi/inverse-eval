#!/usr/bin/env python3

"""
Parameter constraints utility for ensuring physical validity of ReflecTorch predictions.
"""

import logging


logger = logging.getLogger(__name__)


def apply_physical_constraints(prediction_dict, layer_count=1):
    """
    Apply physical constraints to ReflecTorch predictions to prevent negative
    thickness and roughness values.

    Args:
        prediction_dict: Dictionary from ReflecTorch inference containing predictions
        layer_count: Number of layers (1 or 2)

    Returns:
        Modified prediction_dict with constrained parameters
    """

    # Get parameter arrays
    predicted_params = prediction_dict.get("predicted_params_array", None)
    polished_params = prediction_dict.get("polished_params_array", None)
    param_names = prediction_dict.get("param_names", [])

    if predicted_params is None or polished_params is None:
        logger.warning("Could not find parameter arrays in prediction_dict")
        return prediction_dict

    # Define constraints based on parameter names
    constraints = {}
    for i, param_name in enumerate(param_names):
        if "thickness" in param_name.lower():
            constraints[i] = {"min": 0.1, "name": param_name}  # Minimum thickness 0.1 Å
        elif "rough" in param_name.lower():
            constraints[i] = {"min": 0.0, "name": param_name}  # Minimum roughness 0.0 Å

    # Apply constraints to both predicted and polished parameters
    constrained_predicted = predicted_params.copy()
    constrained_polished = polished_params.copy()

    violations_found = False

    for param_idx, constraint in constraints.items():
        param_name = constraint["name"]
        min_val = constraint["min"]

        # Check predicted parameters
        if constrained_predicted[param_idx] < min_val:
            logger.warning(
                "Constraint violation in predicted %s: %.4f < %s",
                param_name,
                constrained_predicted[param_idx],
                min_val,
            )
            constrained_predicted[param_idx] = min_val
            violations_found = True

        # Check polished parameters
        if constrained_polished[param_idx] < min_val:
            logger.warning(
                "Constraint violation in polished %s: %.4f < %s",
                param_name,
                constrained_polished[param_idx],
                min_val,
            )
            constrained_polished[param_idx] = min_val
            violations_found = True

    if violations_found:
        logger.info("Applied physical constraints to prevent negative values")

        # Update prediction dictionary
        prediction_dict["predicted_params_array"] = constrained_predicted
        prediction_dict["polished_params_array"] = constrained_polished

        # Also update any curves that depend on these parameters
        # Note: We don't regenerate the curves here to avoid complexity,
        # but the constraints ensure the parameters are physically valid

    return prediction_dict


def validate_physical_parameters(params, param_names, experiment_id="unknown"):
    """
    Validate that physical parameters meet constraints.

    Args:
        params: Array of parameter values
        param_names: List of parameter names
        experiment_id: Experiment identifier for logging

    Returns:
        bool: True if all parameters are physically valid
    """

    violations = []

    for i, (param_name, value) in enumerate(zip(param_names, params)):
        if "thickness" in param_name.lower() and value < 0:
            violations.append(f"{param_name}: {value:.4f} < 0")
        elif "rough" in param_name.lower() and value < 0:
            violations.append(f"{param_name}: {value:.4f} < 0")

    if violations:
        logger.warning("Physical constraint violations in %s:", experiment_id)
        for violation in violations:
            logger.warning("  - %s", violation)
        return False

    return True


def get_parameter_constraints_info():
    """
    Get information about the parameter constraints being applied.

    Returns:
        Dictionary with constraint information
    """

    constraints = {
        "thickness": {
            "min_value": 0.1,
            "reason": "Physical layers must have positive thickness",
            "unit": "Å",
        },
        "roughness": {
            "min_value": 0.0,
            "reason": "Interface roughness cannot be negative",
            "unit": "Å",
        },
    }

    return constraints


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Print constraint information
    logger.info("PARAMETER CONSTRAINTS FOR REFLECTORCH PREDICTIONS")
    logger.info("=" * 60)

    constraints = get_parameter_constraints_info()

    for param_type, info in constraints.items():
        logger.info("\n%s:", param_type.upper())
        logger.info("  Minimum value: %s %s", info["min_value"], info["unit"])
        logger.info("  Reason: %s", info["reason"])

    logger.info("\nThese constraints prevent the 'Negative roughness encountered' and")
    logger.info(
        "'Negative thickness encountered' errors by ensuring physical validity."
    )
