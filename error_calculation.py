#!/usr/bin/env python3
"""
Error calculation utilities for reflectometry analysis.

This module calculates regular and constraint-based parameter MAPE.
"""

import numpy as np
import logging

from constraints_utils import get_constraint_widths


logger = logging.getLogger(__name__)


def calculate_parameter_metrics(pred_params, true_params, param_names):
    """Calculate regular and constraint-based MAPE for model parameters."""
    if len(pred_params) != len(true_params) or len(param_names) != len(true_params):
        logger.warning("Parameter values and names must have matching lengths")
        return {
            "overall": {"mape": -1.0, "constraint_mape": -1.0},
            "by_type": {},
            "by_parameter": {},
        }

    predicted = np.asarray(pred_params, dtype=float)
    truth = np.asarray(true_params, dtype=float)
    errors = predicted - truth

    zero_mask = np.abs(truth) < 1e-10
    percentage_errors = np.abs(errors)
    percentage_errors[~zero_mask] = np.abs(errors[~zero_mask] / truth[~zero_mask]) * 100
    if np.any(zero_mask):
        logger.warning(
            "Zero true values found for parameters: %s",
            [param_names[i] for i in np.where(zero_mask)[0]],
        )

    widths = get_constraint_widths()
    constraint_widths = np.empty(len(param_names), dtype=float)
    for index, name in enumerate(param_names):
        width = widths.get(name)
        if width is None:
            raise ValueError(
                f"Unknown parameter type: {name}. Please add it to model_constraints.json"
            )
        constraint_widths[index] = width
    constraint_errors = np.abs(errors) / constraint_widths * 100

    by_type = {}
    for metric_type, indices in {
        "thickness": [
            i for i, name in enumerate(param_names) if "thickness" in name.lower()
        ],
        "roughness": [
            i for i, name in enumerate(param_names) if "rough" in name.lower()
        ],
        "sld": [i for i, name in enumerate(param_names) if "sld" in name.lower()],
    }.items():
        if indices:
            by_type[metric_type] = {
                "mape": float(np.mean(percentage_errors[indices])),
                "constraint_mape": float(np.mean(constraint_errors[indices])),
            }

    by_parameter = {
        name: {
            "percentage_error": float(percentage_errors[index]),
            "constraint_percentage_error": float(constraint_errors[index]),
            "constraint_width": float(constraint_widths[index]),
        }
        for index, name in enumerate(param_names)
    }
    metrics = {
        "overall": {
            "mape": float(np.mean(percentage_errors)),
            "constraint_mape": float(np.mean(constraint_errors)),
        },
        "by_type": by_type,
        "by_parameter": by_parameter,
    }
    logger.info(
        "Overall constraint-based MAPE: %.2f%%", metrics["overall"]["constraint_mape"]
    )
    return metrics


def print_metrics_report(param_metrics):
    """Print overall and per-parameter constraint-based MAPE values."""
    metrics = param_metrics or {}
    overall = metrics.get("overall", {}).get("constraint_mape")
    value = "N/A" if overall is None else f"{overall:.2f}%"
    print(f"Overall constraint-based MAPE: {value}")

    for name, parameter_metrics in metrics.get("by_parameter", {}).items():
        value = parameter_metrics.get("constraint_percentage_error")
        value = "N/A" if value is None else f"{value:.2f}%"
        print(f"  {name}: {value}")
