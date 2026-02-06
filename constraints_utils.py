"""
Utility module for loading and accessing model constraints.

This module provides centralized access to physical constraint definitions
for reflectometry parameters.
"""

import json
from pathlib import Path
from typing import Dict, Tuple


# Cache for loaded constraints
_CONSTRAINTS_CACHE = None


def load_constraints() -> Dict:
    """
    Load constraint definitions from JSON file.

    Returns:
        Dictionary containing constraint definitions
    """
    global _CONSTRAINTS_CACHE

    if _CONSTRAINTS_CACHE is not None:
        return _CONSTRAINTS_CACHE

    # Get the path to the constraints file
    constraints_file = Path(__file__).parent / "model_constraints.json"

    if not constraints_file.exists():
        raise FileNotFoundError(f"Constraints file not found: {constraints_file}")

    with open(constraints_file, "r") as f:
        _CONSTRAINTS_CACHE = json.load(f)

    return _CONSTRAINTS_CACHE


def get_constraint_ranges() -> Dict[str, Tuple[float, float]]:
    """
    Get constraint ranges for all parameters.

    Returns:
        Dictionary mapping parameter names to (min, max) tuples
    """
    constraints = load_constraints()

    ranges = {}
    for param_name, param_info in constraints["constraints"].items():
        ranges[param_name] = (param_info["min"], param_info["max"])

    return ranges


def get_constraint_widths() -> Dict[str, float]:
    """
    Get constraint widths (max - min) for all parameters.

    Returns:
        Dictionary mapping parameter names to constraint widths
    """
    constraints = load_constraints()
    return constraints["constraint_widths"].copy()


def get_constraint_range(param_name: str) -> Tuple[float, float]:
    """
    Get constraint range for a specific parameter.

    Args:
        param_name: Parameter name

    Returns:
        Tuple of (min, max) constraint values

    Raises:
        KeyError: If parameter name not found
    """
    ranges = get_constraint_ranges()

    if param_name not in ranges:
        raise KeyError(
            f"Unknown parameter: {param_name}. Available: {list(ranges.keys())}"
        )

    return ranges[param_name]


def get_constraint_width(param_name: str) -> float:
    """
    Get constraint width for a specific parameter.

    Args:
        param_name: Parameter name

    Returns:
        Constraint width (max - min)

    Raises:
        KeyError: If parameter name not found
    """
    widths = get_constraint_widths()

    if param_name not in widths:
        raise KeyError(
            f"Unknown parameter: {param_name}. Available: {list(widths.keys())}"
        )

    return widths[param_name]


def get_parameter_info(param_name: str) -> Dict:
    """
    Get full information about a parameter including units and description.

    Args:
        param_name: Parameter name

    Returns:
        Dictionary with parameter information

    Raises:
        KeyError: If parameter name not found
    """
    constraints = load_constraints()

    if param_name not in constraints["constraints"]:
        raise KeyError(
            f"Unknown parameter: {param_name}. Available: {list(constraints['constraints'].keys())}"
        )

    return constraints["constraints"][param_name].copy()
