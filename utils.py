#!/usr/bin/env python3
"""
General utility functions for the reflectometry analysis pipeline.

This module contains utility functions that are used across multiple modules
but don't belong to any specific domain (like preprocessing, plotting, etc.).
"""

import json
import numpy as np


def convert_to_json_serializable(obj):
    """
    Recursively convert objects to JSON serializable format.

    Args:
        obj: Object to convert

    Returns:
        JSON serializable version of the object
    """
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return list(convert_to_json_serializable(list(obj)))
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif hasattr(obj, "__dict__"):
        # Handle objects with attributes (like BasicParams)
        return {k: convert_to_json_serializable(v) for k, v in obj.__dict__.items()}
    else:
        # Try to handle other types
        try:
            json.dumps(obj)  # Test if it's already JSON serializable
            return obj
        except (TypeError, ValueError):
            # If all else fails, convert to string
            return str(obj)


def validate_layer_count(layer_count):
    """
    Validate that layer count is within acceptable range.

    Args:
        layer_count: Number of layers to validate

    Returns:
        bool: True if valid

    Raises:
        ValueError: If layer count is invalid
    """
    if not isinstance(layer_count, int):
        raise ValueError(f"Layer count must be an integer, got {type(layer_count)}")

    if layer_count < 0 or layer_count > 2:
        raise ValueError(f"Layer count must be 0, 1, or 2, got {layer_count}")

    return True


def format_parameter_value(param_name, value, units=True):
    """
    Format parameter values for display with appropriate units and precision.

    Args:
        param_name: Name of the parameter
        value: Parameter value
        units: Whether to include units in output

    Returns:
        str: Formatted parameter string
    """
    if "sld" in param_name.lower():
        # SLD values in scientific notation
        if units:
            return f"{value:.2e} Å⁻²"
        else:
            return f"{value:.2e}"
    elif "thickness" in param_name.lower():
        # Thickness in Angstroms
        if units:
            return f"{value:.1f} Å"
        else:
            return f"{value:.1f}"
    elif "rough" in param_name.lower():
        # Roughness in Angstroms
        if units:
            return f"{value:.1f} Å"
        else:
            return f"{value:.1f}"
    else:
        # Default formatting
        if units:
            return f"{value:.3f}"
        else:
            return f"{value:.3f}"


def ensure_directory_exists(directory_path):
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory_path: Path to directory (str or Path object)

    Returns:
        Path: Path object for the directory
    """
    from pathlib import Path

    directory = Path(directory_path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def format_time_duration(seconds):
    """
    Format a time duration in seconds to a human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        str: Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"


if __name__ == "__main__":
    print("Utility functions module loaded successfully.")
    print("Available functions:")
    print("  - convert_to_json_serializable()")
    print("  - validate_layer_count()")
    print("  - format_parameter_value()")
    print("  - ensure_directory_exists()")
    print("  - format_time_duration()")
