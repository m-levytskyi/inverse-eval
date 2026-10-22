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
