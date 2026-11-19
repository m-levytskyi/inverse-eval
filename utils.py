#!/usr/bin/env python3
"""
General utility functions for the reflectometry analysis pipeline.

This module contains utility functions that are used across multiple modules
but don't belong to any specific domain (like preprocessing, plotting, etc.).
"""

import json
import numpy as np


def to_numpy(value):
    """Detach tensor-like values before exposing their NumPy representation."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def validate_positional_experiment_ids(experiment_ids, row_count):
    """Require the complete zero-based sequence used by positional curve pickles."""
    expected = [f"s{index:06d}" for index in range(row_count)]
    if list(experiment_ids) != expected:
        raise ValueError(
            "Positional pickle rows require the complete ordered experiment sequence "
            "s000000 through the final row. Missing, duplicate, filtered, or reordered "
            "files cannot be matched safely."
        )


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


def write_json_atomic(path, data):
    """Replace a JSON file only after its complete, finite contents reach disk."""
    import os
    from pathlib import Path
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(
                convert_to_json_serializable(data), temporary, indent=2, allow_nan=False
            )
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
