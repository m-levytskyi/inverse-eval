#!/usr/bin/env python3
"""
NF (Normalizing Flow) sample statistics computation.

This module provides functions to compute statistical summaries across
NF posterior samples for uncertainty quantification and calibration analysis.
"""

import numpy as np


def compute_nf_sample_statistics(predicted_params_array, log_prob):
    """
    Compute statistical summaries across NF parameter samples.

    Args:
        predicted_params_array: 2D array (num_samples, num_params) - all parameter samples
        log_prob: 1D array (num_samples,) - log probability/likelihood for each sample

    Returns:
        dict: Dictionary containing:
            - nf_params_mean: 1D array (num_params,) - mean of each parameter across samples
            - nf_params_std: 1D array (num_params,) - std dev of each parameter across samples
            - nf_params_percentiles: 2D array (5, num_params) - percentiles [5, 25, 50, 75, 95]
            - nf_log_prob_mean: scalar - mean log probability across samples
            - nf_log_prob_std: scalar - std dev of log probability across samples
    """
    # Validate inputs
    predicted_params_array = np.asarray(predicted_params_array)
    log_prob = np.asarray(log_prob)

    if predicted_params_array.ndim != 2:
        raise ValueError(
            f"predicted_params_array must be 2D (num_samples, num_params), "
            f"got shape {predicted_params_array.shape}"
        )

    if log_prob.ndim != 1:
        raise ValueError(
            f"log_prob must be 1D (num_samples,), got shape {log_prob.shape}"
        )

    num_samples, num_params = predicted_params_array.shape

    if log_prob.shape[0] != num_samples:
        raise ValueError(
            f"log_prob length ({log_prob.shape[0]}) must match "
            f"number of samples ({num_samples})"
        )

    # Compute parameter statistics across samples (axis=0)
    params_mean = np.mean(predicted_params_array, axis=0)
    params_std = np.std(predicted_params_array, axis=0, ddof=1)  # Sample std dev

    # Compute percentiles: 5th, 25th, 50th, 75th, 95th
    percentiles = np.percentile(
        predicted_params_array, q=[5, 25, 50, 75, 95], axis=0
    )  # Shape: (5, num_params)

    # Compute log probability statistics
    log_prob_mean = float(np.mean(log_prob))
    log_prob_std = float(np.std(log_prob, ddof=1))

    return {
        "nf_params_mean": params_mean,
        "nf_params_std": params_std,
        "nf_params_percentiles": percentiles,
        "nf_log_prob_mean": log_prob_mean,
        "nf_log_prob_std": log_prob_std,
    }
