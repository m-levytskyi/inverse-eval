#!/usr/bin/env python3
"""
Data preprocessing utilities for reflectometry data.

This module provides functions for cleaning and preprocessing experimental
neutron reflectometry data before analysis.
"""

import numpy as np
import logging


logger = logging.getLogger(__name__)


def remove_negative_values(q_exp, curve_exp, sigmas_exp):
    """
    Remove data points with negative intensity values.

    Args:
        q_exp: Q values
        curve_exp: Reflectivity values
        sigmas_exp: Uncertainty values

    Returns:
        Tuple of filtered (q_exp, curve_exp, sigmas_exp)
    """
    positive_mask = curve_exp > 0
    negative_count = np.sum(~positive_mask)

    if negative_count > 0:
        logger.info("Removing %s points with negative intensity", negative_count)

        q_exp = q_exp[positive_mask]
        curve_exp = curve_exp[positive_mask]
        sigmas_exp = sigmas_exp[positive_mask]

    return q_exp, curve_exp, sigmas_exp


def filter_high_error_points(
    q_exp,
    curve_exp,
    sigmas_exp,
    error_threshold=0.5,
    consecutive_threshold=3,
    remove_singles=False,
):
    """
    Filter or truncate data points with high relative errors.

    Args:
        q_exp: Q values
        curve_exp: Reflectivity values
        sigmas_exp: Uncertainty values
        error_threshold: Relative error threshold (default: 0.5 = 50%)
        consecutive_threshold: Number of consecutive high-error points to trigger truncation
        remove_singles: Whether to remove isolated high-error points

    Returns:
        Tuple of filtered (q_exp, curve_exp, sigmas_exp)
    """
    logger.info("Filtering high error points (threshold: %s%%)", error_threshold * 100)

    # Calculate relative errors
    relative_errors = sigmas_exp / curve_exp
    high_error_mask = relative_errors > error_threshold

    logger.info("Found %s points with high relative error", np.sum(high_error_mask))

    # Find consecutive high-error regions
    consecutive_count = 0
    truncation_index = len(q_exp)

    for i, is_high_error in enumerate(high_error_mask):
        if is_high_error:
            consecutive_count += 1
            if consecutive_count >= consecutive_threshold:
                truncation_index = i - consecutive_threshold + 1
                break
        else:
            consecutive_count = 0

    # Truncate at first long consecutive high-error region
    if truncation_index < len(q_exp):
        logger.info(
            "Truncating data at index %s due to consecutive high-error points",
            truncation_index,
        )
        q_exp = q_exp[:truncation_index]
        curve_exp = curve_exp[:truncation_index]
        sigmas_exp = sigmas_exp[:truncation_index]
        high_error_mask = high_error_mask[:truncation_index]

    # Remove isolated high-error points if requested
    if remove_singles:
        isolated_high_error = []
        for i, is_high_error in enumerate(high_error_mask):
            if is_high_error:
                # Check if it's isolated (no high-error neighbors)
                prev_ok = (i == 0) or not high_error_mask[i - 1]
                next_ok = (i == len(high_error_mask) - 1) or not high_error_mask[i + 1]

                if prev_ok and next_ok:
                    isolated_high_error.append(i)

        if isolated_high_error:
            logger.info(
                "Removing %s isolated high-error points", len(isolated_high_error)
            )
            keep_mask = np.ones(len(q_exp), dtype=bool)
            keep_mask[isolated_high_error] = False

            q_exp = q_exp[keep_mask]
            curve_exp = curve_exp[keep_mask]
            sigmas_exp = sigmas_exp[keep_mask]

    return q_exp, curve_exp, sigmas_exp


def preprocess_experimental_data(
    q_exp,
    curve_exp,
    sigmas_exp,
    error_threshold=0.5,
    consecutive_threshold=3,
    remove_singles=False,
):
    """
    Apply comprehensive preprocessing to experimental data.

    Args:
        q_exp: Q values
        curve_exp: Reflectivity values
        sigmas_exp: Uncertainty values
        error_threshold: Relative error threshold for filtering
        consecutive_threshold: Consecutive high-error points threshold
        remove_singles: Whether to remove isolated high-error points

    Returns:
        Tuple of preprocessed (q_exp, curve_exp, sigmas_exp)
    """
    logger.info("Starting comprehensive data preprocessing")
    logger.info("Initial data points: %s", len(q_exp))

    original_count = len(q_exp)

    # Step 1: Remove negative values
    q_exp, curve_exp, sigmas_exp = remove_negative_values(q_exp, curve_exp, sigmas_exp)

    # Step 2: Filter high error points
    q_exp, curve_exp, sigmas_exp = filter_high_error_points(
        q_exp,
        curve_exp,
        sigmas_exp,
        error_threshold,
        consecutive_threshold,
        remove_singles,
    )

    final_count = len(q_exp)
    removed_count = original_count - final_count
    removal_percentage = (removed_count / original_count) * 100

    logger.info("Preprocessing complete:")
    logger.info("  Original points: %s", original_count)
    logger.info("  Final points: %s", final_count)
    logger.info("  Removed: %s (%.1f%%)", removed_count, removal_percentage)
    logger.info("  Q range: %.4f - %.4f Å⁻¹", q_exp.min(), q_exp.max())

    return q_exp, curve_exp, sigmas_exp
