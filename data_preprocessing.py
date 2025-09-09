#!/usr/bin/env python3
"""
Data preprocessing utilities for reflectometry data.

This module provides functions for cleaning and preprocessing experimental
neutron reflectometry data before analysis.
"""

import numpy as np
from pathlib import Path


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
        print(f"Removing {negative_count} points with negative intensity")
        
        q_exp = q_exp[positive_mask]
        curve_exp = curve_exp[positive_mask]
        sigmas_exp = sigmas_exp[positive_mask]
    
    return q_exp, curve_exp, sigmas_exp


def filter_high_error_points(q_exp, curve_exp, sigmas_exp, 
                           error_threshold=0.5, consecutive_threshold=3, 
                           remove_singles=False):
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
    print(f"Filtering high error points (threshold: {error_threshold*100}%)")
    
    # Calculate relative errors
    relative_errors = sigmas_exp / curve_exp
    high_error_mask = relative_errors > error_threshold
    
    print(f"Found {np.sum(high_error_mask)} points with high relative error")
    
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
        print(f"Truncating data at index {truncation_index} due to consecutive high-error points")
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
                prev_ok = (i == 0) or not high_error_mask[i-1]
                next_ok = (i == len(high_error_mask)-1) or not high_error_mask[i+1]
                
                if prev_ok and next_ok:
                    isolated_high_error.append(i)
        
        if isolated_high_error:
            print(f"Removing {len(isolated_high_error)} isolated high-error points")
            keep_mask = np.ones(len(q_exp), dtype=bool)
            keep_mask[isolated_high_error] = False
            
            q_exp = q_exp[keep_mask]
            curve_exp = curve_exp[keep_mask]
            sigmas_exp = sigmas_exp[keep_mask]
    
    return q_exp, curve_exp, sigmas_exp


def remove_outliers(q_exp, curve_exp, sigmas_exp, z_threshold=3.0):
    """
    Remove statistical outliers using z-score method.
    
    Args:
        q_exp: Q values
        curve_exp: Reflectivity values
        sigmas_exp: Uncertainty values
        z_threshold: Z-score threshold for outlier detection
        
    Returns:
        Tuple of filtered (q_exp, curve_exp, sigmas_exp)
    """
    print(f"Removing outliers (z-score threshold: {z_threshold})")
    
    # Calculate z-scores for reflectivity values
    log_curve = np.log10(curve_exp)
    mean_log = np.mean(log_curve)
    std_log = np.std(log_curve)
    z_scores = np.abs((log_curve - mean_log) / std_log)
    
    outlier_mask = z_scores > z_threshold
    outlier_count = np.sum(outlier_mask)
    
    if outlier_count > 0:
        print(f"Removing {outlier_count} outliers")
        keep_mask = ~outlier_mask
        
        q_exp = q_exp[keep_mask]
        curve_exp = curve_exp[keep_mask]
        sigmas_exp = sigmas_exp[keep_mask]
    
    return q_exp, curve_exp, sigmas_exp


def smooth_data(q_exp, curve_exp, sigmas_exp, window_size=3):
    """
    Apply simple moving average smoothing to the data.
    
    Args:
        q_exp: Q values
        curve_exp: Reflectivity values
        sigmas_exp: Uncertainty values
        window_size: Size of smoothing window
        
    Returns:
        Tuple of smoothed (q_exp, curve_exp, sigmas_exp)
    """
    if window_size <= 1:
        return q_exp, curve_exp, sigmas_exp
    
    print(f"Applying smoothing with window size {window_size}")
    
    # Use convolution for smoothing
    kernel = np.ones(window_size) / window_size
    
    # Pad the data to handle edges
    pad_size = window_size // 2
    curve_padded = np.pad(curve_exp, pad_size, mode='edge')
    sigma_padded = np.pad(sigmas_exp, pad_size, mode='edge')
    
    # Apply smoothing
    curve_smoothed = np.convolve(curve_padded, kernel, mode='valid')
    sigma_smoothed = np.convolve(sigma_padded, kernel, mode='valid')
    
    return q_exp, curve_smoothed, sigma_smoothed


def preprocess_experimental_data(q_exp, curve_exp, sigmas_exp, 
                                error_threshold=0.5, consecutive_threshold=3,
                                remove_singles=False, remove_outliers_flag=False,
                                smooth_flag=False, smooth_window=3):
    """
    Apply comprehensive preprocessing to experimental data.
    
    Args:
        q_exp: Q values
        curve_exp: Reflectivity values
        sigmas_exp: Uncertainty values
        error_threshold: Relative error threshold for filtering
        consecutive_threshold: Consecutive high-error points threshold
        remove_singles: Whether to remove isolated high-error points
        remove_outliers_flag: Whether to remove statistical outliers
        smooth_flag: Whether to apply smoothing
        smooth_window: Smoothing window size
        
    Returns:
        Tuple of preprocessed (q_exp, curve_exp, sigmas_exp)
    """
    print("Starting comprehensive data preprocessing")
    print(f"Initial data points: {len(q_exp)}")
    
    original_count = len(q_exp)
    
    # Step 1: Remove negative values
    q_exp, curve_exp, sigmas_exp = remove_negative_values(q_exp, curve_exp, sigmas_exp)
    
    # Step 2: Filter high error points
    q_exp, curve_exp, sigmas_exp = filter_high_error_points(
        q_exp, curve_exp, sigmas_exp, error_threshold, 
        consecutive_threshold, remove_singles
    )
    
    # Step 3: Remove outliers (optional)
    if remove_outliers_flag:
        q_exp, curve_exp, sigmas_exp = remove_outliers(q_exp, curve_exp, sigmas_exp)
    
    # Step 4: Apply smoothing (optional)
    if smooth_flag:
        q_exp, curve_exp, sigmas_exp = smooth_data(q_exp, curve_exp, sigmas_exp, smooth_window)
    
    final_count = len(q_exp)
    removed_count = original_count - final_count
    removal_percentage = (removed_count / original_count) * 100
    
    print(f"Preprocessing complete:")
    print(f"  Original points: {original_count}")
    print(f"  Final points: {final_count}")
    print(f"  Removed: {removed_count} ({removal_percentage:.1f}%)")
    print(f"  Q range: {q_exp.min():.4f} - {q_exp.max():.4f} Å⁻¹")
    
    return q_exp, curve_exp, sigmas_exp


def load_and_preprocess_file(file_path, skip_rows=1, **preprocess_kwargs):
    """
    Load and preprocess data from a file.
    
    Args:
        file_path: Path to data file
        skip_rows: Number of header rows to skip
        **preprocess_kwargs: Arguments for preprocessing function
        
    Returns:
        Tuple of preprocessed (q_exp, curve_exp, sigmas_exp)
    """
    print(f"Loading and preprocessing data from: {file_path}")
    
    # Load data
    data = np.loadtxt(file_path, skiprows=skip_rows)
    
    if data.shape[1] < 3:
        raise ValueError(f"Data file must have at least 3 columns (Q, R, dR), found {data.shape[1]}")
    
    q_exp = data[:, 0]
    curve_exp = data[:, 1]
    sigmas_exp = data[:, 2]
    
    print(f"Loaded data: {data.shape[0]} points, Q range: {q_exp.min():.4f} - {q_exp.max():.4f}")
    
    # Apply preprocessing
    q_exp, curve_exp, sigmas_exp = preprocess_experimental_data(
        q_exp, curve_exp, sigmas_exp, **preprocess_kwargs
    )
    
    return q_exp, curve_exp, sigmas_exp


def save_preprocessed_data(q_exp, curve_exp, sigmas_exp, output_path, header="Q(A^-1) R dR"):
    """
    Save preprocessed data to a file.
    
    Args:
        q_exp: Q values
        curve_exp: Reflectivity values
        sigmas_exp: Uncertainty values
        output_path: Output file path
        header: Header string for the file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Stack data
    data = np.column_stack([q_exp, curve_exp, sigmas_exp])
    
    # Save with header
    np.savetxt(output_path, data, header=header, fmt='%.6e')
    
    print(f"Preprocessed data saved to: {output_path}")


def main():
    """Example usage of preprocessing functions."""
    
    # Example file path
    input_file = "data/MARIA_VIPR_dataset/1/s005888_experimental_curve.dat"
    output_file = "preprocessed_data/s005888_preprocessed.dat"
    
    if not Path(input_file).exists():
        print(f"Example file not found: {input_file}")
        print("Please provide a valid input file path.")
        return
    
    # Load and preprocess
    q_exp, curve_exp, sigmas_exp = load_and_preprocess_file(
        input_file,
        error_threshold=0.3,
        consecutive_threshold=3,
        remove_singles=True,
        remove_outliers_flag=True,
        smooth_flag=False
    )
    
    # Save preprocessed data
    save_preprocessed_data(q_exp, curve_exp, sigmas_exp, output_file)


if __name__ == "__main__":
    main()
