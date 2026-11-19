"""Validation and unweighted summaries of NF proposal samples."""

import numpy as np

from utils import to_numpy


def validate_nf_samples(samples, log_likelihoods, valid_mask=None):
    """Validate a posterior without losing original row indices."""
    samples = to_numpy(samples).astype(float, copy=False)
    log_likelihoods = to_numpy(log_likelihoods).astype(float, copy=False)
    if log_likelihoods.ndim != 1 or not log_likelihoods.size:
        raise ValueError("log_likelihoods must be a non-empty 1D array")
    # The vendor squeezes the sample dimension when num_samples=1.
    if samples.ndim == 1 and log_likelihoods.size == 1:
        samples = samples[None, :]
    if samples.ndim != 2 or not samples.shape[1]:
        raise ValueError("samples must be 2D (num_samples, num_params)")
    if samples.shape[0] != log_likelihoods.size:
        raise ValueError("log_likelihoods length must match the number of samples")
    finite = np.isfinite(log_likelihoods) & np.all(np.isfinite(samples), axis=1)
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask)
        if valid_mask.dtype != np.bool_ or valid_mask.shape != finite.shape:
            raise ValueError("valid_mask must be a boolean vector matching the samples")
        finite &= valid_mask
    if not finite.any():
        raise ValueError("posterior contains no finite sample/log-likelihood rows")
    return samples, log_likelihoods, finite


def select_global_ml_index(samples, log_likelihoods, valid_mask=None):
    """Return the original index of the finite global maximum-likelihood sample."""
    _, log_likelihoods, valid_mask = validate_nf_samples(
        samples, log_likelihoods, valid_mask
    )
    indices = np.flatnonzero(valid_mask)
    return int(indices[np.argmax(log_likelihoods[valid_mask])])


def compute_nf_sample_statistics(predicted_params_array, log_prob):
    """Summarize finite proposal draws; singleton sample standard deviations are null."""
    samples, log_prob, valid = validate_nf_samples(predicted_params_array, log_prob)
    samples, log_prob = samples[valid], log_prob[valid]
    multiple = len(samples) > 1
    return {
        "nf_params_mean": np.mean(samples, axis=0),
        "nf_params_std": np.std(samples, axis=0, ddof=1) if multiple else None,
        "nf_params_percentiles": np.percentile(samples, [5, 25, 50, 75, 95], axis=0),
        "nf_log_prob_mean": float(np.mean(log_prob)),
        "nf_log_prob_std": float(np.std(log_prob, ddof=1)) if multiple else None,
        "nf_statistics_kind": "unweighted_proposal",
        "nf_valid_sample_count": int(valid.sum()),
        "nf_invalid_sample_count": int((~valid).sum()),
    }
