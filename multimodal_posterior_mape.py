"""Utilities for comparing single-ML and multimodal posterior MAPE."""

from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import silhouette_score

from constraints_utils import get_constraint_width


def _as_2d_samples(samples) -> np.ndarray:
    samples_array = np.asarray(samples, dtype=float)
    if samples_array.ndim != 2:
        raise ValueError(
            f"samples must be 2D (num_samples, num_params), got {samples_array.shape}"
        )
    if samples_array.shape[0] == 0 or samples_array.shape[1] == 0:
        raise ValueError("samples must contain at least one sample and one parameter")
    return samples_array


def _as_1d_log_likelihoods(log_likelihoods, num_samples: int) -> np.ndarray:
    log_likelihoods_array = np.asarray(log_likelihoods, dtype=float)
    if log_likelihoods_array.ndim != 1:
        raise ValueError(
            f"log_likelihoods must be 1D, got {log_likelihoods_array.shape}"
        )
    if log_likelihoods_array.shape[0] != num_samples:
        raise ValueError(
            "log_likelihoods length must match samples, "
            f"got {log_likelihoods_array.shape[0]} and {num_samples}"
        )
    return log_likelihoods_array


def _constraint_widths(param_names: Iterable[str]) -> np.ndarray:
    widths = np.array([get_constraint_width(name) for name in param_names], dtype=float)
    if np.any(widths <= 0):
        raise ValueError(f"constraint widths must be positive, got {widths}")
    return widths


def _valid_rows(samples: np.ndarray, log_likelihoods: np.ndarray) -> np.ndarray:
    return np.isfinite(log_likelihoods) & np.all(np.isfinite(samples), axis=1)


def constraint_mape(pred_params, true_params, param_names: Iterable[str]) -> float:
    """Mean absolute error normalized by model constraint width, in percent."""
    pred = np.asarray(pred_params, dtype=float)
    true = np.asarray(true_params, dtype=float)
    if pred.shape != true.shape:
        raise ValueError(
            f"pred_params and true_params shape mismatch: {pred.shape} vs {true.shape}"
        )

    widths = _constraint_widths(param_names)
    if widths.shape[0] != pred.shape[0]:
        raise ValueError(
            f"param_names length ({widths.shape[0]}) must match params ({pred.shape[0]})"
        )
    return float(np.mean(np.abs(pred - true) / widths * 100.0))


def select_current_ml_sample(samples, log_likelihoods, true_params, param_names):
    """Select the same single point estimate as the current NF pipeline."""
    samples_array = _as_2d_samples(samples)
    log_likelihoods_array = _as_1d_log_likelihoods(
        log_likelihoods, samples_array.shape[0]
    )
    valid_mask = _valid_rows(samples_array, log_likelihoods_array)
    if not np.any(valid_mask):
        raise ValueError("posterior contains no finite sample/log-likelihood rows")

    valid_indices = np.flatnonzero(valid_mask)
    selected_index = int(valid_indices[np.argmax(log_likelihoods_array[valid_mask])])
    selected_params = samples_array[selected_index]

    return {
        "selected_index": selected_index,
        "selected_params": selected_params,
        "selected_log_likelihood": float(log_likelihoods_array[selected_index]),
        "constraint_mape": constraint_mape(selected_params, true_params, param_names),
    }


def _cluster_features(samples: np.ndarray, param_names: Iterable[str]) -> np.ndarray:
    return samples / _constraint_widths(param_names)


def _try_kmeans(features: np.ndarray, k: int, random_state: int):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(
            features
        )

    unique_labels = np.unique(labels)
    if unique_labels.size < 2 or unique_labels.size >= features.shape[0]:
        return labels, None, "silhouette undefined for this label count"

    score = float(silhouette_score(features, labels))
    return labels, score, None


def select_multimodal_ml_sample(
    samples,
    log_likelihoods,
    true_params,
    param_names,
    k_values=(1, 2, 3, 4, 5),
    random_state: int = 0,
):
    """Select the closest local-ML posterior mode after k-means clustering.

    `k=1` is treated as the current global ML baseline with silhouette 0.0,
    because the sklearn silhouette score is undefined for one cluster.
    """
    samples_array = _as_2d_samples(samples)
    log_likelihoods_array = _as_1d_log_likelihoods(
        log_likelihoods, samples_array.shape[0]
    )
    valid_mask = _valid_rows(samples_array, log_likelihoods_array)
    if not np.any(valid_mask):
        raise ValueError("posterior contains no finite sample/log-likelihood rows")

    valid_indices = np.flatnonzero(valid_mask)
    valid_samples = samples_array[valid_mask]
    valid_log_likelihoods = log_likelihoods_array[valid_mask]
    features = _cluster_features(valid_samples, param_names)

    k_scores = []
    best_labels = np.zeros(valid_samples.shape[0], dtype=int)
    best_score = float("-inf")
    selected_k = 1

    for raw_k in k_values:
        k = int(raw_k)
        if k < 1:
            k_scores.append(
                {
                    "k": k,
                    "silhouette": None,
                    "usable": False,
                    "reason": "k must be >= 1",
                }
            )
            continue

        if k == 1:
            score = 0.0
            labels = np.zeros(valid_samples.shape[0], dtype=int)
            reason = None
        elif k >= valid_samples.shape[0]:
            score = None
            labels = None
            reason = "k must be smaller than the number of finite samples"
        else:
            labels, score, reason = _try_kmeans(features, k, random_state)

        usable = labels is not None and score is not None
        k_scores.append(
            {"k": k, "silhouette": score, "usable": usable, "reason": reason}
        )
        if usable and score > best_score:
            best_score = score
            best_labels = labels
            selected_k = k

    mode_rows = []
    for cluster_label in sorted(np.unique(best_labels).tolist()):
        cluster_positions = np.flatnonzero(best_labels == cluster_label)
        local_position = int(
            cluster_positions[np.argmax(valid_log_likelihoods[cluster_positions])]
        )
        original_index = int(valid_indices[local_position])
        local_params = samples_array[original_index]
        mode_rows.append(
            {
                "cluster": int(cluster_label),
                "sample_index": original_index,
                "log_likelihood": float(log_likelihoods_array[original_index]),
                "constraint_mape": constraint_mape(
                    local_params, true_params, param_names
                ),
                "params": local_params,
            }
        )

    selected_mode = min(mode_rows, key=lambda row: row["constraint_mape"])

    labels_all = np.full(samples_array.shape[0], -1, dtype=int)
    labels_all[valid_indices] = best_labels

    return {
        "selected_k": selected_k,
        "selected_silhouette": best_score,
        "selected_cluster": selected_mode["cluster"],
        "selected_index": selected_mode["sample_index"],
        "selected_params": selected_mode["params"],
        "selected_log_likelihood": selected_mode["log_likelihood"],
        "selected_constraint_mape": selected_mode["constraint_mape"],
        "mode_rows": mode_rows,
        "k_scores": k_scores,
        "labels": labels_all,
    }
