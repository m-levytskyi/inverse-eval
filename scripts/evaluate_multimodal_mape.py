#!/usr/bin/env python3
"""Batch comparison for current ML and multimodal posterior MAPE."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/inverse-eval-matplotlib-cache")
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from reflectorch import EasyInferenceModel

from device_utils import detect_torch_device
from error_calculation import calculate_parameter_metrics
from multimodal_posterior_mape import (
    select_current_ml_sample,
    select_multimodal_ml_sample,
)
from parameter_discovery import (
    discover_experiment_files,
    get_prior_bounds_for_experiment,
    parse_true_parameters_from_model_file,
)
from simple_pipeline import (
    _extend_prior_bounds_for_nf,
    _map_pred_param_name_to_canonical_true_name,
    load_experimental_data,
)


DEFAULT_BATCH_JSON = (
    "batch_results_siddhartha_raw_full/"
    "001_Noneexps_1layers_30constraint_12august2026_13_47/"
    "batch_results.json"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", default="raw")
    parser.add_argument("--data-directory", default="siddhartha_data/1_new")
    parser.add_argument("--saved-batch-json", default=DEFAULT_BATCH_JSON)
    parser.add_argument("--output-dir", default="multimodal_eval_results")
    parser.add_argument("--config-name", default="example_nf_config_reflectorch.yaml")
    parser.add_argument("--nf-num-samples", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--k-values", default="1,2,3,4,5")
    parser.add_argument("--use-theoretical", action="store_true")
    return parser.parse_args()


def to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def quiet_call(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def success_ids(saved_batch_json: Path, n: int, seed: int, use_all: bool):
    data = json.loads(saved_batch_json.read_text())
    ids = [exp_id for exp_id, row in data.items() if row.get("success")]
    if use_all:
        return ids, data
    rng = random.Random(seed)
    rng.shuffle(ids)
    return ids[:n], data


def align_samples_to_true(samples, pred_param_names, true_param_names, layer_count):
    canonical_to_index = {}
    for index, pred_name in enumerate(pred_param_names):
        canonical = _map_pred_param_name_to_canonical_true_name(pred_name, layer_count)
        if canonical in true_param_names and canonical not in canonical_to_index:
            canonical_to_index[canonical] = index

    missing = [name for name in true_param_names if name not in canonical_to_index]
    if missing:
        raise ValueError(f"Predicted posterior is missing true parameters: {missing}")

    indices = [canonical_to_index[name] for name in true_param_names]
    return np.asarray(samples, dtype=float)[:, indices]


def parameter_metrics(pred_params, true_params, true_param_names, prior_bounds):
    return quiet_call(
        calculate_parameter_metrics,
        pred_params,
        true_params,
        true_param_names,
        prior_bounds=prior_bounds,
        priors_type="constraint_based",
    )


def evaluate_one(
    experiment_id,
    inference_model,
    data_directory,
    saved_batch,
    k_values,
    nf_num_samples,
    use_theoretical,
):
    data_file, model_file, detected_layer_count = quiet_call(
        discover_experiment_files,
        experiment_id,
        str(data_directory),
        1,
        use_theoretical=use_theoretical,
    )
    if not data_file or not model_file:
        raise FileNotFoundError(f"Could not find data/model files for {experiment_id}")

    layer_count = detected_layer_count or 1
    q_exp, curve_exp, sigmas_exp = quiet_call(
        load_experimental_data,
        data_file,
        enable_preprocessing=True,
        threshold=0.75,
        consecutive=5,
        remove_singles=False,
    )
    true_params_dict = quiet_call(
        parse_true_parameters_from_model_file, str(model_file)
    )
    true_block = true_params_dict[f"{layer_count}_layer"]
    true_params = np.asarray(true_block["params"], dtype=float)
    true_param_names = list(true_block["param_names"])
    prior_bounds = quiet_call(
        get_prior_bounds_for_experiment,
        experiment_id,
        true_params_dict,
        priors_type="constraint_based",
        deviation=0.30,
        layer_count=layer_count,
        fix_sld_mode="none",
    )

    q_model, curve_interp = quiet_call(
        inference_model.interpolate_data_to_model_q, q_exp, curve_exp
    )
    posterior = quiet_call(
        inference_model.preprocess_and_sample,
        reflectivity_curve=curve_interp,
        q_values=q_model,
        num_samples=nf_num_samples,
        prior_bounds=_extend_prior_bounds_for_nf(prior_bounds),
        q_resolution=0.1,
        calc_sampled_curves=False,
        calc_sampled_sld_profiles=False,
        calc_log_likelihoods=True,
        enable_importance_sampling=True,
        clip_prediction=True,
    )

    samples = align_samples_to_true(
        to_numpy(posterior["predicted_params_array"]).astype(float),
        list(posterior["param_names"]),
        true_param_names,
        layer_count,
    )
    log_likelihoods = to_numpy(posterior["log_likelihoods"]).astype(float)

    current = select_current_ml_sample(
        samples,
        log_likelihoods,
        true_params,
        true_param_names,
    )
    multimodal = select_multimodal_ml_sample(
        samples,
        log_likelihoods,
        true_params,
        true_param_names,
        k_values=k_values,
        random_state=0,
    )
    current_metrics = parameter_metrics(
        current["selected_params"], true_params, true_param_names, prior_bounds
    )
    multimodal_metrics = parameter_metrics(
        multimodal["selected_params"], true_params, true_param_names, prior_bounds
    )

    saved_overall = (
        saved_batch.get(experiment_id, {}).get("param_metrics", {}).get("overall", {})
    )
    return {
        "experiment_id": experiment_id,
        "saved_current_constraint_mape": saved_overall.get("constraint_mape"),
        "current_constraint_mape": current_metrics["overall"]["constraint_mape"],
        "multimodal_constraint_mape": multimodal_metrics["overall"]["constraint_mape"],
        "constraint_mape_delta": multimodal_metrics["overall"]["constraint_mape"]
        - current_metrics["overall"]["constraint_mape"],
        "current_mape": current_metrics["overall"]["mape"],
        "multimodal_mape": multimodal_metrics["overall"]["mape"],
        "mape_delta": multimodal_metrics["overall"]["mape"]
        - current_metrics["overall"]["mape"],
        "selected_k": multimodal["selected_k"],
        "selected_silhouette": multimodal["selected_silhouette"],
        "num_modes": len(multimodal["mode_rows"]),
        "current_sample_index": current["selected_index"],
        "multimodal_sample_index": multimodal["selected_index"],
        "same_sample": current["selected_index"] == multimodal["selected_index"],
        "current_log_likelihood": current["selected_log_likelihood"],
        "multimodal_log_likelihood": multimodal["selected_log_likelihood"],
    }


def summarize(rows):
    deltas = np.array([row["constraint_mape_delta"] for row in rows], dtype=float)
    current = np.array([row["current_constraint_mape"] for row in rows], dtype=float)
    multimodal = np.array(
        [row["multimodal_constraint_mape"] for row in rows], dtype=float
    )
    current_mape = np.array([row["current_mape"] for row in rows], dtype=float)
    multimodal_mape = np.array([row["multimodal_mape"] for row in rows], dtype=float)
    mape_deltas = np.array([row["mape_delta"] for row in rows], dtype=float)
    selected_ks = {}
    for row in rows:
        selected_ks[str(row["selected_k"])] = (
            selected_ks.get(str(row["selected_k"]), 0) + 1
        )

    return {
        "count": len(rows),
        "current_constraint_mape_mean": float(np.mean(current)),
        "current_constraint_mape_median": float(np.median(current)),
        "multimodal_constraint_mape_mean": float(np.mean(multimodal)),
        "multimodal_constraint_mape_median": float(np.median(multimodal)),
        "constraint_mape_delta_mean": float(np.mean(deltas)),
        "constraint_mape_delta_median": float(np.median(deltas)),
        "constraint_mape_delta_sum": float(np.sum(deltas)),
        "current_mape_mean": float(np.mean(current_mape)),
        "current_mape_median": float(np.median(current_mape)),
        "multimodal_mape_mean": float(np.mean(multimodal_mape)),
        "multimodal_mape_median": float(np.median(multimodal_mape)),
        "mape_delta_mean": float(np.mean(mape_deltas)),
        "mape_delta_median": float(np.median(mape_deltas)),
        "improved_count": int(np.sum(deltas < -1e-9)),
        "same_count": int(np.sum(np.isclose(deltas, 0.0, atol=1e-9))),
        "worse_count": int(np.sum(deltas > 1e-9)),
        "same_sample_count": sum(row["same_sample"] for row in rows),
        "selected_k_counts": selected_ks,
    }


def write_outputs(output_dir: Path, rows, summary, config):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.csv"
    summary_path = output_dir / "summary.json"

    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_path.write_text(
        json.dumps({"config": config, "summary": summary, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    return rows_path, summary_path


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    warnings.filterwarnings(
        "ignore",
        message="Inputs to the softmax are not scaled down.*",
        category=UserWarning,
    )

    k_values = tuple(int(value.strip()) for value in args.k_values.split(","))
    experiment_ids, saved_batch = success_ids(
        Path(args.saved_batch_json), args.n, args.seed, args.all
    )
    device = detect_torch_device(args.device)
    model = EasyInferenceModel(config_name=args.config_name, device=device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir) / f"{args.label}_{len(experiment_ids)}_{timestamp}"
    )

    print("Config:")
    sample_mode = "all successful cases" if args.all else "random successful cases"
    print(f"  experiments: {len(experiment_ids)} {sample_mode}")
    print(f"  label: {args.label}")
    print(f"  seed: {args.seed}")
    print(f"  data_directory: {args.data_directory}")
    print(f"  saved_batch_json: {args.saved_batch_json}")
    print(f"  config_name: {args.config_name}")
    print(f"  nf_num_samples: {args.nf_num_samples}")
    print(f"  k_values: {k_values}")
    print(f"  device: {device}")
    print(f"  output_dir: {output_dir}")

    rows = []
    start = time.time()
    for index, experiment_id in enumerate(experiment_ids, start=1):
        row_start = time.time()
        row = evaluate_one(
            experiment_id,
            model,
            Path(args.data_directory),
            saved_batch,
            k_values,
            args.nf_num_samples,
            args.use_theoretical,
        )
        rows.append(row)
        print(
            f"[{index:03d}/{len(experiment_ids)}] {experiment_id} "
            f"current={row['current_constraint_mape']:.3f} "
            f"multi={row['multimodal_constraint_mape']:.3f} "
            f"delta={row['constraint_mape_delta']:+.3f} "
            f"k={row['selected_k']} "
            f"{time.time() - row_start:.1f}s"
        )

    summary = summarize(rows)
    config = {
        "n": len(experiment_ids),
        "label": args.label,
        "all": args.all,
        "seed": args.seed,
        "data_directory": args.data_directory,
        "saved_batch_json": args.saved_batch_json,
        "config_name": args.config_name,
        "nf_num_samples": args.nf_num_samples,
        "k_values": k_values,
        "device": device,
        "total_seconds": time.time() - start,
    }
    rows_path, summary_path = write_outputs(output_dir, rows, summary, config)

    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"Rows: {rows_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
