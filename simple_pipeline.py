#!/usr/bin/env python3
"""
Single experiment processing pipeline for reflectometry analysis.

This module contains functions to process individual experiments, focusing on
the core workflow for analyzing a single reflectometry experiment.
"""

import torch
import numpy as np
import logging
from pathlib import Path
from typing import Literal
from reflectorch import EasyInferenceModel
from device_utils import detect_torch_device

# Import our modular utilities
from plotting_utils import plot_simple_comparison
from parameter_discovery import (
    discover_experiment_files,
    parse_true_parameters_from_model_file,
    generate_true_sld_profile,
    get_prior_bounds_for_experiment,
    get_parameter_names_for_layer_count,
)
from error_calculation import (
    calculate_parameter_metrics,
    print_metrics_report,
)
from data_preprocessing import preprocess_experimental_data
from parameter_constraints import apply_physical_constraints
from nf_statistics import (
    compute_nf_sample_statistics,
    select_global_ml_index,
    validate_nf_samples,
)
from utils import to_numpy
from constraints_utils import get_constraint_ranges

# Set seed for reproducibility
torch.manual_seed(42)


logger = logging.getLogger(__name__)


def load_experimental_data(
    data_file_path,
    enable_preprocessing=True,
    threshold=0.5,
    consecutive=3,
    remove_singles=False,
):
    """Load and parse experimental data from file with optional preprocessing."""
    logger.info(f"Loading experimental data from: {data_file_path}")

    data = np.loadtxt(data_file_path, skiprows=1)
    logger.info(f"Data shape: {data.shape}")

    q_exp = data[..., 0]
    curve_exp = data[..., 1]

    # Handle different data formats
    if data.shape[1] == 2:
        # Simple 2-column data (Q, R): create minimal dummy error bars
        sigmas_exp = np.full_like(curve_exp, 1e-6)
        logger.info("Detected 2-column data (Q, R) - using minimal dummy errors")
    elif data.shape[1] == 3:
        # Theoretical data: create minimal dummy error bars
        sigmas_exp = np.full_like(curve_exp, 1e-6)
        logger.info(
            "Detected theoretical data (3 columns) - using minimal dummy errors"
        )
    else:
        # Experimental data: use actual error bars
        sigmas_exp = data[..., 2]
        logger.info("Detected experimental data (4 columns) - using actual errors")

    logger.info(f"Raw Q range: {q_exp.min():.4f} - {q_exp.max():.4f} Å⁻¹")
    logger.info(f"Raw curve shape: {curve_exp.shape}")
    logger.info(
        f"Raw relative error range: {(sigmas_exp / curve_exp).min():.4f} - {(sigmas_exp / curve_exp).max():.4f}"
    )

    # Apply preprocessing
    q_exp, curve_exp, sigmas_exp = (
        preprocess_experimental_data(
            q_exp,
            curve_exp,
            sigmas_exp,
            error_threshold=threshold,
            consecutive_threshold=consecutive,
            remove_singles=remove_singles,
        )
        if enable_preprocessing
        else (q_exp, curve_exp, sigmas_exp)
    )

    logger.info(f"Final Q range: {q_exp.min():.4f} - {q_exp.max():.4f} Å⁻¹")
    logger.info(f"Final curve shape: {curve_exp.shape}")
    logger.info(
        f"Final relative error range: {(sigmas_exp / curve_exp).min():.4f} - {(sigmas_exp / curve_exp).max():.4f}"
    )

    return q_exp, curve_exp, sigmas_exp


def run_inference(
    inference_model,
    q_exp,
    curve_exp,
    prior_bounds,
    q_resolution=0.1,
    apply_constraints=True,
    clip_prediction=False,
    use_q_shift=True,
    polish_prediction=True,
    sigmas_exp=None,
    ambient_sld=0.0,
):
    """Run the inference prediction in the absolute SLD coordinate system."""
    logger.info("Performing inference prediction...")

    # Interpolate data to model grid
    if sigmas_exp is not None:
        q_model, exp_curve_interp, sigmas_interp = (
            inference_model.interpolate_data_to_model_q(
                q_exp, curve_exp, sigmas_exp=sigmas_exp
            )
        )
        logger.info(f"Model Q range: {q_model.min():.4f} - {q_model.max():.4f} Å⁻¹")
        logger.info(f"Interpolated curve shape: {exp_curve_interp.shape}")
        logger.info(f"Interpolated sigmas shape: {sigmas_interp.shape}")
    else:
        q_model, exp_curve_interp = inference_model.interpolate_data_to_model_q(
            q_exp, curve_exp
        )
        sigmas_interp = None
        logger.info(f"Model Q range: {q_model.min():.4f} - {q_model.max():.4f} Å⁻¹")
        logger.info(f"Interpolated curve shape: {exp_curve_interp.shape}")

    # Perform prediction
    predict_kwargs = {
        "reflectivity_curve": exp_curve_interp,
        "prior_bounds": prior_bounds,
        "q_values": q_model,
        "q_resolution": q_resolution,
        "ambient_sld": ambient_sld,
        "kwargs_param_labels": {"number_top_to_bottom": True},
        "clip_prediction": clip_prediction,
        "polish_prediction": polish_prediction,
        "use_q_shift": use_q_shift,
        "calc_pred_curve": True,
        "calc_pred_sld_profile": True,
        "calc_polished_sld_profile": True,
    }
    if sigmas_interp is not None:
        predict_kwargs["sigmas"] = sigmas_interp

    prediction_dict = inference_model.predict(**predict_kwargs)

    # Apply physical constraints to prevent negative thickness/roughness (if enabled)
    if apply_constraints:
        logger.info("Applying physical constraints...")
        prediction_dict = apply_physical_constraints(prediction_dict)
    else:
        logger.info("Physical constraints disabled - skipping constraint application")

    return q_model, prediction_dict


def _nf_prediction_to_single_prediction_dict(nf_prediction_dict):
    """Adapt a validated finite global-ML sample into the single-prediction schema."""
    required = (
        "predicted_params_array",
        "log_likelihoods",
        "sampled_curves",
        "q_plot_pred",
        "sampled_sld_profiles",
        "sampled_sld_xaxis",
        "param_names",
    )
    missing = [name for name in required if nf_prediction_dict.get(name) is None]
    if missing:
        raise ValueError(f"NF prediction is missing required fields: {missing}")
    samples, log_likelihoods, valid = validate_nf_samples(
        nf_prediction_dict["predicted_params_array"],
        nf_prediction_dict["log_likelihoods"],
    )
    names = list(nf_prediction_dict["param_names"])
    if len(names) != samples.shape[1]:
        raise ValueError("NF parameter names must match the sample columns")
    q_axis = to_numpy(nf_prediction_dict["q_plot_pred"])
    z_axis = to_numpy(nf_prediction_dict["sampled_sld_xaxis"])
    arrays = {}
    for name, axis in (("sampled_curves", q_axis), ("sampled_sld_profiles", z_axis)):
        array = to_numpy(nf_prediction_dict[name])
        if array.ndim == 1 and len(samples) == 1:
            array = array[None, :]
        if array.ndim != 2 or array.shape[0] != len(samples) or not array.shape[1]:
            raise ValueError(f"NF {name} must be non-empty 2D with one row per sample")
        if axis.ndim == 1 and axis.size == array.shape[1]:
            valid &= np.all(np.isfinite(axis))
        elif name == "sampled_sld_profiles" and axis.shape == array.shape:
            valid &= np.all(np.isfinite(axis), axis=1)
        else:
            raise ValueError(f"NF {name} shape must match sample count and axis length")
        valid &= np.all(np.isfinite(array), axis=1)
        arrays[name] = array
    if not valid.any():
        raise ValueError("NF posterior contains no finite complete sample rows")
    best_idx = select_global_ml_index(samples, log_likelihoods, valid)
    return {
        "param_names": names,
        "predicted_params_array": samples[best_idx].copy(),
        "polished_params_array": samples[best_idx].copy(),
        "q_plot_pred": q_axis,
        "predicted_curve": arrays["sampled_curves"][best_idx],
        "polished_curve": arrays["sampled_curves"][best_idx],
        "predicted_sld_xaxis": z_axis[best_idx] if z_axis.ndim == 2 else z_axis,
        "predicted_sld_profile": arrays["sampled_sld_profiles"][best_idx],
        "sld_profile_polished": arrays["sampled_sld_profiles"][best_idx],
        "nf_best_idx": best_idx,
        "nf_best_log_likelihood": float(log_likelihoods[best_idx]),
    }


def _extend_prior_bounds_for_nf(prior_bounds, inference_model=None):
    """Append selected-model nuisance bounds; retain the historical no-model call."""
    bounds = list(prior_bounds)
    if inference_model is None:
        if len(bounds) in (5, 8):
            return bounds + [(0.9, 1.1), (-10.0, -4.0)]
        if len(bounds) in (7, 10):
            return bounds
        raise ValueError(f"Unsupported NF prior_bounds length={len(bounds)}")
    sampler = inference_model.trainer.loader.prior_sampler
    parameter_count = sampler.param_dim
    physical_count = 3 * sampler.max_num_layers + 2
    if len(bounds) == parameter_count:
        return bounds
    if len(bounds) != physical_count or parameter_count < physical_count:
        raise ValueError("Prior bounds do not match the selected model parameter count")
    lower = to_numpy(sampler.min_bounds).reshape(-1)
    upper = to_numpy(sampler.max_bounds).reshape(-1)
    return bounds + list(zip(lower[physical_count:], upper[physical_count:]))


def _map_pred_param_name_to_canonical_true_name(
    pred_param_name: str, layer_count: int
) -> str:
    """Map reflectorch param labels to this repo's canonical parameter names."""
    n = (pred_param_name or "").strip().lower()
    if not n:
        return ""

    # Nuisance parameters used by NF configs
    if n in {"r_scale", "log10_background", "q_shift"}:
        return n

    # Normalize common formatting
    n = n.replace("-", " ")
    n = " ".join(n.split())

    if "thickness" in n:
        if layer_count == 1:
            return "thickness"
        if "l1" in n or "layer1" in n:
            return "thickness1"
        if "l2" in n or "layer2" in n:
            return "thickness2"

    if "rough" in n:
        if "sub" in n:
            return "sub_rough"
        if layer_count == 1:
            if "l1" in n or "layer1" in n:
                return "amb_rough"
        if layer_count == 2:
            if "l1" in n or "layer1" in n:
                return "amb_rough"
            if "l2" in n or "layer2" in n:
                return "int_rough"

    if "sld" in n:
        if "sub" in n:
            return "sub_sld"
        if layer_count == 1:
            if "l1" in n or "layer1" in n:
                return "layer_sld"
        if layer_count == 2:
            if "l1" in n or "layer1" in n:
                return "layer1_sld"
            if "l2" in n or "layer2" in n:
                return "layer2_sld"

    return n.replace(" ", "_")


def run_nf_inference(
    inference_model,
    q_exp,
    curve_exp,
    prior_bounds,
    q_resolution=0.1,
    apply_constraints=True,
    nf_num_samples=1000,
    nf_enable_importance_sampling=True,
    clip_prediction=True,
    sigmas_exp=None,
    ambient_sld=0.0,
):
    """Sample once on the vendor grid, retaining the legacy likelihood noise model."""
    if not isinstance(nf_num_samples, (int, np.integer)) or nf_num_samples < 1:
        raise ValueError("nf_num_samples must be a positive integer")
    posterior = inference_model.preprocess_and_sample(
        reflectivity_curve=curve_exp,
        q_values=q_exp,
        sigmas=sigmas_exp,
        num_samples=nf_num_samples,
        prior_bounds=_extend_prior_bounds_for_nf(prior_bounds, inference_model),
        q_resolution=q_resolution,
        ambient_sld=ambient_sld,
        kwargs_param_labels={"number_top_to_bottom": True},
        calc_sampled_curves=True,
        calc_sampled_sld_profiles=True,
        calc_log_likelihoods=True,
        enable_importance_sampling=nf_enable_importance_sampling,
        clip_prediction=clip_prediction,
        enable_error_bars_filtering=False,
        rel_err_factor=0.2,
    )
    prediction = _nf_prediction_to_single_prediction_dict(posterior)
    prediction.update(
        compute_nf_sample_statistics(
            posterior["predicted_params_array"], posterior["log_likelihoods"]
        )
    )
    ess = posterior.get("ess")
    prediction["nf_ess"] = float(ess) if ess is not None and np.isfinite(ess) else None
    prediction["nf_likelihood_noise"] = (
        {"source": "provided_sigmas"}
        if sigmas_exp is not None
        else {"source": "relative", "relative_error": 0.2, "absolute_floor": 1e-12}
    )
    if apply_constraints:
        prediction = apply_physical_constraints(prediction)
    return np.asarray(prediction["q_plot_pred"]), prediction


def display_results(prediction_dict):
    """Display prediction results in a formatted way."""
    logger.info("\nPrediction Results:")
    logger.info("-" * 50)

    pred_params = prediction_dict["predicted_params_array"]
    polished_params = prediction_dict["polished_params_array"]
    param_names = prediction_dict["param_names"]

    for param_name, pred_val, polished_val in zip(
        param_names, pred_params, polished_params
    ):
        logger.info(
            f"{param_name.ljust(18)} -> Predicted: {pred_val:.3f}    Polished: {polished_val:.3f}"
        )


def resolve_inference_config(
    layer_count=1, inference_backend="predict", config_name=None
):
    """Resolve the supported default model without loading weights."""
    if layer_count not in (1, 2):
        raise ValueError(f"Unsupported layer count: {layer_count}")
    if inference_backend not in {"predict", "nf"}:
        raise ValueError(f"Unknown inference backend: {inference_backend}")
    if config_name is None:
        if inference_backend == "nf":
            if layer_count != 1:
                raise ValueError(
                    "Provide an explicit NF config matching the requested layer_count"
                )
            config_name = "example_nf_config_reflectorch.yaml"
        else:
            config_name = (
                f"b_mc_point_neutron_conv_standard_L{layer_count}_InputQDq.yaml"
            )
    return config_name if config_name.endswith(".yaml") else config_name + ".yaml"


def _validate_inference_model(inference_model, layer_count, config_name=None):
    sampler = inference_model.trainer.loader.prior_sampler
    if sampler.max_num_layers != layer_count:
        raise ValueError(
            f"Model has {sampler.max_num_layers} layers but experiment has {layer_count}"
        )
    if config_name is not None:
        actual = getattr(inference_model, "config_name", None)
        expected = config_name.removesuffix(".yaml")
        if actual is not None and actual.removesuffix(".yaml") != expected:
            raise ValueError(
                f"Injected model config {actual} does not match {config_name}"
            )
    return sampler


def create_inference_model(
    layer_count=1, inference_backend="predict", config_name=None, inference_device=None
):
    """Create one model that can be reused across compatible experiments."""
    config_name = resolve_inference_config(layer_count, inference_backend, config_name)
    model = EasyInferenceModel(
        config_name=config_name, device=detect_torch_device(inference_device)
    )
    _validate_inference_model(model, layer_count, config_name)
    return model


def get_inference_prior_limits(inference_model, layer_count, ambient_sld=0.0):
    """Expose selected-model limits in absolute coordinates and canonical names."""
    if not np.isfinite(ambient_sld):
        raise ValueError("ambient_sld must be finite")
    sampler = _validate_inference_model(inference_model, layer_count)
    labels = sampler.param_model.get_param_labels(number_top_to_bottom=True)
    names = [
        _map_pred_param_name_to_canonical_true_name(name, layer_count)
        for name in labels
    ]
    required = get_parameter_names_for_layer_count(layer_count)
    if names[: len(required)] != required:
        raise ValueError(
            "Selected model parameter order does not match the supported layer model"
        )
    arrays = [
        to_numpy(value).reshape(-1)
        for value in (
            sampler.min_bounds,
            sampler.max_bounds,
            sampler.min_delta,
            sampler.max_delta,
        )
    ]
    if any(len(values) != len(names) for values in arrays):
        raise ValueError("Selected model limits do not match its parameter labels")
    ranges, widths = {}, {}
    for index, name in enumerate(required):
        lower, upper, minimum, maximum = [float(values[index]) for values in arrays]
        shift = ambient_sld if "sld" in name else 0.0
        ranges[name] = (lower + shift, upper + shift)
        widths[name] = (minimum, maximum)
    return ranges, widths


def align_samples_to_true(samples, pred_param_names, true_param_names, layer_count):
    """Align physical parameters by name while preserving all sample row indices."""
    samples = to_numpy(samples)
    if samples.ndim != 2 or samples.shape[1] != len(pred_param_names):
        raise ValueError("Sample columns must match predicted parameter names")
    canonical_to_index = {}
    for index, name in enumerate(pred_param_names):
        canonical = _map_pred_param_name_to_canonical_true_name(name, layer_count)
        if canonical in canonical_to_index:
            raise ValueError(f"Duplicate predicted parameter name: {canonical}")
        canonical_to_index[canonical] = index
    missing = [name for name in true_param_names if name not in canonical_to_index]
    if missing:
        raise ValueError(f"Predicted posterior is missing true parameters: {missing}")
    return samples[:, [canonical_to_index[name] for name in true_param_names]]


def run_single_experiment(
    experiment_id,
    layer_count=1,
    enable_preprocessing=True,
    preprocessing_threshold=0.5,
    preprocessing_consecutive=3,
    preprocessing_remove_singles=False,
    apply_constraints=True,
    priors_type="constraint_based",
    priors_deviation=0.5,
    fix_sld_mode="none",
    use_theoretical=False,
    inference_backend: Literal["predict", "nf"] = "predict",
    config_name: str | None = None,
    nf_num_samples: int = 1000,
    nf_enable_importance_sampling: bool = True,
    clip_prediction: bool = True,
    use_sigmas_input: bool = False,
    inference_device: str | None = None,
    data_directory: str = "data",
    inference_model=None,
):
    """
    Run a single experiment inference with configurable options.

    Args:
        experiment_id: ID of the experiment to analyze
        layer_count: Number of layers (1 or 2)
        enable_preprocessing: Whether to enable data preprocessing
        preprocessing_threshold: Error threshold for preprocessing
        preprocessing_consecutive: Consecutive points threshold
        preprocessing_remove_singles: Remove isolated high-error points
        apply_constraints: Whether to apply physical constraints to parameters
        priors_type: Type of priors to use ("narrow" or "constraint_based")
        priors_deviation: Deviation for narrow priors (e.g., 0.3 for 30%) or
                         constraint percentage for constraint_based priors
        fix_sld_mode: SLD fixing mode - "none", "backing", or "all"
        use_theoretical: If True, use theoretical curves; if False (default), use experimental curves
        clip_prediction: Whether to clip predicted parameters to prior bounds (default: True)
        use_sigmas_input: Provide loaded uncertainties for likelihood/polishing;
                          the selected model controls whether they also enter the network.
        inference_device: Optional explicit torch device ("cpu", "cuda", or "mps").
                         If None, automatically prefers CUDA, then MPS, then CPU.
        data_directory: Base data directory to search for experiment files (default: "data")
        inference_model: Optional compatible model to reuse across experiments.

    Returns:
        Dictionary with results including parameters and metrics
    """
    # Discover experiment files
    data_file, model_file, detected_layer_count = discover_experiment_files(
        experiment_id, data_directory, layer_count, use_theoretical=use_theoretical
    )
    if not data_file:
        raise FileNotFoundError(f"Could not find data file for {experiment_id}")

    # Use detected layer count if available
    final_layer_count = detected_layer_count if detected_layer_count else layer_count

    # Load experimental data with preprocessing
    q_exp, curve_exp, sigmas_exp = load_experimental_data(
        data_file,
        enable_preprocessing=enable_preprocessing,
        threshold=preprocessing_threshold,
        consecutive=preprocessing_consecutive,
        remove_singles=preprocessing_remove_singles,
    )

    # Load true parameters if available
    true_params_dict = None
    if model_file:
        true_params_dict = parse_true_parameters_from_model_file(str(model_file))

    if inference_backend not in {"predict", "nf"}:
        raise ValueError(f"Unknown inference backend: {inference_backend}")
    if inference_model is None:
        inference_model = create_inference_model(
            final_layer_count, inference_backend, config_name, inference_device
        )
    else:
        _validate_inference_model(inference_model, final_layer_count, config_name)
    resolved_device = str(inference_model.device)
    if inference_device is not None and resolved_device != str(
        detect_torch_device(inference_device)
    ):
        raise ValueError(
            "Injected model device does not match the requested inference_device"
        )
    true_block = (true_params_dict or {}).get(f"{final_layer_count}_layer", {})
    ambient_sld = float(true_block.get("ambient_sld", 0.0))
    model_ranges, width_limits = get_inference_prior_limits(
        inference_model, final_layer_count, ambient_sld
    )
    prior_bounds = get_prior_bounds_for_experiment(
        experiment_id,
        true_params_dict,
        priors_type=priors_type,
        deviation=priors_deviation,
        layer_count=final_layer_count,
        fix_sld_mode=fix_sld_mode,
        model_ranges=model_ranges,
        width_limits=width_limits,
    )

    # Run inference
    sigmas_for_inference = sigmas_exp if use_sigmas_input else None

    if inference_backend == "nf":
        q_model, prediction_dict = run_nf_inference(
            inference_model,
            q_exp,
            curve_exp,
            prior_bounds,
            q_resolution=0.1,
            apply_constraints=apply_constraints,
            nf_num_samples=nf_num_samples,
            nf_enable_importance_sampling=nf_enable_importance_sampling,
            clip_prediction=clip_prediction,
            sigmas_exp=sigmas_for_inference,
            ambient_sld=ambient_sld,
        )
    else:
        q_model, prediction_dict = run_inference(
            inference_model,
            q_exp,
            curve_exp,
            prior_bounds,
            q_resolution=0.1,
            apply_constraints=apply_constraints,
            clip_prediction=clip_prediction,
            sigmas_exp=sigmas_for_inference,
            ambient_sld=ambient_sld,
        )

    param_metrics = None
    if true_params_dict and f"{final_layer_count}_layer" in true_params_dict:
        true_param_block = true_params_dict[f"{final_layer_count}_layer"]
        true_params = true_param_block["params"]
        true_param_names = true_param_block["param_names"]

        pred_params_for_metrics = prediction_dict["polished_params_array"]
        pred_param_names = prediction_dict.get("param_names", [])

        pred_params_for_metrics = align_samples_to_true(
            np.asarray(pred_params_for_metrics)[None, :],
            pred_param_names,
            true_param_names,
            final_layer_count,
        )[0]

        param_metrics = calculate_parameter_metrics(
            pred_params_for_metrics,
            true_params,
            true_param_names,
        )

    # Prepare results dictionary
    results = {
        "experiment_id": experiment_id,
        "input_files": {
            "data_file": str(Path(data_file).resolve()),
            "model_file": str(Path(model_file).resolve()) if model_file else None,
        },
        "layer_count": final_layer_count,
        "prediction_dict": prediction_dict,
        "param_metrics": param_metrics,
        "true_params_dict": true_params_dict,
        "q_exp": q_exp,
        "curve_exp": curve_exp,
        "sigmas_exp": sigmas_exp,
        "q_model": q_model,
        "prior_bounds": prior_bounds,
        "ambient_sld": ambient_sld,
        "inference_device": resolved_device,
        "priors_config": {
            "priors_type": priors_type,
            "priors_deviation": priors_deviation,
            "fix_sld_mode": fix_sld_mode,
            "effective_widths": [upper - lower for lower, upper in prior_bounds],
            "requested_widths": [
                0.002
                if (name == "sub_sld" and fix_sld_mode != "none")
                or ("sld" in name and fix_sld_mode == "all")
                else priors_deviation
                * (
                    get_constraint_ranges()[name][1] - get_constraint_ranges()[name][0]
                    if priors_type == "constraint_based"
                    else 2 * abs(value)
                )
                for name, value in zip(true_block["param_names"], true_block["params"])
            ],
        },
    }

    return results


def main():
    """Main function to run a single experiment with specific settings."""
    # Experiment configuration
    experiment_name = "s007384"
    layer_count = 1

    logger.info(f"Running inference for experiment: {experiment_name}")
    logger.info(f"Layer count: {layer_count}")
    logger.info("Running with 99% constraint-based priors and preprocessing OFF.")
    logger.info("=" * 60)

    # Run the experiment with specified settings
    results = run_single_experiment(
        experiment_id=experiment_name,
        layer_count=layer_count,
        enable_preprocessing=True,
        priors_type="constraint_based",
        priors_deviation=0.99,  # 99% constraint
        use_theoretical=False,
    )

    # Unpack results for clarity
    prediction_dict = results["prediction_dict"]
    param_metrics = results["param_metrics"]
    true_params_dict = results["true_params_dict"]
    q_exp = results["q_exp"]
    curve_exp = results["curve_exp"]
    sigmas_exp = results["sigmas_exp"]
    q_model = results["q_model"]

    # Display results
    display_results(prediction_dict)

    # Print metrics report
    print_metrics_report(param_metrics)

    # Generate true SLD profile for plotting
    if true_params_dict:
        generate_true_sld_profile(true_params_dict)

    # Create plots
    plot_simple_comparison(
        q_exp,
        curve_exp,
        sigmas_exp,
        q_model,
        prediction_dict["predicted_curve"],
        prediction_dict["polished_curve"],
        prediction_dict["predicted_sld_xaxis"],
        prediction_dict["predicted_sld_profile"],
        prediction_dict["sld_profile_polished"],
        experiment_name=experiment_name,
        show=True,
    )

    logger.info(f"\nInference completed for experiment {experiment_name}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Optionally run the main function
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--run-example":
        main()
    else:
        logger.info("\nUse --run-example to run the built-in example")
