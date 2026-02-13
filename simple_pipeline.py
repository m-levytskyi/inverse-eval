#!/usr/bin/env python3
"""
Single experiment processing pipeline for reflectometry analysis.

This module contains functions to process individual experiments, focusing on
the core workflow for analyzing a single reflectometry experiment.
"""

import torch
import numpy as np
from typing import Literal
from reflectorch import EasyInferenceModel

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
    calculate_fit_metrics,
    calculate_parameter_metrics,
    print_metrics_report,
)
from data_preprocessing import preprocess_experimental_data
from parameter_constraints import apply_physical_constraints
from nf_statistics import compute_nf_sample_statistics

# Set seed for reproducibility
torch.manual_seed(42)


def load_experimental_data(
    data_file_path,
    enable_preprocessing=True,
    threshold=0.5,
    consecutive=3,
    remove_singles=False,
):
    """Load and parse experimental data from file with optional preprocessing."""
    print(f"Loading experimental data from: {data_file_path}")

    data = np.loadtxt(data_file_path, skiprows=1)
    print(f"Data shape: {data.shape}")

    q_exp = data[..., 0]
    curve_exp = data[..., 1]

    # Handle different data formats
    if data.shape[1] == 2:
        # Simple 2-column data (Q, R): create minimal dummy error bars
        sigmas_exp = np.full_like(curve_exp, 1e-6)
        print("Detected 2-column data (Q, R) - using minimal dummy errors")
    elif data.shape[1] == 3:
        # Theoretical data: create minimal dummy error bars
        sigmas_exp = np.full_like(curve_exp, 1e-6)
        print("Detected theoretical data (3 columns) - using minimal dummy errors")
    else:
        # Experimental data: use actual error bars
        sigmas_exp = data[..., 2]
        print("Detected experimental data (4 columns) - using actual errors")

    print(f"Raw Q range: {q_exp.min():.4f} - {q_exp.max():.4f} Å⁻¹")
    print(f"Raw curve shape: {curve_exp.shape}")
    print(
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

    print(f"Final Q range: {q_exp.min():.4f} - {q_exp.max():.4f} Å⁻¹")
    print(f"Final curve shape: {curve_exp.shape}")
    print(
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
):
    """Run the inference prediction."""
    print("Performing inference prediction...")

    # Interpolate data to model grid
    if sigmas_exp is not None:
        q_model, exp_curve_interp, sigmas_interp = (
            inference_model.interpolate_data_to_model_q(
                q_exp, curve_exp, sigmas_exp=sigmas_exp
            )
        )
        print(f"Model Q range: {q_model.min():.4f} - {q_model.max():.4f} Å⁻¹")
        print(f"Interpolated curve shape: {exp_curve_interp.shape}")
        print(f"Interpolated sigmas shape: {sigmas_interp.shape}")
    else:
        q_model, exp_curve_interp = inference_model.interpolate_data_to_model_q(
            q_exp, curve_exp
        )
        sigmas_interp = None
        print(f"Model Q range: {q_model.min():.4f} - {q_model.max():.4f} Å⁻¹")
        print(f"Interpolated curve shape: {exp_curve_interp.shape}")

    # Perform prediction
    predict_kwargs = {
        "reflectivity_curve": exp_curve_interp,
        "prior_bounds": prior_bounds,
        "q_values": q_model,
        "q_resolution": q_resolution,
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
        print("Applying physical constraints...")
        prediction_dict = apply_physical_constraints(prediction_dict)
    else:
        print("Physical constraints disabled - skipping constraint application")

    return q_model, prediction_dict


def _nf_prediction_to_single_prediction_dict(nf_prediction_dict):
    """Adapt NF `preprocess_and_sample` output into the existing single-prediction schema.

    This selects the single best sample by max log-likelihood and maps it onto keys
    expected elsewhere in the pipeline.
    """
    if (
        "log_likelihoods" not in nf_prediction_dict
        or nf_prediction_dict["log_likelihoods"] is None
    ):
        raise ValueError(
            "NF prediction_dict is missing required 'log_likelihoods' (calc_log_likelihoods=True)"
        )

    log_likelihoods = np.asarray(nf_prediction_dict["log_likelihoods"])
    if log_likelihoods.ndim != 1 or log_likelihoods.size == 0:
        raise ValueError("NF 'log_likelihoods' must be a non-empty 1D array")

    best_idx = int(np.argmax(log_likelihoods))

    if "predicted_params_array" not in nf_prediction_dict:
        raise ValueError("NF prediction_dict is missing 'predicted_params_array'")
    pred_params_all = np.asarray(nf_prediction_dict["predicted_params_array"])
    if pred_params_all.ndim != 2:
        raise ValueError(
            "NF 'predicted_params_array' must be a 2D array (num_samples, num_params)"
        )
    pred_params = pred_params_all[best_idx]

    if "sampled_curves" not in nf_prediction_dict:
        raise ValueError(
            "NF prediction_dict is missing 'sampled_curves' (calc_sampled_curves=True)"
        )
    sampled_curves = np.asarray(nf_prediction_dict["sampled_curves"])
    if sampled_curves.ndim != 2:
        raise ValueError("NF 'sampled_curves' must be a 2D array (num_samples, n_q)")
    predicted_curve = sampled_curves[best_idx]

    if "q_plot_pred" not in nf_prediction_dict:
        raise ValueError("NF prediction_dict is missing 'q_plot_pred'")

    if "sampled_sld_profiles" not in nf_prediction_dict:
        raise ValueError(
            "NF prediction_dict is missing 'sampled_sld_profiles' (calc_sampled_sld_profiles=True)"
        )
    sampled_sld_profiles = np.asarray(nf_prediction_dict["sampled_sld_profiles"])
    if sampled_sld_profiles.ndim != 2:
        raise ValueError(
            "NF 'sampled_sld_profiles' must be a 2D array (num_samples, n_z)"
        )
    predicted_sld_profile = sampled_sld_profiles[best_idx]

    if "sampled_sld_xaxis" not in nf_prediction_dict:
        raise ValueError("NF prediction_dict is missing 'sampled_sld_xaxis'")
    predicted_sld_xaxis = np.asarray(nf_prediction_dict["sampled_sld_xaxis"])

    param_names = nf_prediction_dict.get("param_names")
    if param_names is None:
        raise ValueError("NF prediction_dict is missing 'param_names'")

    return {
        # parameters
        "param_names": param_names,
        "predicted_params_array": pred_params,
        "polished_params_array": pred_params,
        # curves
        "q_plot_pred": np.asarray(nf_prediction_dict["q_plot_pred"]),
        "predicted_curve": predicted_curve,
        "polished_curve": predicted_curve,
        # sld
        "predicted_sld_xaxis": predicted_sld_xaxis,
        "predicted_sld_profile": predicted_sld_profile,
        "sld_profile_polished": predicted_sld_profile,
        # minimal NF metadata
        "nf_best_idx": best_idx,
        "nf_best_log_likelihood": float(log_likelihoods[best_idx]),
    }


def _extend_prior_bounds_for_nf(prior_bounds):
    """Extend standard prior bounds with NF nuisance-parameter bounds when needed.

    The example NF config expects two additional parameters beyond the standard
    geometry/SLD parameters: r_scale and log10_background.
    """
    prior_bounds_list = list(prior_bounds)

    if len(prior_bounds_list) in (5, 8):
        # Append nuisance bounds in the expected order.
        # r_scale: [0.9, 1.1]
        # log10_background: [-10, -4]
        prior_bounds_list.extend([(0.9, 1.1), (-10.0, -4.0)])
        return prior_bounds_list

    if len(prior_bounds_list) in (7, 10):
        return prior_bounds_list

    raise ValueError(
        f"Unsupported NF prior_bounds length={len(prior_bounds_list)}. "
        "Expected 5/7 (1-layer) or 8/10 (2-layer), depending on whether nuisance "
        "parameters (r_scale, log10_background) are included."
    )


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
):
    """Run NF inference via `preprocess_and_sample` and adapt to the standard schema."""
    print("Performing NF inference (preprocess_and_sample)...")

    if sigmas_exp is not None:
        q_model, exp_curve_interp, sigmas_interp = (
            inference_model.interpolate_data_to_model_q(
                q_exp, curve_exp, sigmas_exp=sigmas_exp
            )
        )
        print(f"Model Q range: {q_model.min():.4f} - {q_model.max():.4f} Å⁻¹")
        if q_model.max() < 0.05:
            print(
                f"⚠️  WARNING: Model Q max ({q_model.max():.4f}) is very low! Inference may be unreliable."
            )
        print(f"Interpolated curve shape: {exp_curve_interp.shape}")
        print(f"Interpolated sigmas shape: {sigmas_interp.shape}")
    else:
        q_model, exp_curve_interp = inference_model.interpolate_data_to_model_q(
            q_exp, curve_exp
        )
        sigmas_interp = None
        print(f"Model Q range: {q_model.min():.4f} - {q_model.max():.4f} Å⁻¹")
        if q_model.max() < 0.05:
            print(
                f"⚠️  WARNING: Model Q max ({q_model.max():.4f}) is very low! Inference may be unreliable."
            )
        print(f"Interpolated curve shape: {exp_curve_interp.shape}")

    preprocess_kwargs = {
        "reflectivity_curve": exp_curve_interp,
        "q_values": q_model,
        "num_samples": nf_num_samples,
        "prior_bounds": _extend_prior_bounds_for_nf(prior_bounds),
        "q_resolution": q_resolution,
        "calc_sampled_curves": True,
        "calc_sampled_sld_profiles": True,
        "calc_log_likelihoods": True,
        "enable_importance_sampling": nf_enable_importance_sampling,
        "clip_prediction": clip_prediction,
    }
    if sigmas_interp is not None:
        preprocess_kwargs["sigmas"] = sigmas_interp

    nf_prediction_dict = inference_model.preprocess_and_sample(**preprocess_kwargs)

    # Compute NF sample statistics before selecting best sample
    print("Computing NF sample statistics...")
    nf_stats = compute_nf_sample_statistics(
        predicted_params_array=nf_prediction_dict["predicted_params_array"],
        log_prob=nf_prediction_dict["log_likelihoods"],
    )

    prediction_dict = _nf_prediction_to_single_prediction_dict(nf_prediction_dict)

    # Merge NF statistics into prediction_dict
    prediction_dict.update(nf_stats)

    if apply_constraints:
        print("Applying physical constraints...")
        prediction_dict = apply_physical_constraints(prediction_dict)
    else:
        print("Physical constraints disabled - skipping constraint application")

    q_pred = np.asarray(prediction_dict["q_plot_pred"])
    return q_pred, prediction_dict


def display_results(prediction_dict):
    """Display prediction results in a formatted way."""
    print("\nPrediction Results:")
    print("-" * 50)

    pred_params = prediction_dict["predicted_params_array"]
    polished_params = prediction_dict["polished_params_array"]
    param_names = prediction_dict["param_names"]

    for param_name, pred_val, polished_val in zip(
        param_names, pred_params, polished_params
    ):
        print(
            f"{param_name.ljust(18)} -> Predicted: {pred_val:.3f}    Polished: {polished_val:.3f}"
        )


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
        use_sigmas_input: Use sigmas as additional input channel to neural network (requires 2-channel model)

    Returns:
        Dictionary with results including parameters and metrics
    """
    data_directory = "data"

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

    # Get prior bounds
    prior_bounds = get_prior_bounds_for_experiment(
        experiment_id,
        true_params_dict,
        priors_type=priors_type,
        deviation=priors_deviation,
        layer_count=final_layer_count,
        fix_sld_mode=fix_sld_mode,
    )

    # Initialize inference model
    if config_name is None:
        if inference_backend == "nf":
            if final_layer_count != 1:
                raise ValueError(
                    "Default NF config 'example_nf_config_reflectorch.yaml' is a 1-layer model. "
                    "Provide --config-name for an NF model matching the requested layer_count."
                )
            config_name = "example_nf_config_reflectorch.yaml"
        else:
            config_name = "b_mc_point_neutron_conv_standard_L1_InputQDq"
    inference_model = EasyInferenceModel(config_name=config_name, device="cpu")

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
        )

    # Calculate metrics
    fit_metrics = calculate_fit_metrics(
        curve_exp, prediction_dict["polished_curve"], sigmas_exp, q_exp, q_model
    )

    param_metrics = None
    if true_params_dict and f"{final_layer_count}_layer" in true_params_dict:
        true_param_block = true_params_dict[f"{final_layer_count}_layer"]
        true_params = true_param_block["params"]
        true_param_names = true_param_block["param_names"]

        pred_params_for_metrics = prediction_dict["polished_params_array"]
        pred_param_names = prediction_dict.get("param_names", [])

        # NF models can include nuisance parameters (e.g., r_scale, log10_background) and
        # can label standard parameters differently (e.g., "Thickness L1").
        # Align predicted parameters to the true parameter set by name.
        if len(pred_params_for_metrics) != len(true_params) and pred_param_names:
            canonical_to_index: dict[str, int] = {}
            for i, pred_name in enumerate(pred_param_names):
                canonical = _map_pred_param_name_to_canonical_true_name(
                    pred_name, final_layer_count
                )
                if canonical and canonical not in canonical_to_index:
                    canonical_to_index[canonical] = i

            missing = [
                true_name
                for true_name in true_param_names
                if true_name not in canonical_to_index
            ]
            if missing:
                raise ValueError(
                    f"Predicted param_names missing required true params: {missing}. "
                    f"Predicted names: {pred_param_names}"
                )

            indices = [canonical_to_index[true_name] for true_name in true_param_names]
            pred_params_for_metrics = np.asarray(pred_params_for_metrics)[indices]

        param_metrics = calculate_parameter_metrics(
            pred_params_for_metrics,
            true_params,
            true_param_names,
            prior_bounds=prior_bounds,
            priors_type=priors_type,
        )

    # Prepare results dictionary
    results = {
        "experiment_id": experiment_id,
        "layer_count": final_layer_count,
        "prediction_dict": prediction_dict,
        "fit_metrics": fit_metrics,
        "param_metrics": param_metrics,
        "true_params_dict": true_params_dict,
        "q_exp": q_exp,
        "curve_exp": curve_exp,
        "sigmas_exp": sigmas_exp,
        "q_model": q_model,
        "prior_bounds": prior_bounds,
        "priors_config": {
            "priors_type": priors_type,
            "priors_deviation": priors_deviation,
            "fix_sld_mode": fix_sld_mode,
        },
    }

    return results


def main():
    """Main function to run a single experiment with specific settings."""
    # Experiment configuration
    experiment_name = "s007384"
    layer_count = 1

    print(f"Running inference for experiment: {experiment_name}")
    print(f"Layer count: {layer_count}")
    print("Running with 99% constraint-based priors and preprocessing OFF.")
    print("=" * 60)

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
    fit_metrics = results["fit_metrics"]
    param_metrics = results["param_metrics"]
    true_params_dict = results["true_params_dict"]
    q_exp = results["q_exp"]
    curve_exp = results["curve_exp"]
    sigmas_exp = results["sigmas_exp"]
    q_model = results["q_model"]

    # Display results
    display_results(prediction_dict)

    # Print metrics report
    print_metrics_report(
        fit_metrics, param_metrics, "b_mc_point_neutron_conv_standard_L1_InputQDq"
    )

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

    print(f"\nInference completed for experiment {experiment_name}")


if __name__ == "__main__":
    print("Simple pipeline module loaded successfully.")
    print("Available functions:")
    print("  - run_single_experiment() - Main function for processing one experiment")
    print("  - load_experimental_data() - Load and preprocess data from file")
    print("  - run_inference() - Run reflectorch inference")
    print("  - display_results() - Display prediction results")
    print("\nTo run the example:")
    print("  main()")

    # Optionally run the main function
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--run-example":
        main()
    else:
        print("\nUse --run-example to run the built-in example")
