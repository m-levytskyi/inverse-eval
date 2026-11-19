#!/usr/bin/env python3
"""
Parameter discovery and prior generation utilities.

This module provides functions for discovering experiment files, parsing true parameters,
and generating various types of prior bounds for reflectometry inference.
"""

import logging
from pathlib import Path

import numpy as np

from constraints_utils import get_constraint_ranges
from sld_profile_utils import sld_profile

logger = logging.getLogger(__name__)


def _find_experiment_pairs(
    data_directory, layer_count, use_theoretical, experiment_id=None
):
    """Find complete same-directory pairs, validating path and model layer counts."""
    if layer_count not in (None, 1, 2):
        raise ValueError(f"Unsupported layer count: {layer_count}")
    root = Path(data_directory)
    suffix = "_theoretical_curve.dat" if use_theoretical else "_experimental_curve.dat"
    pattern = f"{experiment_id}{suffix}" if experiment_id is not None else f"s*{suffix}"
    for curve in sorted(root.rglob(pattern)):
        path_layer = next(
            (
                int(directory.name)
                for directory in [curve.parent, *curve.parent.parents]
                if directory.name in {"1", "2"}
                and (directory == root or root in directory.parents)
            ),
            None,
        )
        if layer_count is not None and path_layer not in (None, layer_count):
            continue
        exp_id = curve.name.removesuffix(suffix)
        model = next(
            (
                candidate
                for extension in ("txt", "dat")
                if (
                    candidate := curve.with_name(f"{exp_id}_model.{extension}")
                ).is_file()
            ),
            None,
        )
        if model is None:
            continue
        parsed = parse_true_parameters_from_model_file(model)
        model_layer = next((n for n in (1, 2) if f"{n}_layer" in parsed), None)
        if model_layer is None:
            logger.warning("Skipping %s: no supported one- or two-layer model", model)
            continue
        if path_layer is not None and model_layer not in (None, path_layer):
            raise ValueError(f"Layer directory and model disagree for {model}")
        detected_layer = model_layer or path_layer
        if layer_count is not None and detected_layer not in (None, layer_count):
            continue
        yield exp_id, curve, model, detected_layer


def discover_experiment_files(
    experiment_id, data_directory, layer_count=None, use_theoretical=False
):
    """Return one same-directory curve/model pair; reject ambiguous experiment IDs."""
    pairs = list(
        _find_experiment_pairs(
            data_directory, layer_count, use_theoretical, experiment_id
        )
    )
    if len(pairs) > 1:
        raise ValueError(
            f"Ambiguous experiment {experiment_id}: {[str(p[1]) for p in pairs]}"
        )
    if not pairs:
        return None, None, None
    _, curve, model, detected_layer = pairs[0]
    return curve, model, detected_layer


def parse_true_parameters_from_model_file(model_file_path):
    """
    Parse true parameters from a Motofit-style model file.

    This function parses tabular model files where each row represents a layer:
    - fronting: ambient medium (usually air)
    - layer1, layer2, etc: material layers
    - backing: substrate

    Columns are: layer_name, sld(A^-2), thickness(A), roughness(A)

    Args:
        model_file_path: Path to model file

    Returns:
        Dictionary with parsed parameters for different layer interpretations
    """
    logger.info(f"Parsing true parameters from: {model_file_path}")

    if not Path(model_file_path).exists():
        logger.warning("Model file does not exist: %s", model_file_path)
        return {}

    with open(model_file_path, "r") as f:
        lines = f.readlines()

    # Parse the tabular format
    layers = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) >= 4:
            layer_name = parts[0]
            try:
                sld = float(parts[1])
                thickness_str = parts[2]
                roughness_str = parts[3]

                # Handle special values for thickness
                if thickness_str.lower() in ["inf", "none", "nan"]:
                    thickness = float("inf")
                else:
                    thickness = float(thickness_str)

                # Handle special values for roughness
                if roughness_str.lower() in ["none", "nan"]:
                    roughness = 0.0
                elif roughness_str.lower() == "inf":
                    roughness = float("inf")
                else:
                    roughness = float(roughness_str)

                layers[layer_name] = {
                    "sld": sld,
                    "thickness": thickness,
                    "roughness": roughness,
                }
                logger.info(
                    f"Parsed layer '{layer_name}': sld={sld}, thickness={thickness}, roughness={roughness}"
                )

            except (ValueError, IndexError) as e:
                logger.warning("Failed to parse line %r: %s", line, e)
                continue

    logger.info(f"Parsed layers: {list(layers.keys())}")

    true_params_dict = {}

    # Count actual material layers (exclude fronting and backing)
    material_layers = [name for name in layers.keys() if name.startswith("layer")]
    num_material_layers = len(material_layers)
    logger.info(f"Found {num_material_layers} material layers: {material_layers}")

    # Parse as 1-layer model (if we have 1 material layer)
    if num_material_layers == 1 and "layer1" in layers:
        logger.info("Parsing as 1-layer model")

        fronting = layers.get("fronting", {})
        layer1 = layers.get("layer1", {})
        backing = layers.get("backing", {})

        # 1-layer parameters: [thickness, amb_rough, sub_rough, layer_sld, sub_sld]
        thickness = layer1.get("thickness", 0.0)
        amb_rough = fronting.get("roughness", 0.0)  # fronting roughness
        sub_rough = layer1.get("roughness", 0.0)  # layer1 roughness
        layer_sld = layer1.get("sld", 0.0) * 1e6  # Convert to 10^-6 units
        sub_sld = backing.get("sld", 0.0) * 1e6  # Convert to 10^-6 units

        params_1_layer = [thickness, amb_rough, sub_rough, layer_sld, sub_sld]
        names_1_layer = get_parameter_names_for_layer_count(1)

        true_params_dict["1_layer"] = {
            "params": params_1_layer,
            "param_names": names_1_layer,
            "ambient_sld": fronting.get("sld", 0.0) * 1e6,
        }
        logger.info(f"1-layer parameters: {params_1_layer}")
        logger.info(
            f"SLD values converted to 10^-6 units - layer_sld: {layer_sld:.2f}, sub_sld: {sub_sld:.2f}"
        )

    # Parse as 2-layer model (if we have 2 material layers)
    if num_material_layers == 2 and "layer1" in layers and "layer2" in layers:
        logger.info("Parsing as 2-layer model")

        fronting = layers.get("fronting", {})
        layer1 = layers.get("layer1", {})
        layer2 = layers.get("layer2", {})
        backing = layers.get("backing", {})

        # 2-layer parameters: [thickness1, thickness2, amb_rough, int_rough, sub_rough,
        #                      layer1_sld, layer2_sld, sub_sld]
        thickness1 = layer1.get("thickness", 0.0)
        thickness2 = layer2.get("thickness", 0.0)
        amb_rough = fronting.get("roughness", 0.0)  # fronting roughness
        int_rough = layer1.get("roughness", 0.0)  # interface between layer1 and layer2
        sub_rough = layer2.get(
            "roughness", 0.0
        )  # interface between layer2 and substrate
        layer1_sld = layer1.get("sld", 0.0) * 1e6  # Convert to 10^-6 units
        layer2_sld = layer2.get("sld", 0.0) * 1e6  # Convert to 10^-6 units
        sub_sld = backing.get("sld", 0.0) * 1e6  # Convert to 10^-6 units

        params_2_layer = [
            thickness1,
            thickness2,
            amb_rough,
            int_rough,
            sub_rough,
            layer1_sld,
            layer2_sld,
            sub_sld,
        ]
        names_2_layer = get_parameter_names_for_layer_count(2)

        true_params_dict["2_layer"] = {
            "params": params_2_layer,
            "param_names": names_2_layer,
            "ambient_sld": fronting.get("sld", 0.0) * 1e6,
        }
        logger.info(f"2-layer parameters: {params_2_layer}")
        logger.info(
            f"SLD values converted to 10^-6 units - layer1_sld: {layer1_sld:.2f}, layer2_sld: {layer2_sld:.2f}, sub_sld: {sub_sld:.2f}"
        )

    return true_params_dict


def get_parameter_names_for_layer_count(layer_count):
    """
    Get parameter names for a given layer count.

    Args:
        layer_count: Number of layers (1 or 2)

    Returns:
        List of parameter names
    """
    if layer_count == 1:
        return ["thickness", "amb_rough", "sub_rough", "layer_sld", "sub_sld"]
    elif layer_count == 2:
        return [
            "thickness1",
            "thickness2",
            "amb_rough",
            "int_rough",
            "sub_rough",
            "layer1_sld",
            "layer2_sld",
            "sub_sld",
        ]
    else:
        raise ValueError(f"Unsupported layer count: {layer_count}")


def generate_true_sld_profile(
    true_params_dict, x_range=(0, 1000), n_points=1000, ambient_sld=None, x_axis=None
):
    """
    Generate a true SLD profile from parsed parameters.

    Args:
        true_params_dict: Dictionary with parsed true parameters
        x_range: Tuple of (min, max) depth values in Angstroms
        n_points: Number of points in the profile
        ambient_sld: Optional override of the parsed fronting SLD, in 10^-6 Angstrom^-2
        x_axis: Optional explicit depth axis in Angstroms

    Returns:
        Tuple of (x_axis, sld_profile) or (None, None) if no valid data
    """
    logger.info("Generating true SLD profile")

    if not true_params_dict:
        logger.info("No true parameters available")
        return None, None

    # Prefer 2-layer interpretation if available
    layer_key = "2_layer" if "2_layer" in true_params_dict else "1_layer"
    if layer_key not in true_params_dict:
        logger.info("No valid layer interpretation found")
        return None, None

    params = true_params_dict[layer_key]["params"]
    param_names = true_params_dict[layer_key]["param_names"]
    if ambient_sld is None:
        ambient_sld = true_params_dict[layer_key].get("ambient_sld", 0.0)

    logger.info(f"Using {layer_key} interpretation")
    logger.info(f"Parameters: {dict(zip(param_names, params))}")

    x_axis = (
        np.linspace(x_range[0], x_range[1], n_points)
        if x_axis is None
        else np.asarray(x_axis, dtype=float)
    )

    if layer_key == "1_layer":
        # 1-layer: [thickness, amb_rough, sub_rough, layer_sld, sub_sld]
        thickness, amb_rough, sub_rough, layer_sld, sub_sld = params

        sld_values = [ambient_sld, layer_sld, sub_sld]
        interfaces = [0, thickness]
        roughnesses = [amb_rough, sub_rough]

    elif layer_key == "2_layer":
        # 2-layer: [thickness1, thickness2, amb_rough, int_rough, sub_rough,
        #           layer1_sld, layer2_sld, sub_sld]
        (
            thickness1,
            thickness2,
            amb_rough,
            int_rough,
            sub_rough,
            layer1_sld,
            layer2_sld,
            sub_sld,
        ) = params

        sld_values = [ambient_sld, layer1_sld, layer2_sld, sub_sld]
        interfaces = [0, thickness1, thickness1 + thickness2]
        roughnesses = [amb_rough, int_rough, sub_rough]

    profile = sld_profile(x_axis, sld_values, interfaces, roughnesses)
    logger.info(f"Generated SLD profile with {n_points} points over range {x_range}")
    return x_axis, profile


def _bounded_prior(value, requested_width, value_range, width_range, name):
    """Keep the requested width inside model limits without moving truth outside it."""
    lower, upper = value_range
    minimum_width, maximum_width = width_range
    if not np.all(
        np.isfinite(
            [value, requested_width, lower, upper, minimum_width, maximum_width]
        )
    ):
        raise ValueError(f"Non-finite prior values for {name}")
    if (
        lower >= upper
        or minimum_width <= 0
        or maximum_width < minimum_width
        or requested_width < 0
    ):
        raise ValueError(f"Invalid prior limits for {name}")
    if not lower <= value <= upper:
        raise ValueError(
            f"True {name}={value} is outside supported inference range [{lower}, {upper}]"
        )
    maximum_width = min(maximum_width, upper - lower)
    if maximum_width < minimum_width:
        raise ValueError(
            f"Supported inference range for {name} is narrower than its minimum prior width"
        )
    width = min(max(requested_width, minimum_width), maximum_width)
    low = min(max(value - width / 2, lower), upper - width)
    return float(low), float(low + width)


def _prior_limits(model_ranges=None, width_limits=None):
    """Retain legacy point-model limits for callers without a selected model."""
    evaluation_ranges = get_constraint_ranges()
    ranges = dict(evaluation_ranges)
    if model_ranges is not None:
        for name, (lower, upper) in model_ranges.items():
            if name in ranges:
                ranges[name] = (
                    max(lower, ranges[name][0]),
                    min(upper, ranges[name][1]),
                )
    widths = {
        name: (0.01, 5.0 if "sld" in name else upper - lower)
        for name, (lower, upper) in evaluation_ranges.items()
    }
    if width_limits is not None:
        widths.update(width_limits)
    return ranges, widths


def apply_sld_fixing(
    bounds,
    true_params_dict,
    layer_count,
    fix_sld_mode,
    model_ranges=None,
    width_limits=None,
):
    """Use the narrowest supported SLD intervals for the requested fixing mode."""
    if fix_sld_mode not in {"none", "backing", "all"}:
        raise ValueError(f"Unknown fix_sld_mode: {fix_sld_mode}")
    if fix_sld_mode == "none":
        return bounds
    block = true_params_dict[f"{layer_count}_layer"]
    names = get_parameter_names_for_layer_count(layer_count)
    ranges, widths = _prior_limits(model_ranges, width_limits)
    fixed = list(bounds)
    for index, (name, value) in enumerate(zip(names, block["params"])):
        if name == "sub_sld" or (fix_sld_mode == "all" and "sld" in name):
            fixed[index] = _bounded_prior(
                value, 0.002, ranges[name], widths[name], name
            )
    return fixed


def get_constraint_based_prior_bounds(
    true_params_dict,
    layer_count,
    constraint_percentage,
    model_ranges=None,
    width_limits=None,
):
    """Generate priors from fixed evaluation spans, capped by the selected model."""
    return get_prior_bounds_for_experiment(
        "unknown",
        true_params_dict,
        priors_type="constraint_based",
        deviation=constraint_percentage,
        layer_count=layer_count,
        model_ranges=model_ranges,
        width_limits=width_limits,
    )


def get_prior_bounds_for_experiment(
    experiment_id,
    true_params_dict=None,
    priors_type="constraint_based",
    deviation=0.5,
    layer_count=1,
    fix_sld_mode="none",
    model_ranges=None,
    width_limits=None,
):
    """Build absolute-coordinate priors while retaining fixed evaluation spans."""
    if priors_type not in {"constraint_based", "narrow"}:
        raise ValueError(f"Unsupported priors_type='{priors_type}'")
    if not np.isfinite(deviation) or deviation < 0:
        raise ValueError("Prior deviation must be finite and nonnegative")
    block = (true_params_dict or {}).get(f"{layer_count}_layer")
    if block is None:
        raise ValueError(
            f"No true parameters found for {experiment_id} ({layer_count} layers)"
        )
    names = get_parameter_names_for_layer_count(layer_count)
    values = block["params"]
    if len(values) != len(names):
        raise ValueError("True parameter count does not match layer count")
    ranges, widths = _prior_limits(model_ranges, width_limits)
    evaluation_ranges = get_constraint_ranges()
    bounds = []
    for name, value in zip(names, values):
        span = evaluation_ranges[name][1] - evaluation_ranges[name][0]
        requested_width = deviation * (
            span if priors_type == "constraint_based" else 2 * abs(value)
        )
        bounds.append(
            _bounded_prior(value, requested_width, ranges[name], widths[name], name)
        )
    return apply_sld_fixing(
        bounds,
        true_params_dict,
        layer_count,
        fix_sld_mode,
        model_ranges=model_ranges,
        width_limits=width_limits,
    )


def discover_batch_experiments(
    data_directory,
    layer_count=None,
    num_experiments=None,
    experiment_ids=None,
    use_theoretical=False,
):
    """Discover complete pairs in stable order; keep explicit IDs for failure reporting."""
    if experiment_ids:
        return list(dict.fromkeys(experiment_ids))
    if num_experiments is not None and num_experiments < 0:
        raise ValueError("num_experiments must be nonnegative")
    pairs = {}
    for exp_id, curve, model, detected_layer in _find_experiment_pairs(
        data_directory, layer_count, use_theoretical
    ):
        if exp_id in pairs:
            raise ValueError(
                f"Ambiguous experiment {exp_id}: {pairs[exp_id]} and {curve}"
            )
        pairs[exp_id] = curve
    ids = sorted(pairs)
    return ids if num_experiments is None else ids[:num_experiments]


def check_experiment_within_constraints(experiment_id, true_params_dict, layer_count):
    """
    Check if an experiment's true parameters fall within the model constraint ranges.

    An experiment is considered an outlier if any of its true parameter values fall
    outside the MODEL CONSTRAINTS (e.g., thickness: 1-1000, roughness: 0-60, SLD: -8 to 16).

    Args:
        experiment_id: Experiment identifier
        true_params_dict: Dictionary with true parameters
        layer_count: Number of layers (1 or 2)

    Returns:
        Tuple of (is_within_constraints: bool, outlier_parameters: list)
        - is_within_constraints: True if all parameters are within model constraints
        - outlier_parameters: List of (param_name, value, constraint_min, constraint_max) for outliers
    """
    layer_key = f"{layer_count}_layer"
    if layer_key not in true_params_dict:
        return True, []

    true_params = true_params_dict[layer_key]["params"]
    param_names = get_parameter_names_for_layer_count(layer_count)
    constraint_ranges = get_constraint_ranges()

    outlier_parameters = []
    for param_name, param_value in zip(param_names, true_params):
        if param_name not in constraint_ranges:
            logger.debug("No constraint range configured for %s", param_name)
            continue
        constraint_min, constraint_max = constraint_ranges[param_name]
        if (
            not np.isfinite(param_value)
            or param_value < constraint_min
            or param_value > constraint_max
        ):
            outlier_parameters.append(
                (param_name, param_value, constraint_min, constraint_max)
            )

    return len(outlier_parameters) == 0, outlier_parameters


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("Parameter discovery module")
    logger.info(
        "Functions: discover_experiment_files, parse_true_parameters_from_model_file, "
    )
    logger.info(
        "           generate_true_sld_profile, get_prior_bounds_for_experiment, "
    )
    logger.info(
        "           get_parameter_names_for_layer_count, discover_batch_experiments, "
    )
    logger.info("           check_experiment_within_constraints")
