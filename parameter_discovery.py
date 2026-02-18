#!/usr/bin/env python3
"""
Parameter discovery and prior generation utilities.

This module provides functions for discovering experiment files, parsing true parameters,
and generating various types of prior bounds for reflectometry inference.
"""

import numpy as np
from pathlib import Path
from constraints_utils import get_constraint_ranges, get_constraint_range


def discover_experiment_files(
    experiment_id, data_directory, layer_count=None, use_theoretical=False
):
    """
    Discover experimental data and model files for a given experiment ID.

    Args:
        experiment_id: Experiment identifier (e.g., 's000000')
        data_directory: Base data directory to search
        layer_count: Number of layers (1 or 2). If None, searches both
        use_theoretical: If True, use theoretical curves; if False (default), use experimental curves

    Returns:
        Tuple of (data_file_path, model_file_path, detected_layer_count) or (None, None, None) if not found
    """
    data_dir = Path(data_directory)
    exp_data_file = None
    exp_model_file = None
    detected_layer_count = None

    print(f"Searching for experiment files for {experiment_id}")
    print(f"Base directory: {data_dir}")
    if layer_count:
        print(f"Looking specifically for {layer_count}-layer data")

    # Define search patterns - choose based on use_theoretical flag
    if use_theoretical:
        data_patterns = [f"{experiment_id}_theoretical_curve.dat"]
        print("Using THEORETICAL curves")
    else:
        data_patterns = [f"{experiment_id}_experimental_curve.dat"]
        print("Using EXPERIMENTAL curves")

    model_patterns = [f"{experiment_id}_model.txt", f"{experiment_id}_model.dat"]

    # Define layer directories to search
    layer_dirs_to_search = []
    if layer_count is not None:
        layer_dirs_to_search = [str(layer_count)]
    else:
        layer_dirs_to_search = ["1", "2"]  # Search both if not specified

    # Search in subdirectories recursively
    def search_directory(directory, depth=0):
        nonlocal exp_data_file, exp_model_file, detected_layer_count

        if depth > 3:  # Prevent infinite recursion
            return

        if not directory.is_dir():
            return

        print(f"Searching in: {directory}")

        # Check if this is a layer directory (contains '1' or '2')
        dir_name = directory.name
        current_layer_count = None

        # Check if directory name indicates layer count
        if dir_name in layer_dirs_to_search:
            current_layer_count = int(dir_name)
            print(
                f"Found layer directory: {directory} (layer count: {current_layer_count})"
            )

        # Look for data files in current directory
        for pattern in data_patterns:
            data_file = directory / pattern
            if data_file.exists():
                print(f"Found experimental data: {data_file}")
                exp_data_file = data_file
                if current_layer_count:
                    detected_layer_count = current_layer_count
                break

        # Look for model files in current directory
        for pattern in model_patterns:
            model_file = directory / pattern
            if model_file.exists():
                print(f"Found model file: {model_file}")
                exp_model_file = model_file
                if current_layer_count and detected_layer_count is None:
                    detected_layer_count = current_layer_count
                break

        # If both files found, stop searching
        if exp_data_file and exp_model_file:
            return

        # Recursively search subdirectories
        try:
            for subdir in directory.iterdir():
                if subdir.is_dir():
                    search_directory(subdir, depth + 1)
                    if exp_data_file and exp_model_file:
                        return
        except PermissionError:
            print(f"Permission denied accessing: {directory}")

    # Start search from data directory
    search_directory(data_dir)

    if not exp_data_file:
        print(f"ERROR: Experimental data file for {experiment_id} not found")
        print(f"Searched patterns: {data_patterns}")
        print(f"In layer directories: {layer_dirs_to_search}")
    if not exp_model_file:
        print(f"ERROR: Model file for {experiment_id} not found")
        print(f"Searched patterns: {model_patterns}")
        print(f"In layer directories: {layer_dirs_to_search}")

    if detected_layer_count:
        print(f"Detected layer count: {detected_layer_count}")

    return exp_data_file, exp_model_file, detected_layer_count


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
    print(f"Parsing true parameters from: {model_file_path}")

    if not Path(model_file_path).exists():
        print(f"ERROR: Model file does not exist: {model_file_path}")
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
                print(
                    f"Parsed layer '{layer_name}': sld={sld}, thickness={thickness}, roughness={roughness}"
                )

            except (ValueError, IndexError) as e:
                print(f"Failed to parse line '{line}': {e}")
                continue

    print(f"Parsed layers: {list(layers.keys())}")

    true_params_dict = {}

    # Count actual material layers (exclude fronting and backing)
    material_layers = [name for name in layers.keys() if name.startswith("layer")]
    num_material_layers = len(material_layers)
    print(f"Found {num_material_layers} material layers: {material_layers}")

    # Parse as 1-layer model (if we have 1 material layer)
    if num_material_layers == 1 and "layer1" in layers:
        try:
            print("Parsing as 1-layer model")

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
            }
            print(f"1-layer parameters: {params_1_layer}")
            print(
                f"SLD values converted to 10^-6 units - layer_sld: {layer_sld:.2f}, sub_sld: {sub_sld:.2f}"
            )

        except Exception as e:
            print(f"Failed to parse as 1-layer model: {e}")

    # Parse as 2-layer model (if we have 2 material layers)
    if num_material_layers == 2 and "layer1" in layers and "layer2" in layers:
        try:
            print("Parsing as 2-layer model")

            fronting = layers.get("fronting", {})
            layer1 = layers.get("layer1", {})
            layer2 = layers.get("layer2", {})
            backing = layers.get("backing", {})

            # 2-layer parameters: [thickness1, thickness2, amb_rough, int_rough, sub_rough,
            #                      layer1_sld, layer2_sld, sub_sld]
            thickness1 = layer1.get("thickness", 0.0)
            thickness2 = layer2.get("thickness", 0.0)
            amb_rough = fronting.get("roughness", 0.0)  # fronting roughness
            int_rough = layer1.get(
                "roughness", 0.0
            )  # interface between layer1 and layer2
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
            }
            print(f"2-layer parameters: {params_2_layer}")
            print(
                f"SLD values converted to 10^-6 units - layer1_sld: {layer1_sld:.2f}, layer2_sld: {layer2_sld:.2f}, sub_sld: {sub_sld:.2f}"
            )

        except Exception as e:
            print(f"Failed to parse as 2-layer model: {e}")

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


def generate_true_sld_profile(true_params_dict, x_range=(0, 1000), n_points=1000):
    """
    Generate a true SLD profile from parsed parameters.

    Args:
        true_params_dict: Dictionary with parsed true parameters
        x_range: Tuple of (min, max) depth values in Angstroms
        n_points: Number of points in the profile

    Returns:
        Tuple of (x_axis, sld_profile) or (None, None) if no valid data
    """
    print("Generating true SLD profile")

    if not true_params_dict:
        print("No true parameters available")
        return None, None

    # Prefer 2-layer interpretation if available
    layer_key = "2_layer" if "2_layer" in true_params_dict else "1_layer"
    if layer_key not in true_params_dict:
        print("No valid layer interpretation found")
        return None, None

    params = true_params_dict[layer_key]["params"]
    param_names = true_params_dict[layer_key]["param_names"]

    print(f"Using {layer_key} interpretation")
    print(f"Parameters: {dict(zip(param_names, params))}")

    x_axis = np.linspace(x_range[0], x_range[1], n_points)
    sld_profile = np.zeros_like(x_axis)

    if layer_key == "1_layer":
        # 1-layer: [thickness, amb_rough, sub_rough, layer_sld, sub_sld]
        thickness, amb_rough, sub_rough, layer_sld, sub_sld = params

        # Simple step function model
        layer_mask = (x_axis >= 0) & (x_axis <= thickness)
        substrate_mask = x_axis > thickness

        sld_profile[layer_mask] = layer_sld
        sld_profile[substrate_mask] = sub_sld

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

        # Simple step function model
        layer1_mask = (x_axis >= 0) & (x_axis <= thickness1)
        layer2_mask = (x_axis > thickness1) & (x_axis <= thickness1 + thickness2)
        substrate_mask = x_axis > thickness1 + thickness2

        sld_profile[layer1_mask] = layer1_sld
        sld_profile[layer2_mask] = layer2_sld
        sld_profile[substrate_mask] = sub_sld

    print(f"Generated SLD profile with {n_points} points over range {x_range}")
    return x_axis, sld_profile


def apply_sld_fixing(bounds, true_params_dict, layer_count, fix_sld_mode):
    """
    Apply SLD fixing to prior bounds based on the specified mode.

    Args:
        bounds: List of (min, max) tuples for prior bounds
        true_params_dict: Dictionary with true parameters
        layer_count: Number of layers (1 or 2)
        fix_sld_mode: SLD fixing mode - "backing", "all", or "none"

    Returns:
        Updated list of (min, max) tuples with fixed SLD bounds
    """
    if fix_sld_mode == "none" or not true_params_dict:
        return bounds

    layer_key = f"{layer_count}_layer"
    if layer_key not in true_params_dict:
        print(
            f"Warning: No true parameters found for {layer_count} layer - cannot apply SLD fixing"
        )
        return bounds

    true_params = true_params_dict[layer_key]["params"]
    param_names = get_parameter_names_for_layer_count(layer_count)

    # Create a copy of bounds to modify
    fixed_bounds = list(bounds)

    print(f"Applying SLD fixing mode: {fix_sld_mode}")

    # Define which SLDs to fix based on mode
    if fix_sld_mode == "backing":
        if layer_count == 1:
            # Can only fix backing (sub_sld) - fronting SLD is not a parameter
            sld_params_to_fix = ["sub_sld"]  # backing only
        elif layer_count == 2:
            # Can only fix backing (sub_sld) - fronting SLD is not a parameter
            sld_params_to_fix = ["sub_sld"]  # backing only
    elif fix_sld_mode == "all":
        if layer_count == 1:
            sld_params_to_fix = ["layer_sld", "sub_sld"]  # all SLDs
        elif layer_count == 2:
            sld_params_to_fix = ["layer1_sld", "layer2_sld", "sub_sld"]  # all SLDs
    else:
        raise ValueError(f"Unknown fix_sld_mode: {fix_sld_mode}")

    # Apply fixing
    for i, param_name in enumerate(param_names):
        if param_name in sld_params_to_fix:
            true_value = true_params[i]
            # Set tight bounds around the true value (±0.001 to essentially fix it)
            epsilon = 0.001
            fixed_bounds[i] = (true_value - epsilon, true_value + epsilon)
            print(
                f"  Fixed {param_name}: {true_value:.3f} (bounds: [{true_value - epsilon:.3f}, {true_value + epsilon:.3f}])"
            )

    return fixed_bounds


def get_constraint_based_prior_bounds(
    true_params_dict, layer_count, constraint_percentage
):
    """
    Generate constraint-based prior bounds using a percentage of the model constraint span.

    Args:
        true_params_dict: Dictionary with true parameters
        layer_count: Number of layers (1 or 2)
        constraint_percentage: Percentage of constraint span to use (e.g., 0.3 for 30%)

    Returns:
        List of (min, max) tuples for prior bounds
    """
    layer_key = f"{layer_count}_layer"
    if layer_key not in true_params_dict:
        raise ValueError(f"No true parameters found for {layer_count} layer")

    true_params = true_params_dict[layer_key]["params"]
    param_names = get_parameter_names_for_layer_count(layer_count)

    # Load model constraints from centralized definition
    model_constraints = get_constraint_ranges()

    # Allowed widths (same as in narrow priors)
    allowed_widths = {
        "thickness": (0.01, 1000.0),
        "amb_rough": (0.01, 60.0),
        "sub_rough": (0.01, 60.0),
        "int_rough": (0.01, 60.0),  # for 2-layer
        "layer_sld": (0.01, 5.0),
        "layer1_sld": (0.01, 5.0),  # for 2-layer
        "layer2_sld": (0.01, 5.0),  # for 2-layer
        "sub_sld": (0.01, 5.0),
        "thickness1": (0.01, 1000.0),  # for 2-layer
        "thickness2": (0.01, 1000.0),  # for 2-layer
    }

    bounds = []

    print(
        f"Generating constraint-based prior bounds ({constraint_percentage * 100:.0f}% of constraint span)"
    )

    for i, (param_name, param_value) in enumerate(zip(param_names, true_params)):
        # Get constraints for this parameter
        model_min, model_max = model_constraints.get(param_name, (-1e6, 1e6))
        width_min, width_max = allowed_widths.get(param_name, (0.01, 1e6))

        # Calculate span of model constraints
        constraint_span = model_max - model_min

        # Calculate width as percentage of constraint span
        target_width = constraint_percentage * constraint_span

        # Clip to allowed widths
        target_width = max(width_min, min(target_width, width_max))

        # Center around true value
        half_width = target_width / 2
        min_val = param_value - half_width
        max_val = param_value + half_width

        # Adjust if bounds exceed model constraints
        if max_val > model_max:
            # Shift left
            shift = max_val - model_max
            max_val = model_max
            min_val = max(min_val - shift, model_min)

        if min_val < model_min:
            # Shift right
            shift = model_min - min_val
            min_val = model_min
            max_val = min(max_val + shift, model_max)

        # Final safety check - ensure bounds are within model constraints
        min_val = max(min_val, model_min)
        max_val = min(max_val, model_max)

        bounds.append((min_val, max_val))

        print(
            f"  {param_name}: true={param_value:.3f}, constraint_span={constraint_span:.1f}, "
            f"target_width={target_width:.3f} -> [{min_val:.3f}, {max_val:.3f}]"
        )

    return bounds


def get_prior_bounds_for_experiment(
    experiment_id,
    true_params_dict=None,
    priors_type="constraint_based",
    deviation=0.5,
    layer_count=1,
    fix_sld_mode="none",
):
    """
    Generate appropriate prior bounds for an experiment.

    Args:
        experiment_id: Experiment identifier
        true_params_dict: Dictionary with true parameters (for narrow/constraint_based priors)
        priors_type: "narrow" or "constraint_based"
        deviation: Relative deviation for narrow priors (e.g., 0.5 for 50%) or
                  constraint percentage for constraint_based priors (e.g., 0.3 for 30%)
        layer_count: Number of layers (1 or 2)
        fix_sld_mode: SLD fixing mode - "none", "backing", or "all"

    Returns:
        List of (min, max) tuples for prior bounds
    """
    print(
        f"Generating {priors_type} prior bounds for {experiment_id} ({layer_count} layer{'s' if layer_count > 1 else ''})"
    )
    if fix_sld_mode != "none":
        print(f"SLD fixing mode: {fix_sld_mode}")

    if priors_type == "constraint_based" and true_params_dict:
        # Use constraint-based priors
        bounds = get_constraint_based_prior_bounds(
            true_params_dict, layer_count, deviation
        )

        # Apply SLD fixing if requested
        if fix_sld_mode != "none":
            bounds = apply_sld_fixing(
                bounds, true_params_dict, layer_count, fix_sld_mode
            )

        return bounds

    elif priors_type == "narrow" and true_params_dict:
        # Use true parameters to generate narrow priors
        layer_key = f"{layer_count}_layer"
        if layer_key in true_params_dict:
            true_params = true_params_dict[layer_key]["params"]
            param_names = get_parameter_names_for_layer_count(layer_count)
            bounds = []

            # Load model constraints from centralized definition
            model_constraints = get_constraint_ranges()

            allowed_widths = {
                "thickness": (0.01, 1000.0),
                "amb_rough": (0.01, 60.0),
                "sub_rough": (0.01, 60.0),
                "int_rough": (0.01, 60.0),  # for 2-layer
                "layer_sld": (0.01, 5.0),
                "layer1_sld": (0.01, 5.0),  # for 2-layer
                "layer2_sld": (0.01, 5.0),  # for 2-layer
                "sub_sld": (0.01, 5.0),
                "thickness1": (0.01, 1000.0),  # for 2-layer
                "thickness2": (0.01, 1000.0),  # for 2-layer
            }

            for i, (param_name, param_value) in enumerate(
                zip(param_names, true_params)
            ):
                # Get constraints for this parameter
                model_min, model_max = model_constraints.get(param_name, (-1e6, 1e6))
                width_min, width_max = allowed_widths.get(param_name, (0.01, 1e6))

                # Calculate initial bounds based on deviation
                if param_value > 0:
                    min_val = param_value * (1 - deviation)
                    max_val = param_value * (1 + deviation)
                else:
                    # For negative parameters, use relative deviation around the true value
                    min_val = param_value * (1 + deviation)  # More negative
                    max_val = param_value * (1 - deviation)  # Less negative

                # Ensure proper ordering (min < max)
                if min_val > max_val:
                    min_val, max_val = max_val, min_val

                # Calculate current width
                current_width = max_val - min_val

                # Constrain width to allowed limits
                if current_width < width_min:
                    # Expand bounds symmetrically around true value
                    center = param_value
                    half_width = width_min / 2
                    min_val = center - half_width
                    max_val = center + half_width
                elif current_width > width_max:
                    # Shrink bounds symmetrically around true value
                    center = param_value
                    half_width = width_max / 2
                    min_val = center - half_width
                    max_val = center + half_width

                # Constrain to model limits
                min_val = max(min_val, model_min)
                max_val = min(max_val, model_max)

                # Ensure min <= max after constraining to model limits
                if min_val > max_val:
                    # If true parameter is outside model constraints, use full model range
                    if param_value > model_max:
                        # True value too high, use full model range or centered around model_max
                        max_val = model_max
                        min_val = max(model_max - width_max, model_min)
                    elif param_value < model_min:
                        # True value too low, use full model range or centered around model_min
                        min_val = model_min
                        max_val = min(model_min + width_max, model_max)
                    else:
                        # Should not happen, but safety fallback
                        min_val = model_min
                        max_val = model_max

                # Final check: ensure minimum width is still respected after model constraints
                final_width = max_val - min_val
                if final_width < width_min:
                    # If constraining to model limits made width too small,
                    # expand within model limits as much as possible
                    available_width = model_max - model_min
                    if available_width >= width_min:
                        center = (min_val + max_val) / 2
                        half_width = width_min / 2
                        min_val = max(center - half_width, model_min)
                        max_val = min(center + half_width, model_max)
                        # Adjust if still outside bounds
                        if max_val > model_max:
                            max_val = model_max
                            min_val = max_val - width_min
                        if min_val < model_min:
                            min_val = model_min
                            max_val = min_val + width_min

                bounds.append((min_val, max_val))

            print(
                f"Generated narrow priors with {deviation * 100}% deviation (constrained to model limits)"
            )

            # Apply SLD fixing if requested
            if fix_sld_mode != "none":
                bounds = apply_sld_fixing(
                    bounds, true_params_dict, layer_count, fix_sld_mode
                )

            # Log the detailed bounds for debugging
            print("Final narrow prior bounds details:")
            for i, (name, (min_val, max_val)) in enumerate(zip(param_names, bounds)):
                width = max_val - min_val
                print(f"  {name}: [{min_val:.3f}, {max_val:.3f}] (width: {width:.3f})")

            return bounds

    raise ValueError(
        f"Unsupported priors_type='{priors_type}' for layer_count={layer_count}"
    )


def discover_batch_experiments(
    data_directory, layer_count=None, num_experiments=None, experiment_ids=None
):
    """
    Discover available experiments in the data directory for batch processing.

    Args:
        data_directory: Base data directory to search
        layer_count: Number of layers to search for (1 or 2). If None, searches all
        num_experiments: Maximum number of experiments to return (None for all)
        experiment_ids: List of specific experiment IDs to validate (None for discovery)

    Returns:
        List of experiment IDs
    """
    data_dir = Path(data_directory)

    if experiment_ids:
        print(f"\nValidating {len(experiment_ids)} provided experiment IDs")
        return experiment_ids

    # Original discovery logic
    print(f"\nDiscovering experiments in {data_dir}")
    if layer_count is not None:
        print(f"  Looking for {layer_count}-layer experiments")
    if num_experiments is not None:
        print(f"  Limiting to first {num_experiments} experiments")

    experiments = []

    # Try MARIA dataset structure first
    maria_dataset_path = data_dir / "MARIA_VIPR_dataset"
    if maria_dataset_path.exists():
        print(f"  Found MARIA dataset: {maria_dataset_path}")

        # Search in subdirectories (e.g., 0/, 1/, 2/)
        layer_dirs_to_search = []
        if layer_count is not None:
            layer_dirs_to_search = [str(layer_count)]
        else:
            layer_dirs_to_search = ["0", "1", "2"]

        for layer_dir_name in layer_dirs_to_search:
            layer_dir = maria_dataset_path / layer_dir_name
            if layer_dir.is_dir():
                print(f"  Checking layer directory: {layer_dir.name}")

                # Find all experimental data files
                exp_files = list(layer_dir.glob("s*_experimental_curve.dat"))

                for exp_file in exp_files:
                    exp_id = exp_file.name.replace("_experimental_curve.dat", "")

                    # Check if model file exists
                    model_file = layer_dir / f"{exp_id}_model.txt"
                    if not model_file.exists():
                        model_file = layer_dir / f"{exp_id}_model.dat"

                    if model_file.exists():
                        experiments.append(exp_id)
                    else:
                        print(f"  Skipping {exp_id}: no model file found")

    else:
        print(f"  Directory not found: {maria_dataset_path}")

        # Try test data structure
        test_data_path = data_dir / "test_data"
        if test_data_path.exists():
            print(f"  Found test data: {test_data_path}")

            if layer_count is not None:
                layer_test_dir = test_data_path / str(layer_count)
                if layer_test_dir.exists():
                    exp_files = list(layer_test_dir.glob("s*_experimental_curve.dat"))
                    for exp_file in exp_files:
                        exp_id = exp_file.name.replace("_experimental_curve.dat", "")
                        experiments.append(exp_id)
                else:
                    print(f"  Layer directory not found: {layer_test_dir}")
            else:
                # Search all layer directories
                for layer_dir in test_data_path.iterdir():
                    if layer_dir.is_dir() and layer_dir.name.isdigit():
                        exp_files = list(layer_dir.glob("s*_experimental_curve.dat"))
                        for exp_file in exp_files:
                            exp_id = exp_file.name.replace(
                                "_experimental_curve.dat", ""
                            )
                            experiments.append(exp_id)
        else:
            print(f"  Directory not found: {test_data_path}")

            # Fallback: Search directly in data_directory
            print(f"  Searching directly in: {data_dir}")
            exp_files = list(data_dir.glob("s*_experimental_curve.dat"))

            if exp_files:
                print(f"  Found {len(exp_files)} experiment files in root directory")
                for exp_file in exp_files:
                    exp_id = exp_file.name.replace("_experimental_curve.dat", "")
                    experiments.append(exp_id)
            else:
                print("  No experiment files found in root directory")

    print(f"Found {len(experiments)} experiments")

    # Limit to requested number
    if num_experiments is not None and len(experiments) > num_experiments:
        print(f"Limiting to first {num_experiments} experiments")
        experiments = experiments[:num_experiments]

    return experiments


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

    outlier_parameters = []
    for param_name, param_value in zip(param_names, true_params):
        try:
            constraint_min, constraint_max = get_constraint_range(param_name)
            if param_value < constraint_min or param_value > constraint_max:
                outlier_parameters.append(
                    (param_name, param_value, constraint_min, constraint_max)
                )
        except KeyError:
            pass

    return len(outlier_parameters) == 0, outlier_parameters


if __name__ == "__main__":
    print("Parameter discovery module")
    print(
        "Functions: discover_experiment_files, parse_true_parameters_from_model_file, "
    )
    print("           generate_true_sld_profile, get_prior_bounds_for_experiment, ")
    print(
        "           get_parameter_names_for_layer_count, discover_batch_experiments, "
    )
    print("           check_experiment_within_constraints")
