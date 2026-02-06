#!/usr/bin/env python3
"""
Extract prior bounds from MARIA dataset analysis for inference
Provides min/max/mean values for each parameter by layer count
"""

import json
import pandas as pd
from pathlib import Path


def extract_prior_bounds():
    """Extract prior bounds for each layer count from analysis results"""

    # Load the layer-specific statistics
    layer_stats_file = Path("maria_dataset_layer_statistics.json")
    summary_file = Path("maria_dataset_summary.json")

    if not layer_stats_file.exists():
        print("Error: maria_dataset_layer_statistics.json not found")
        return None

    if not summary_file.exists():
        print("Error: maria_dataset_summary.json not found")
        return None

    with open(layer_stats_file, "r") as f:
        layer_stats = json.load(f)

    with open(summary_file, "r") as f:
        summary_data = json.load(f)

    prior_bounds = {}

    print("=" * 80)
    print("MARIA DATASET PRIOR BOUNDS FOR INFERENCE")
    print("=" * 80)

    # Process each layer count
    for layer_count in [0, 1, 2]:
        layer_key = f"{layer_count}_layers"

        if layer_key not in layer_stats:
            continue

        data = layer_stats[layer_key]
        bounds_info = {
            "layer_count": layer_count,
            "total_experiments": data["total_experiments"],
            "parameters": {},
        }

        print(
            f"\n{layer_count}-LAYER EXPERIMENTS ({data['total_experiments']:,} experiments)"
        )
        print(f"{'=' * 60}")

        if layer_count == 0:
            # Interface-only experiments
            print("Interface parameters:")

            # Fronting material
            front_sld = data["fronting_sld"]
            front_rough = data["fronting_roughness"]
            back_sld = data["backing_sld"]
            back_rough = data["backing_roughness"]

            bounds_info["parameters"] = {
                "fronting_sld": {
                    "min": front_sld["min"],
                    "max": front_sld["max"],
                    "mean": front_sld["mean"],
                    "std": front_sld["std"],
                },
                "fronting_roughness": {
                    "min": front_rough["min"],
                    "max": front_rough["max"],
                    "mean": front_rough["mean"],
                    "std": front_rough["std"],
                },
                "backing_sld": {
                    "min": back_sld["min"],
                    "max": back_sld["max"],
                    "mean": back_sld["mean"],
                    "std": back_sld["std"],
                },
                "backing_roughness": {
                    "min": back_rough["min"],
                    "max": back_rough["max"],
                    "mean": back_rough["mean"],
                    "std": back_rough["std"],
                },
            }

            print(
                f"  Fronting SLD:       [{front_sld['min']:.2e}, {front_sld['max']:.2e}] (mean: {front_sld['mean']:.2e})"
            )
            print(
                f"  Fronting Roughness: [{front_rough['min']:.1f}, {front_rough['max']:.1f}] (mean: {front_rough['mean']:.1f})"
            )
            print(
                f"  Backing SLD:        [{back_sld['min']:.2e}, {back_sld['max']:.2e}] (mean: {back_sld['mean']:.2e})"
            )
            print(
                f"  Backing Roughness:  [{back_rough['min']:.1f}, {back_rough['max']:.1f}] (mean: {back_rough['mean']:.1f})"
            )

        else:
            # Experiments with deposited layers
            print("Deposited layer parameters:")

            # Overall layer statistics
            if "sld" in data:
                sld = data["sld"]
                thickness = data["thickness"]
                roughness = data["roughness"]

                bounds_info["parameters"]["overall"] = {
                    "sld": {
                        "min": sld["min"],
                        "max": sld["max"],
                        "mean": sld["mean"],
                        "std": sld["std"],
                    },
                    "thickness": {
                        "min": thickness["min"],
                        "max": thickness["max"],
                        "mean": thickness["mean"],
                        "std": thickness["std"],
                    },
                    "roughness": {
                        "min": roughness["min"],
                        "max": roughness["max"],
                        "mean": roughness["mean"],
                        "std": roughness["std"],
                    },
                }

                print(
                    f"  Overall SLD:        [{sld['min']:.2e}, {sld['max']:.2e}] (mean: {sld['mean']:.2e})"
                )
                print(
                    f"  Overall Thickness:  [{thickness['min']:.1f}, {thickness['max']:.1f}] (mean: {thickness['mean']:.1f})"
                )
                print(
                    f"  Overall Roughness:  [{roughness['min']:.1f}, {roughness['max']:.1f}] (mean: {roughness['mean']:.1f})"
                )

            # Layer position specific bounds
            if "layer_positions" in data:
                print(f"\nLayer-specific parameters:")
                bounds_info["parameters"]["by_position"] = {}

                for pos_key, pos_data in sorted(data["layer_positions"].items()):
                    layer_num = pos_key.split("_")[1]

                    sld = pos_data["sld"]
                    thickness = pos_data["thickness"]
                    roughness = pos_data["roughness"]

                    bounds_info["parameters"]["by_position"][pos_key] = {
                        "sld": {
                            "min": sld["min"],
                            "max": sld["max"],
                            "mean": sld["mean"],
                            "std": sld["std"],
                        },
                        "thickness": {
                            "min": thickness["min"],
                            "max": thickness["max"],
                            "mean": thickness["mean"],
                            "std": thickness["std"],
                        },
                        "roughness": {
                            "min": roughness["min"],
                            "max": roughness["max"],
                            "mean": roughness["mean"],
                            "std": roughness["std"],
                        },
                    }

                    print(
                        f"  Layer {layer_num} SLD:        [{sld['min']:.2e}, {sld['max']:.2e}] (mean: {sld['mean']:.2e})"
                    )
                    print(
                        f"  Layer {layer_num} Thickness:  [{thickness['min']:.1f}, {thickness['max']:.1f}] (mean: {thickness['mean']:.1f})"
                    )
                    print(
                        f"  Layer {layer_num} Roughness:  [{roughness['min']:.1f}, {roughness['max']:.1f}] (mean: {roughness['mean']:.1f})"
                    )

            # Interface materials for experiments with layers
            if "fronting_sld" in data:
                print(f"\nInterface parameters:")
                front_sld = data["fronting_sld"]
                front_rough = data["fronting_roughness"]
                back_sld = data["backing_sld"]
                back_rough = data["backing_roughness"]

                bounds_info["parameters"]["interface"] = {
                    "fronting_sld": {
                        "min": front_sld["min"],
                        "max": front_sld["max"],
                        "mean": front_sld["mean"],
                        "std": front_sld["std"],
                    },
                    "fronting_roughness": {
                        "min": front_rough["min"],
                        "max": front_rough["max"],
                        "mean": front_rough["mean"],
                        "std": front_rough["std"],
                    },
                    "backing_sld": {
                        "min": back_sld["min"],
                        "max": back_sld["max"],
                        "mean": back_sld["mean"],
                        "std": back_sld["std"],
                    },
                    "backing_roughness": {
                        "min": back_rough["min"],
                        "max": back_rough["max"],
                        "mean": back_rough["mean"],
                        "std": back_rough["std"],
                    },
                }

                print(
                    f"  Fronting SLD:       [{front_sld['min']:.2e}, {front_sld['max']:.2e}] (mean: {front_sld['mean']:.2e})"
                )
                print(
                    f"  Fronting Roughness: [{front_rough['min']:.1f}, {front_rough['max']:.1f}] (mean: {front_rough['mean']:.1f})"
                )
                print(
                    f"  Backing SLD:        [{back_sld['min']:.2e}, {back_sld['max']:.2e}] (mean: {back_sld['mean']:.2e})"
                )
                print(
                    f"  Backing Roughness:  [{back_rough['min']:.1f}, {back_rough['max']:.1f}] (mean: {back_rough['mean']:.1f})"
                )

        prior_bounds[layer_key] = bounds_info

    # Also extract combined bounds from summary for comparison
    print(f"\n" + "=" * 80)
    print("COMBINED BOUNDS ACROSS ALL EXPERIMENTS")
    print("=" * 80)

    param_stats = summary_data["parameter_statistics"]
    combined_bounds = {}

    for param_type in ["sld", "thickness", "roughness"]:
        if param_type in param_stats:
            print(f"\n{param_type.upper()} bounds:")
            combined_bounds[param_type] = {}

            for layer_pos, stats in param_stats[param_type].items():
                combined_bounds[param_type][layer_pos] = {
                    "min": stats["min"],
                    "max": stats["max"],
                    "mean": stats["mean"],
                    "std": stats["std"],
                }
                if param_type == "sld":
                    print(
                        f"  {layer_pos}: [{stats['min']:.2e}, {stats['max']:.2e}] (mean: {stats['mean']:.2e})"
                    )
                else:
                    print(
                        f"  {layer_pos}: [{stats['min']:.1f}, {stats['max']:.1f}] (mean: {stats['mean']:.1f})"
                    )

    prior_bounds["combined"] = combined_bounds

    return prior_bounds


def generate_reflectorch_prior_bounds(layer_count=2):
    """Generate prior bounds in reflectorch format for a specific layer count"""

    bounds_file = Path("maria_dataset_prior_bounds.json")
    if not bounds_file.exists():
        print("Please run extract_prior_bounds() first to generate the bounds file")
        return None

    with open(bounds_file, "r") as f:
        prior_bounds = json.load(f)

    layer_key = f"{layer_count}_layers"
    if layer_key not in prior_bounds:
        print(f"No data available for {layer_count}-layer experiments")
        return None

    data = prior_bounds[layer_key]

    print(f"\n" + "=" * 80)
    print(f"REFLECTORCH PRIOR BOUNDS FOR {layer_count}-LAYER EXPERIMENTS")
    print("=" * 80)

    if layer_count == 0:
        print("No layer parameters needed for 0-layer experiments")
        return None

    # Generate prior bounds list for reflectorch
    bounds_list = []
    param_names = []

    # Layer thicknesses (top to bottom)
    if "by_position" in data["parameters"]:
        print("Layer thicknesses (top to bottom):")
        for i in range(1, layer_count + 1):
            layer_key_pos = f"layer_{i}"
            if layer_key_pos in data["parameters"]["by_position"]:
                thickness = data["parameters"]["by_position"][layer_key_pos][
                    "thickness"
                ]
                bounds_list.append((thickness["min"], thickness["max"]))
                param_names.append(f"thickness_layer_{i}")
                print(
                    f"  Layer {i}: ({thickness['min']:.1f}, {thickness['max']:.1f}) - mean: {thickness['mean']:.1f}"
                )

    # Interlayer roughnesses (top to bottom) - always layer_count + 1 values
    print(f"\nInterlayer roughnesses (top to bottom) - {layer_count + 1} values:")
    if "interface" in data["parameters"]:
        # Fronting roughness
        front_rough = data["parameters"]["interface"]["fronting_roughness"]
        bounds_list.append((front_rough["min"], front_rough["max"]))
        param_names.append("roughness_fronting")
        print(
            f"  Fronting: ({front_rough['min']:.1f}, {front_rough['max']:.1f}) - mean: {front_rough['mean']:.1f}"
        )

        # Layer roughnesses
        if "by_position" in data["parameters"]:
            for i in range(1, layer_count + 1):
                layer_key_pos = f"layer_{i}"
                if layer_key_pos in data["parameters"]["by_position"]:
                    roughness = data["parameters"]["by_position"][layer_key_pos][
                        "roughness"
                    ]
                    bounds_list.append((roughness["min"], roughness["max"]))
                    param_names.append(f"roughness_layer_{i}")
                    print(
                        f"  Layer {i}: ({roughness['min']:.1f}, {roughness['max']:.1f}) - mean: {roughness['mean']:.1f}"
                    )

    # Real layer SLDs (top to bottom)
    print(f"\nReal layer SLDs (top to bottom):")
    if "by_position" in data["parameters"]:
        for i in range(1, layer_count + 1):
            layer_key_pos = f"layer_{i}"
            if layer_key_pos in data["parameters"]["by_position"]:
                sld = data["parameters"]["by_position"][layer_key_pos]["sld"]
                bounds_list.append((sld["min"], sld["max"]))
                param_names.append(f"sld_layer_{i}")
                print(
                    f"  Layer {i}: ({sld['min']:.2e}, {sld['max']:.2e}) - mean: {sld['mean']:.2e}"
                )

    # Add substrate SLD bound
    if "interface" in data["parameters"]:
        back_sld = data["parameters"]["interface"]["backing_sld"]
        bounds_list.append((back_sld["min"], back_sld["max"]))
        param_names.append("sld_substrate")
        print(
            f"  Substrate: ({back_sld['min']:.2e}, {back_sld['max']:.2e}) - mean: {back_sld['mean']:.2e}"
        )

    print(f"\nReflectorch prior_bounds format:")
    print("prior_bounds = [")
    for i, (bound, name) in enumerate(zip(bounds_list, param_names)):
        comma = "," if i < len(bounds_list) - 1 else ""
        print(f"    {bound}{comma}  # {name}")
    print("]")

    return bounds_list, param_names


def main():
    """Main function to extract and save prior bounds"""

    print("Extracting prior bounds from MARIA dataset analysis...")

    # Extract the bounds
    bounds = extract_prior_bounds()

    if bounds is None:
        print("Failed to extract bounds")
        return

    # Save to JSON file
    output_file = "maria_dataset_prior_bounds.json"
    with open(output_file, "w") as f:
        json.dump(bounds, f, indent=2)

    print(f"\nPrior bounds saved to: {output_file}")

    # Generate example reflectorch bounds for 2-layer experiments
    print(f"\n" + "=" * 80)
    print("EXAMPLE: REFLECTORCH FORMAT FOR 2-LAYER EXPERIMENTS")
    print("=" * 80)

    bounds_list, param_names = generate_reflectorch_prior_bounds(layer_count=2)

    print(f"\nExample usage in reflectorch:")
    print(f"prediction_dict = inference_model.predict(")
    print(f"    reflectivity_curve=exp_curve_interp,")
    print(f"    prior_bounds=prior_bounds,")
    print(f"    clip_prediction=False,")
    print(f"    polish_prediction=True,")
    print(f"    calc_pred_curve=True")
    print(f")")


if __name__ == "__main__":
    main()
