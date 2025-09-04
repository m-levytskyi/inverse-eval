#!/usr/bin/env python3
"""
Custom batch inference script for two 1-layer experiments with custom priors.
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np

from inference_pipeline import InferencePipeline
from sld_profile_utils import sld_profile

# Define experiments, model, and two sets of custom priors
# experiments = ["s003981", "s005888"]
experiments = ["s003981", "s005888", "s004934"] #s006965
models_list = ["b_mc_point_neutron_conv_standard_L1_InputQDq"]

# Define custom priors for each experiment (order matches experiments list)
custom_priors_list = [
    [# s003981
        [0, 500.0],      # L1 thickness (Å)
        [0.5, 30.0],         # ambient/L1 roughness (Å)
        [0.5, 250.0],        # L1/substrate roughness (Å)
        [4.71637e-06, 4.71637e-06],      # L1 SLD (×10⁻⁶ Å⁻²)
        [-4.77301e-07, -4.77301e-07]       # substrate SLD (×10⁻⁶ Å⁻²)
    ],
    [# s005888 - as in GUI
        [142.0, 427.0],
        [3.5, 10.5],
        [14.0, 42.0],
        [11e-06, 16e-06],
        [11e-06, 16e-06]
    ],
    [# s004934 
        [0, 500.0],
        [0.5, 30.0],
        [0.5, 60.0],
        [-1.48075e-07, -1.48075e-07],
        [1.92389e-06, 1.92389e-06]
    ]
]

print("--- Initial Custom Priors Defined ---")
for i, priors in enumerate(custom_priors_list):
    print(f"  Priors for experiment {i+1}:")
    print(f"    ambient/L1 roughness (Å): {priors[1]}")
    print(f"    L1/substrate roughness (Å): {priors[2]}")
print("------------------------------------")

# Output directory for all batch results
output_base = "batch_custom_results"

os.makedirs(output_base, exist_ok=True)

results = {}
for exp_id, custom_priors in zip(experiments, custom_priors_list):
    print(f"\nRunning inference for experiment: {exp_id}")
    print(f"--- Using Custom Priors for {exp_id} ---")
    print(f"  ambient/L1 roughness (Å): {custom_priors[1]}")
    print(f"  L1/substrate roughness (Å): {custom_priors[2]}")
    print("------------------------------------")
    exp_outdir = os.path.join(output_base, exp_id)
    os.makedirs(exp_outdir, exist_ok=True)
    pipeline = InferencePipeline(
        experiment_id=exp_id,
        models_list=models_list,
        data_directory="data",
        priors_type="custom",
        custom_priors=custom_priors,
        preprocess=False,
        layer_count=1,
        output_dir=exp_outdir
    )
    pipeline.run_all_models(show_plots=False)
    results[exp_id] = pipeline.results
    print(f"Results for {exp_id} saved in {exp_outdir}")

    # --- SLD profile plotting ---
    # Get predicted parameters and parameter names from the result
    model_name = models_list[0]
    result = pipeline.results[model_name]
    param_names = result['param_names']
    predicted_params = result['predicted_params']
    print(f"\n--- Roughness Tracking for {exp_id} ---")
    print(f"Predicted (polished) parameters: {predicted_params}")



    # --- Generate SLD profile from predicted parameters using sld_profile utility ---
    # For 1-layer: [thickness, amb_rough, sub_rough, layer_sld, sub_sld]
    t = float(predicted_params[0])
    amb_rough = float(predicted_params[1])
    sub_rough = float(predicted_params[2])
    print(f"  - Predicted ambient/L1 roughness: {amb_rough:.4f} Å")
    print(f"  - Predicted L1/substrate roughness: {sub_rough:.4f} Å")
    sld_layer = float(predicted_params[3])
    sld_sub = float(predicted_params[4])
    # SLDs: [ambient, layer, substrate] (ambient assumed 0)
    slds_pred = [0.0, sld_layer, sld_sub]
    thicknesses_pred = [t]
    roughnesses_pred = [amb_rough, sub_rough]
    print(f"  - Roughness values passed to sld_profile (predicted): {roughnesses_pred}")
    z = np.linspace(0, t + 50, 400)
    interfaces_pred = [0] + list(np.cumsum(thicknesses_pred))
    sld_prof_pred = sld_profile(z, slds_pred, interfaces_pred, roughnesses_pred)
    print(f"Predicted SLD profile for {exp_id}: slds={slds_pred}, thicknesses={thicknesses_pred}, roughnesses={roughnesses_pred}")

    # Ask user for parameters from other model
    print(f"Enter parameters from other model for {exp_id} (comma-separated, order: {', '.join(param_names)}):")
    user_input = input().strip()
    try:
        other_params = [float(x) for x in user_input.split(",")]
    except Exception:
        print("Invalid input, skipping other model SLD plot.")
        other_params = None



    # --- User-input SLD profile ---
    sld_prof_other = None
    if other_params is not None and len(other_params) == 5:
        t_o = float(other_params[0])
        amb_rough_o = float(other_params[1])
        sub_rough_o = float(other_params[2])
        print(f"  - User-input ambient/L1 roughness: {amb_rough_o:.4f} Å")
        print(f"  - User-input L1/substrate roughness: {sub_rough_o:.4f} Å")
        sld_layer_o = float(other_params[3])
        sld_sub_o = float(other_params[4])
        slds_other = [0.0, sld_layer_o, sld_sub_o]
        thicknesses_other = [t_o]
        roughnesses_other = [amb_rough_o, sub_rough_o]
        print(f"  - Roughness values from user input: {roughnesses_other}")
        z_other = np.linspace(0, t_o + 50, 400)
        interfaces_other = [0] + list(np.cumsum(thicknesses_other))
        sld_prof_other = sld_profile(z_other, slds_other, interfaces_other, roughnesses_other)
        print(f"User-input SLD profile for {exp_id}: slds={slds_other}, thicknesses={thicknesses_other}, roughnesses={roughnesses_other}")


    # --- Ground truth SLD profile ---
    # Try to find and parse the *_model.txt file for ground truth parameters
    model_file = None
    # Try common locations
    possible_paths = [
        os.path.join("data", "MARIA_VIPR_dataset", "1", f"{exp_id}_model.txt"),
        os.path.join("data", "MARIA_VIPR_dataset", "2", f"{exp_id}_model.txt"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            model_file = path
            break


    # --- Ground truth SLD profile ---
    z_gt = None
    sld_prof_gt = None
    if model_file:
        try:
            with open(model_file, 'r') as f:
                lines = f.readlines()
            layers = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = re.split(r'\s+', line)
                if len(parts) >= 4:
                    name = parts[0]
                    try:
                        sld = float(parts[1])
                    except Exception:
                        print(f"Warning: Invalid SLD value '{parts[1]}' in {exp_id}, setting to 0.")
                        sld = 0.0
                    try:
                        thickness = float(parts[2]) if parts[2] not in ['inf', 'none', 'NaN'] else 0.0
                    except Exception:
                        print(f"Warning: Invalid thickness value '{parts[2]}' in {exp_id}, setting to 0.")
                        thickness = 0.0
                    try:
                        roughness = float(parts[3]) if parts[3] not in ['none', 'inf', 'NaN'] else 3.0
                    except Exception:
                        print(f"Warning: Invalid roughness value '{parts[3]}' in {exp_id}, setting to 3.0.")
                        roughness = 3.0
                    layers.append({'name': name, 'sld': sld, 'thickness': thickness, 'roughness': roughness})
            if len(layers) == 3:
                # 1-layer: fronting, layer1, backing
                # Only use the film layer thickness (ignore substrate/ambient thickness)
                t_val_raw = layers[1]['thickness']
                t_val_str = str(t_val_raw).strip().lower() if t_val_raw is not None else ''
                try:
                    t_val = float(t_val_raw)
                except Exception:
                    t_val = None
                if t_val is None or t_val_str in ['none', 'inf', 'nan'] or t_val <= 0:
                    print(f"Warning: Film layer thickness is invalid ('{t_val_raw}') in {exp_id}, skipping ground truth SLD plot.")
                    sld_prof_gt = None
                else:
                    print(f"Debug: Using film layer thickness {t_val} for {exp_id}")
                    t_gt = t_val
                    slds_gt = [layers[0]['sld'], layers[1]['sld'], layers[2]['sld']]
                    thicknesses_gt = [t_gt]  # Only the film layer
                    # Roughness from layer1 is ambient/L1, from layer2 (backing) is L1/substrate
                    gt_amb_rough = layers[1]['roughness'] if layers[1]['roughness'] is not None else 3.0
                    gt_sub_rough = layers[2]['roughness'] if layers[2]['roughness'] is not None else 3.0
                    print(f"  - Ground truth ambient/L1 roughness: {gt_amb_rough:.4f} Å (from layer '{layers[1]['name']}')")
                    print(f"  - Ground truth L1/substrate roughness: {gt_sub_rough:.4f} Å (from layer '{layers[2]['name']}')")
                    roughnesses_gt = [gt_amb_rough, gt_sub_rough]
                    print(f"  - Roughness values passed to sld_profile (ground truth): {roughnesses_gt}")
                    z_gt = np.linspace(0, t_gt + 50, 400)
                    interfaces_gt = [0] + list(np.cumsum(thicknesses_gt))
                    sld_prof_gt = sld_profile(z_gt, slds_gt, interfaces_gt, roughnesses_gt)
                    print(f"Ground truth SLD profile for {exp_id}: slds={slds_gt}, thicknesses={thicknesses_gt}, roughnesses={roughnesses_gt}")
            elif len(layers) == 4:
                # 2-layer: combine thicknesses for 1-layer equivalent
                t1_val_raw = layers[1]['thickness']
                t2_val_raw = layers[2]['thickness']
                t1_val_str = str(t1_val_raw).strip().lower() if t1_val_raw is not None else ''
                t2_val_str = str(t2_val_raw).strip().lower() if t2_val_raw is not None else ''
                try:
                    t1 = float(t1_val_raw)
                except Exception:
                    t1 = None
                try:
                    t2 = float(t2_val_raw)
                except Exception:
                    t2 = None
                if (t1 is None or t1_val_str in ['none', 'inf', 'nan'] or t1 <= 0 or
                    t2 is None or t2_val_str in ['none', 'inf', 'nan'] or t2 <= 0):
                    print(f"Warning: One or both film layer thicknesses are invalid ('{t1_val_raw}', '{t2_val_raw}') in {exp_id}, skipping ground truth SLD plot.")
                    sld_prof_gt = None
                else:
                    print(f"Debug: Using film layer thicknesses {t1}, {t2} for {exp_id}")
                    t_gt = t1 + t2
                    weighted_sld = (
                        layers[1]['sld'] * t1 +
                        layers[2]['sld'] * t2
                    ) / t_gt if t_gt > 0 else 0.0
                    slds_gt = [layers[0]['sld'], weighted_sld, layers[3]['sld']]
                    thicknesses_gt = [t_gt]
                    # For 2-layer GT, we take roughness from layer1 and layer2 (film layers)
                    gt_amb_rough = layers[1]['roughness'] if layers[1]['roughness'] is not None else 3.0
                    gt_sub_rough = layers[2]['roughness'] if layers[2]['roughness'] is not None else 3.0
                    print(f"  - Ground truth ambient/L1 roughness: {gt_amb_rough:.4f} Å (from layer '{layers[1]['name']}')")
                    print(f"  - Ground truth L1/substrate roughness: {gt_sub_rough:.4f} Å (from layer '{layers[2]['name']}')")
                    roughnesses_gt = [gt_amb_rough, gt_sub_rough]
                    print(f"  - Roughness values passed to sld_profile (ground truth): {roughnesses_gt}")
                    z_gt = np.linspace(0, t_gt + 50, 400)
                    interfaces_gt = [0] + list(np.cumsum(thicknesses_gt))
                    sld_prof_gt = sld_profile(z_gt, slds_gt, interfaces_gt, roughnesses_gt)
                    print(f"Ground truth SLD profile for {exp_id}: slds={slds_gt}, thicknesses={thicknesses_gt}, roughnesses={roughnesses_gt}")
        except Exception as e:
            print(f"Could not parse ground truth model file for {exp_id}: {e}")
    print("------------------------------------")


    # Plot all SLD profiles (ensure all are plotted if available)
    fig, ax = plt.subplots(figsize=(7,5))
    n_plotted = 0
    if sld_prof_pred is not None:
        ax.plot(z, sld_prof_pred * 1e6, label="Predicted", color="blue")
        print(f"Plotted predicted SLD for {exp_id}.")
        n_plotted += 1
    if sld_prof_other is not None:
        ax.plot(z_other, sld_prof_other * 1e6, label="From GUI", color="red")
        print(f"Plotted user-input SLD for {exp_id}.")
        n_plotted += 1
    if sld_prof_gt is not None:
        ax.plot(z_gt, sld_prof_gt * 1e6, label="Ground Truth", color="green")
        print(f"Plotted ground truth SLD for {exp_id}.")
        n_plotted += 1
    if n_plotted == 0:
        print(f"No SLD profiles available to plot for {exp_id}.")
    ax.set_xlabel('Depth [Å]')
    ax.set_ylabel('SLD [$10^{-6}$ Å$^{-2}$]')
    ax.invert_yaxis()
    ax.legend()
    ax.set_title(f"SLD Profile for {exp_id}")

    # --- Show SLD parameter table in the plot ---
    # Prepare table data (reuse table_rows from below, or build here)
    import csv
    table_rows = []
    # Predicted
    table_rows.append([
        "Predicted",
        f"{predicted_params[0]:.4f}",
        f"{predicted_params[1]:.4f}",
        f"{predicted_params[2]:.4f}",
        f"{(predicted_params[3] * 1e6):.6f}",
        f"{(predicted_params[4] * 1e6):.6f}"
    ])
    # User-input (other model)
    if other_params is not None and len(other_params) == 5:
        table_rows.append([
            "Other Model",
            f"{other_params[0]:.4f}",
            f"{other_params[1]:.4f}",
            f"{other_params[2]:.4f}",
            f"{(other_params[3] * 1e6):.6f}",
            f"{(other_params[4] * 1e6):.6f}"
        ])
    # Ground truth
    if model_file and sld_prof_gt is not None:
        # For 1-layer: layers[1] is the film
        if len(layers) == 3:
            gt_layer = layers[1]
            table_rows.append([
                "Ground Truth",
                f"{gt_layer['thickness']:.4f}",
                f"{gt_layer['roughness']:.4f}",
                f"{layers[2]['roughness']:.4f}",
                f"{(gt_layer['sld'] * 1e6):.6f}",
                f"{(layers[2]['sld'] * 1e6):.6f}"
            ])
        elif len(layers) == 4:
            # For 2-layer, report combined thickness, weighted SLD, and roughnesses
            t1 = layers[1]['thickness']
            t2 = layers[2]['thickness']
            try:
                t1f = float(t1)
            except Exception:
                t1f = 0.0
            try:
                t2f = float(t2)
            except Exception:
                t2f = 0.0
            t_gt = t1f + t2f
            weighted_sld = (
                layers[1]['sld'] * t1f +
                layers[2]['sld'] * t2f
            ) / t_gt if t_gt > 0 else 0.0
            table_rows.append([
                "Ground Truth",
                f"{t_gt:.4f}",
                f"{layers[1]['roughness']:.4f}",
                f"{layers[2]['roughness']:.4f}",
                f"{(weighted_sld * 1e6):.6f}",
                f"{(layers[3]['sld'] * 1e6):.6f}"
            ])
    # Add table to plot
    col_labels = ["Source", "Thickness", "Amb_Rough", "Sub_Rough", "Layer_SLD [$10^{-6}$ Å$^{-2}$]", "Substrate_SLD [$10^{-6}$ Å$^{-2}$]"]
    table = ax.table(cellText=table_rows, colLabels=col_labels, loc='bottom', cellLoc='center', bbox=[0, -0.55, 1, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    plt.subplots_adjust(bottom=0.32)

    plot_path = os.path.join(exp_outdir, f"sld_profile_{exp_id}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"SLD profile plot saved to {plot_path}")

    # --- Save table of SLD parameters ---
    import csv
    table_rows = []
    # Predicted
    table_rows.append([
        "Predicted",
        f"{predicted_params[0]:.4f}",
        f"{predicted_params[1]:.4f}",
        f"{predicted_params[2]:.4f}",
        f"{(predicted_params[3] * 1e6):.6f}",
        f"{(predicted_params[4] * 1e6):.6f}"
    ])
    # User-input (other model)
    if other_params is not None and len(other_params) == 5:
        table_rows.append([
            "Other Model",
            f"{other_params[0]:.4f}",
            f"{other_params[1]:.4f}",
            f"{other_params[2]:.4f}",
            f"{(other_params[3] * 1e6):.6f}",
            f"{(other_params[4] * 1e6):.6f}"
        ])
    # Ground truth
    if model_file and sld_prof_gt is not None:
        # For 1-layer: layers[1] is the film
        if len(layers) == 3:
            gt_layer = layers[1]
            table_rows.append([
                "Ground Truth",
                f"{gt_layer['thickness']:.4f}",
                f"{gt_layer['roughness']:.4f}",
                f"{layers[2]['roughness']:.4f}",
                f"{(gt_layer['sld'] * 1e6):.6f}",
                f"{(layers[2]['sld'] * 1e6):.6f}"
            ])
        elif len(layers) == 4:
            # For 2-layer, report combined thickness, weighted SLD, and roughnesses
            t1 = layers[1]['thickness']
            t2 = layers[2]['thickness']
            try:
                t1f = float(t1)
            except Exception:
                t1f = 0.0
            try:
                t2f = float(t2)
            except Exception:
                t2f = 0.0
            t_gt = t1f + t2f
            weighted_sld = (
                layers[1]['sld'] * t1f +
                layers[2]['sld'] * t2f
            ) / t_gt if t_gt > 0 else 0.0
            table_rows.append([
                "Ground Truth",
                f"{t_gt:.4f}",
                f"{layers[1]['roughness']:.4f}",
                f"{layers[2]['roughness']:.4f}",
                f"{(weighted_sld * 1e6):.6f}",
                f"{(layers[3]['sld'] * 1e6):.6f}"
            ])
    # Write CSV
    table_path = os.path.join(exp_outdir, f"sld_parameters_{exp_id}.csv")
    with open(table_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Source", "Thickness", "Amb_Rough", "Sub_Rough", "Layer_SLD (x10^-6)", "Substrate_SLD (x10^-6)"])
        writer.writerows(table_rows)
    print(f"SLD parameter table saved to {table_path}")

print("\nBatch inference complete.")
