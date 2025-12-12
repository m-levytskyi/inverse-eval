#!/usr/bin/env python3
"""
Inference on ORSO_example experimental data using reference parameters.
Based on: https://github.com/alexandros-koutsioumpas/anaklasis/blob/main/examples-Jupyter/ORSO_example.ipynb
"""

import torch
import numpy as np
from reflectorch import EasyInferenceModel
from simple_pipeline import load_experimental_data, run_inference, display_results
from plotting_utils import plot_simple_comparison

# Set seed for reproducibility
torch.manual_seed(42)

def main():
    """Run inference on ORSO_example experimental data."""
    
    # Configuration
    data_file = "exp_data/ORSO_example.ort"
    model_name = 'b_mc_point_neutron_conv_standard_L2_InputQDq'
    layer_count = 2
    
    # Prior bounds for 2-layer system (Si/SiO2/Polymer/D2O)
    # Based on notebook: p0, p1, p2, p3, p4, p5
    # p0: Si_SiO2_roughness (0, 5)
    # p1: SiO2_thickness (0, 100)
    # p2: SiO2_Polymer_roughness (0, 50)
    # p3: Polymer_sld (-0.5e-6, 3.0e-6)
    # p4: Polymer_thickness (0, 700)
    # p5: Polymer_D2O_roughness (0, 80)
    prior_bounds = [
        (0., 100.), (0., 700.),        # layer thicknesses (SiO2, Polymer)
        (0., 5.), (0., 50.), (0., 80.), # roughnesses (Si/SiO2, SiO2/Polymer, Polymer/D2O)
        (3.47, 3.47), (-0.5, 3.0), (6.35, 6.35)  # SLDs (SiO2 fixed, Polymer, D2O fixed)
    ]
    
    # Reference polished values from notebook
    reference_values = {
        'Si_SiO2_roughness': 4.52,
        'SiO2_thickness': 38.00,
        'SiO2_Polymer_roughness': 9.84,
        'Polymer_sld': 2.43,
        'Polymer_thickness': 259.25,
        'Polymer_D2O_roughness': 4.16,
    }
    
    print("="*60)
    print("ORSO Example Experimental Data Inference")
    print("="*60)
    print(f"Data file: {data_file}")
    print(f"Model: {model_name}")
    print(f"Layer count: {layer_count}")
    print(f"Prior bounds: {prior_bounds}")
    print("="*60)
    
    # Load experimental data using simple_pipeline function
    q_exp, curve_exp, sigmas_exp = load_experimental_data(
        data_file,
        enable_preprocessing=False
    )
    
    # Initialize inference model
    print(f"\nInitializing inference model: {model_name}")
    inference_model = EasyInferenceModel(config_name=model_name, device='cpu')
    
    # Run inference using simple_pipeline function (matching reference parameters)
    q_model, prediction_dict = run_inference(
        inference_model, q_exp, curve_exp, prior_bounds, 
        q_resolution=0.1, 
        apply_constraints=True,
        clip_prediction=True,
        use_q_shift=False
    )
    
    # Display results using simple_pipeline function
    display_results(prediction_dict)
    
    # Print detailed comparison with reference polished values
    print("\n" + "="*60)
    print("Comparison with Reference Polished Values (MCMC)")
    print("="*60)
    
    pred_params = prediction_dict['predicted_params_array']
    polished_params = prediction_dict['polished_params_array']
    param_names = prediction_dict["param_names"]
    
    print("\nParameter           | Predicted | Polished | Ref Polished")
    print("-" * 60)
    
    # Map Reflectorch parameter names to reference names
    # Reflectorch uses L2 for top layer (SiO2), L1 for middle layer (Polymer)
    param_mapping = {
        'Thickness L2': 'SiO2_thickness',
        'Thickness L1': 'Polymer_thickness', 
        'Roughness L2': 'Si_SiO2_roughness',
        'Roughness L1': 'SiO2_Polymer_roughness',
        'Roughness sub': 'Polymer_D2O_roughness',
        'SLD L2': None,  # SiO2 SLD is fixed
        'SLD L1': 'Polymer_sld',
        'SLD sub': None,  # D2O SLD is fixed
    }
    
    for i, param_name in enumerate(param_names):
        pred_val = pred_params[i]
        polished_val = polished_params[i]
        
        # Find matching reference value using mapping
        ref_key = param_mapping.get(param_name)
        ref_val = reference_values.get(ref_key) if ref_key else None
        
        if ref_val is not None:
            print(f'{param_name.ljust(18)} | {pred_val:9.2e} | {polished_val:9.2e} | {ref_val:9.2e}')
        else:
            print(f'{param_name.ljust(18)} | {pred_val:9.2e} | {polished_val:9.2e} | {"(fixed)":>9}')
    
    # Create plots
    print("\nGenerating comparison plots...")
    plot_simple_comparison(
        q_exp, curve_exp, sigmas_exp,
        q_model, 
        prediction_dict['predicted_curve'],
        prediction_dict['polished_curve'],
        prediction_dict['predicted_sld_xaxis'],
        prediction_dict['predicted_sld_profile'],
        prediction_dict['sld_profile_polished'],
        experiment_name="ORSO_example",
        show=True
    )
    
    print("\nInference completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
