#!/usr/bin/env python3
"""
Inference on PTCDI-C3 experimental data using custom priors and X-ray model.
"""

import torch
import numpy as np
from reflectorch import EasyInferenceModel
from simple_pipeline import load_experimental_data, run_inference, display_results
from plotting_utils import plot_simple_comparison

# Set seed for reproducibility
torch.manual_seed(42)

def main():
    """Run inference on PTCDI-C3 experimental data."""
    
    # Configuration
    data_file = "exp_data/data_PTCDI-C3.txt"
    model_name = 'b_mc_point_xray_conv_standard_L2_InputQ'
    layer_count = 2
    
    # Custom prior bounds for 2-layer system
    prior_bounds = [
        (1., 400.), (1., 10.),           # layer thicknesses (top to bottom)
        (0., 30.), (0., 30.), (0., 30.), # interlayer roughnesses (top to bottom)
        (10., 13.), (20., 21.), (20., 21.)  # real layer slds (top to bottom)
    ]
    
    print("="*60)
    print("PTCDI-C3 Experimental Data Inference")
    print("="*60)
    print(f"Data file: {data_file}")
    print(f"Model: {model_name}")
    print(f"Layer count: {layer_count}")
    print(f"Prior bounds: {prior_bounds}")
    print("="*60)
    
    # Load experimental data using simple_pipeline function
    q_exp, curve_exp, sigmas_exp = load_experimental_data(
        data_file,
        enable_preprocessing=True
    )
    
    # Initialize inference model
    print(f"\nInitializing inference model: {model_name}")
    inference_model = EasyInferenceModel(config_name=model_name, device='cpu')
    
    # Run inference using simple_pipeline function (matching reference parameters)
    q_model, prediction_dict = run_inference(
        inference_model, q_exp, curve_exp, prior_bounds, 
        # q_resolution=0.1, 
        apply_constraints=True,
        clip_prediction=True,
        use_q_shift=False
    )
    
    # Display results using simple_pipeline function
    display_results(prediction_dict)
    
    # Print detailed comparison with reference polished values
    print("\n" + "="*60)
    print("Comparison with Reference Polished Values")
    print("="*60)
    
    pred_params = prediction_dict['predicted_params_array']
    polished_params = prediction_dict['polished_params_array']
    param_names = prediction_dict["param_names"]
    
    # Reference polished values
    reference_values = {
        'Thickness L2': 188.75,
        'Thickness L1': 1.00,
        'Roughness L2': 22.15,
        'Roughness L1': 3.62,
        'Roughness sub': 29.91,
        'SLD L2': 12.44,
        'SLD L1': 21.00,
        'SLD sub': 21.00
    }
    
    print("\nParameter           | Predicted | Polished | Ref Polished")
    print("-" * 60)
    
    for i, param_name in enumerate(param_names):
        pred_val = pred_params[i]
        polished_val = polished_params[i]
        
        # Find matching reference value
        ref_val = None
        for ref_name, ref_value in reference_values.items():
            if param_name in ref_name or ref_name.lower().replace(' ', '_') in param_name.lower():
                ref_val = ref_value
                break
        
        if ref_val is not None:
            print(f'{param_name.ljust(18)} | {pred_val:9.2f} | {polished_val:9.2f} | {ref_val:9.2f}')
        else:
            print(f'{param_name.ljust(18)} | {pred_val:9.2f} | {polished_val:9.2f} | {"N/A":>9}')

    
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
        experiment_name="PTCDI-C3",
        show=True
    )
    
    print("\nInference completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
