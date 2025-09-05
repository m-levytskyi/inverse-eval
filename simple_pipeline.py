import torch
import numpy as np
import matplotlib.pyplot as plt
from reflectorch import EasyInferenceModel

# Set seed for reproducibility
torch.manual_seed(42)

def main():
    # Experiment configuration
    experiment_name = "s005888"
    
    print(f"Running inference for experiment: {experiment_name}")
    print("="*50)
    
    # Load experimental data
    data_path = f'data/MARIA_VIPR_dataset/1/{experiment_name}_experimental_curve.dat'
    data = np.loadtxt(data_path, skiprows=1)
    print(f"Data shape: {data.shape}")
    
    q_exp = data[..., 0]
    curve_exp = data[..., 1]
    sigmas_exp = data[..., 2]
    
    print(f"Q range: {q_exp.min():.4f} - {q_exp.max():.4f} Å⁻¹")
    print(f"Curve shape: {curve_exp.shape}")
    print(f"Relative error range: {(sigmas_exp / curve_exp).min():.4f} - {(sigmas_exp / curve_exp).max():.4f}")
    
    # Initialize inference model
    config_name = 'b_mc_point_neutron_conv_standard_L1_InputQDq'
    print(f"\nInitializing model: {config_name}")
    
    inference_model = EasyInferenceModel(config_name=config_name, device='cpu')
    
    # Interpolate data to model grid
    q_model, exp_curve_interp = inference_model.interpolate_data_to_model_q(q_exp, curve_exp)
    print(f"Model Q range: {q_model.min():.4f} - {q_model.max():.4f} Å⁻¹")
    print(f"Interpolated curve shape: {exp_curve_interp.shape}")
    
    # Set prior bounds for Ni on Si structure
    prior_bounds = [
        (142.0, 427.0),    # layer thickness
        (3.5, 10.5),       # fronting roughness
        (14.0, 42.0),      # layer roughness
        (11, 16),          # layer SLD
        (11, 16)           # backing SLD
    ]
    
    print("\nPrior bounds:")
    param_names = ['thickness', 'fronting_roughness', 'layer_roughness', 'layer_sld', 'backing_sld']
    for name, bounds in zip(param_names, prior_bounds):
        print(f"  {name}: {bounds}")
    
    # Perform prediction with dq/q = 0.1 (10% resolution)
    print("\nPerforming prediction...")
    prediction_dict = inference_model.predict(
        reflectivity_curve=exp_curve_interp,
        prior_bounds=prior_bounds,
        q_values=q_model,
        q_resolution=0.1,  # dq/q = 10%
        polish_prediction=True,
        calc_pred_curve=True,
        calc_pred_sld_profile=True,
        calc_polished_sld_profile=True,
    )
    
    # Display results
    print("\nPrediction Results:")
    print("-" * 40)
    pred_params = prediction_dict['predicted_params_array']
    polished_params = prediction_dict['polished_params_array']
    
    for param_name, pred_val, polished_val in zip(prediction_dict["param_names"], pred_params, polished_params):
        print(f'{param_name.ljust(14)} -> Predicted: {pred_val:.2f}       Polished: {polished_val:.2f}')
    
    # Create plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot reflectivity curves
    ax[0].set_yscale('log')
    ax[0].set_xlabel('q [Å⁻¹]', fontsize=14)
    ax[0].set_ylabel('R(q)', fontsize=14)
    ax[0].tick_params(axis='both', which='major', labelsize=12)
    
    # Experimental data with error bars
    el = ax[0].errorbar(q_exp, curve_exp, yerr=sigmas_exp, xerr=None, 
                        c='b', ecolor='purple', elinewidth=1, 
                        marker='o', linestyle='none', markersize=3, 
                        label='exp. curve', zorder=1)
    el.get_children()[1].set_color('purple')
    
    # Predicted curves
    ax[0].plot(q_model, prediction_dict['predicted_curve'], 
               c='red', lw=2, label='pred. curve')
    ax[0].plot(q_model, prediction_dict['polished_curve'], 
               c='orange', ls='--', lw=2, label='polished pred. curve')
    
    ax[0].legend(loc='upper right', fontsize=12)
    ax[0].set_title(f'Reflectivity - {experiment_name}', fontsize=14)
    
    # Plot SLD profiles
    ax[1].set_xlabel('z [Å]', fontsize=14)
    ax[1].set_ylabel('SLD [10⁻⁶ Å⁻²]', fontsize=14)
    ax[1].tick_params(axis='both', which='major', labelsize=12)
    
    ax[1].plot(prediction_dict['predicted_sld_xaxis'], 
               prediction_dict['predicted_sld_profile'], 
               c='red', label='predicted')
    ax[1].plot(prediction_dict['predicted_sld_xaxis'], 
               prediction_dict['sld_profile_polished'], 
               c='orange', ls='--', label='polished')
    
    ax[1].legend(loc='best', fontsize=12)
    ax[1].set_title('SLD Profile', fontsize=14)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nInference completed for experiment {experiment_name}")

if __name__ == "__main__":
    main()
