#!/usr/bin/env python3
"""
Fix the synthetic data generation to avoid negative reflectivity values
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from reflectorch import get_trainer_by_name

def debug_synthetic_data_generation():
    """Debug what's happening in the synthetic data generation"""
    
    print("DEBUGGING SYNTHETIC DATA GENERATION")
    print("="*60)
    
    # Load trainer
    config_name = "b_mc_point_neutron_conv_standard_L1_comp"
    trainer = get_trainer_by_name(config_name, load_weights=False)
    trainer.batch_size = 1
    trainer.loader.prior_sampler.num_layers = 1
    
    # Generate one batch
    raw_batch_data = trainer.loader.get_batch(1)
    
    print("Available keys in raw_batch_data:")
    for key in raw_batch_data.keys():
        print(f"  {key}: {raw_batch_data[key].shape if hasattr(raw_batch_data[key], 'shape') else type(raw_batch_data[key])}")
    
    # Extract different curve types
    q = raw_batch_data['q_values'][0].cpu().numpy()
    
    # Check what curve options are available
    curve_keys = [k for k in raw_batch_data.keys() if 'curve' in k]
    print(f"\nAvailable curve types: {curve_keys}")
    
    for curve_key in curve_keys:
        if curve_key in raw_batch_data:
            curve_data = raw_batch_data[curve_key][0].cpu().numpy()
            print(f"\n{curve_key}:")
            print(f"  Shape: {curve_data.shape}")
            print(f"  Range: {np.min(curve_data):.6f} to {np.max(curve_data):.6f}")
            print(f"  Negative values: {np.sum(curve_data < 0)} / {len(curve_data)} ({100*np.sum(curve_data < 0)/len(curve_data):.1f}%)")
            print(f"  Zero values: {np.sum(curve_data == 0)} / {len(curve_data)}")
            
            # Plot this curve type
            plt.figure(figsize=(10, 6))
            plt.semilogy(q, np.abs(curve_data), 'o-', label=f'{curve_key} (abs)', alpha=0.7)
            if np.any(curve_data < 0):
                plt.plot(q[curve_data < 0], np.abs(curve_data[curve_data < 0]), 'ro', 
                        label=f'Negative values', markersize=8)
            plt.xlabel('Q (Å⁻¹)')
            plt.ylabel('Reflectivity')
            plt.title(f'Synthetic Data - {curve_key}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f'debug_{curve_key}.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved plot: debug_{curve_key}.png")

def fix_synthetic_data_generation():
    """Generate corrected synthetic data"""
    
    print("GENERATING CORRECTED SYNTHETIC DATA")
    print("="*60)
    
    config_name = "b_mc_point_neutron_conv_standard_L1_comp"
    trainer = get_trainer_by_name(config_name, load_weights=False)
    trainer.batch_size = 1
    trainer.loader.prior_sampler.num_layers = 1
    
    # Generate a few test experiments
    out_dir = Path("corrected_synthetic_data") / config_name
    maria_dir = out_dir / "MARIA_VIPR_dataset"
    layer_dir = maria_dir / "1"
    layer_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(3):  # Generate 3 test experiments
        raw_batch_data = trainer.loader.get_batch(1)
        
        # Extract data
        q = raw_batch_data['q_values'][0].cpu().numpy()
        
        # Try different curve types to find the best one
        if 'clean_curves' in raw_batch_data:
            r = raw_batch_data['clean_curves'][0].cpu().numpy()
            curve_type = "clean_curves"
        elif 'noisy_curves' in raw_batch_data:
            r = raw_batch_data['noisy_curves'][0].cpu().numpy()  
            curve_type = "noisy_curves"
        else:
            # Use scaled_noisy_curves but handle negative values
            r = raw_batch_data['scaled_noisy_curves'][0].cpu().numpy()
            curve_type = "scaled_noisy_curves"
        
        print(f"\nExperiment {i+1}: Using {curve_type}")
        print(f"  Q range: {np.min(q):.4f} to {np.max(q):.4f}")
        print(f"  R range: {np.min(r):.6f} to {np.max(r):.6f}")
        print(f"  Negative values: {np.sum(r < 0)} / {len(r)} ({100*np.sum(r < 0)/len(r):.1f}%)")
        
        # Handle negative values
        if np.any(r < 0):
            print(f"  WARNING: Found negative reflectivity values!")
            # Option 1: Use absolute values
            r = np.abs(r)
            print(f"  Fixed using abs(). New range: {np.min(r):.6f} to {np.max(r):.6f}")
        
        # Generate realistic error bars (5-10% of signal)
        sig = 0.05 * r + 1e-6  # 5% relative + small constant
        
        # Get parameter information
        params_obj = raw_batch_data['params']
        param_names = trainer.loader.prior_sampler.param_model.get_param_labels()
        param_values = params_obj.parameters[0].cpu().numpy()
        
        print(f"  Parameters: {dict(zip(param_names, param_values))}")
        
        exp_id = f"synth_1_{i:03d}"
        
        # Save experimental curve
        save_curve = np.vstack([q, r, sig]).T
        curve_file = layer_dir / f"{exp_id}_experimental_curve.dat"
        np.savetxt(curve_file, save_curve)
        
        # Save parameters in MARIA format (using your existing code)
        with open(layer_dir / f"{exp_id}_model.txt", "w") as f:
            f.write("#layer        sld(A^-2)   thickness(A) roughness(A)\n")
            
            # Parse parameter names
            roughness_values = []
            thickness_values = []
            sld_values = []
            
            for name, val in zip(param_names, param_values):
                if 'Roughness' in name:
                    roughness_values.append((name, val))
                elif 'Thickness' in name:
                    thickness_values.append((name, val))
                elif 'SLD' in name:
                    sld_values.append((name, val))
            
            # Write layers
            ambient_roughness = roughness_values[0][1] if roughness_values else 10.0
            f.write(f"fronting      0.00000e+00      inf      {ambient_roughness:.2f}\n")
            
            thick_val = thickness_values[0][1] if thickness_values else 100.0
            sld_val = next((val for name, val in sld_values if 'L1' in name), 1e-6)
            rough_val = next((val for name, val in roughness_values if 'sub' in name), 10.0)
            
            # Convert SLD to MARIA format
            sld_scientific = sld_val / 1e6
            f.write(f"layer1       {sld_scientific:.5e}      {thick_val:.2f}      {rough_val:.2f}\n")
            
            substrate_sld = next((val for name, val in sld_values if 'sub' in name), 0.0)
            substrate_sld_scientific = substrate_sld / 1e6
            f.write(f"backing       {substrate_sld_scientific:.5e}      inf       none\n")
        
        print(f"  Saved: {curve_file}")
        print(f"  Saved: {layer_dir / f'{exp_id}_model.txt'}")
    
    print(f"\\nCorrected synthetic data saved to: {out_dir}")
    return out_dir

if __name__ == "__main__":
    # First debug what's happening
    debug_synthetic_data_generation()
    
    # Then generate corrected data
    corrected_dir = fix_synthetic_data_generation()
    
    print("\\n" + "="*60)
    print("DEBUGGING COMPLETE")
    print("="*60)
    print("Now test the corrected synthetic data with:")
    print(f"  debug_single_experiment('synth_1_000', '{corrected_dir}', 1)")
