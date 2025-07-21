import os
import numpy as np
from pathlib import Path

# 1) Generate data
from reflectorch import get_trainer_by_name
# 2) Run inference
from batch_inference_pipeline import BatchInferencePipeline
# Patch the global narrow‐priors deviation
import inference_pipeline
inference_pipeline.NARROW_PRIORS_DEVIATION = 0.1  # ±10% for high accuracy on synthetic data

def generate_synthetic_data(trainer, layer_count, num_experiments, out_dir):
    """
    Uses the trainer's internal generator to dump num_experiments
    1‐curve batches to disk under out_dir/MARIA_VIPR_dataset/<layer_count>/
    """
    # Create the directory structure expected by BatchInferencePipeline
    maria_dir = Path(out_dir) / "MARIA_VIPR_dataset"
    layer_dir = maria_dir / str(layer_count)
    layer_dir.mkdir(parents=True, exist_ok=True)
    trainer.batch_size = 1
    # Set the actual number of layers
    trainer.loader.prior_sampler.num_layers = layer_count

    for i in range(num_experiments):
        # Get raw batch data from the loader
        raw_batch_data = trainer.loader.get_batch(1)
        
        # Extract data from the raw batch
        q = raw_batch_data['q_values'][0].cpu().numpy()
        r = raw_batch_data['scaled_noisy_curves'][0].cpu().numpy()
        
        # Fix negative reflectivity values from scaled curves
        if np.any(r < 0):
            print(f"  WARNING: Experiment {i} has {np.sum(r < 0)}/{len(r)} negative values. Using abs().")
            r = np.abs(r)
        
        # For sigmas, we need to generate some synthetic noise estimates
        # Let's use a realistic fraction of the curve values as sigmas
        sig = 0.05 * r + 1e-6  # 5% relative error + small constant
        
        # Get parameter names and values
        params_obj = raw_batch_data['params']
        param_names = trainer.loader.prior_sampler.param_model.get_param_labels()
        param_values = params_obj.parameters[0].cpu().numpy()

        exp_id = f"synth_{layer_count}_{i:03d}"
        # save experimental curve (Q, R, σ)
        save_curve = np.vstack([q, r, sig]).T
        np.savetxt(layer_dir / f"{exp_id}_experimental_curve.dat", save_curve)

        # save "true" model parameters in MARIA format
        with open(layer_dir / f"{exp_id}_model.txt", "w") as f:
            f.write("#layer        sld(A^-2)   thickness(A) roughness(A)\n")
            
            # Parse parameter names to determine structure
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
            
            # Sort parameters by layer (L1, L2, sub)
            thickness_values.sort(key=lambda x: 'L1' in x[0])  # L1 first, then L2
            sld_values.sort(key=lambda x: ('L1' in x[0], 'L2' in x[0], 'sub' in x[0]))
            
            # Write fronting (ambient) - always zero SLD
            ambient_roughness = next((val for name, val in roughness_values if 'L1' in name and len([r for r in roughness_values if 'L' in r[0]]) > 1), 
                                   roughness_values[0][1] if roughness_values else 10.0)
            f.write(f"fronting      0.00000e+00      inf      {ambient_roughness:.2f}\n")
            
            # Write layers based on number of thickness parameters
            # Single layer system (models are designed for 1 layer only)
            thick_name, thick_val = thickness_values[0]
            sld_name, sld_val = next((name, val) for name, val in sld_values if 'L1' in name)
            rough_val = next((val for name, val in roughness_values if 'sub' in name), 10.0)
            
            # Convert SLD from reflectorch units to MARIA format
            sld_scientific = sld_val / 1e6
            f.write(f"layer1       {sld_scientific:.5e}      {thick_val:.2f}      {rough_val:.2f}\n")
            
            # Write backing (substrate)
            substrate_sld = next((val for name, val in sld_values if 'sub' in name), 0.0)
            substrate_sld_scientific = substrate_sld / 1e6
            f.write(f"backing       {substrate_sld_scientific:.5e}      inf       none\n")

if __name__ == "__main__":
    # Root directory for synthetic data
    SYN_ROOT = Path("synthetic_data")
    # Trainer configurations to use
    CONFIGS = [
        "b_mc_point_neutron_conv_standard_L1_comp",
        "b_mc_point_neutron_conv_standard_L1_InputQDq",
    ]

    for config_name in CONFIGS:
        print(f"\n\n=== Using trainer config: {config_name} ===")
        trainer = get_trainer_by_name(config_name, load_weights=False)
        out_dir = SYN_ROOT / config_name
        # generate synthetic experiments for 1-layer only (models are single-layer)
        generate_synthetic_data(trainer, layer_count=1, num_experiments=100, out_dir=out_dir)

        # run inference on generated 1-layer data
        print(f"\n\n=== Running inference on 1-layer synthetic data for config {config_name} ===")
        pipeline = BatchInferencePipeline(
            num_experiments=100,
            layer_count=1,
            data_directory=str(out_dir),
            enable_parallel=True,
            max_workers=16,
            enable_caching=False
        )
        pipeline.run()
