#!/usr/bin/env python3
"""
Compare SLD profiles from .dat file and calculated from model.txt parameters.

This script uses existing utilities to:
1. Load SLD profile from .dat file 
2. Parse parameters from model.txt file
3. Calculate SLD profile using existing sld_profile_utils
4. Plot both profiles for comparison using existing plotting utilities
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Import existing utilities
from sld_profile_utils import sld_profile
from plot_sld_profile import plot_sld_profile
from parameter_discovery import parse_true_parameters_from_model_file


def load_sld_dat_file(dat_file_path):
    """
    Load SLD profile from .dat file.
    
    Args:
        dat_file_path: Path to .dat file with columns: z (A), sld (10^-6 A^-2)
        
    Returns:
        Tuple of (z_values, sld_values) in Å and 10^-6 Å^-2 units
    """
    print(f"Loading SLD profile from: {dat_file_path}")
    
    if not Path(dat_file_path).exists():
        raise FileNotFoundError(f"DAT file not found: {dat_file_path}")
    
    # Load data, skipping comment lines
    data = np.loadtxt(dat_file_path)
    
    z_values = data[:, 0]  # Depth in Å
    sld_values = data[:, 1] * 1e6  # Convert to 10^-6 Å^-2 units for plotting
    
    print(f"Loaded {len(z_values)} data points")
    print(f"Depth range: {z_values.min():.1f} to {z_values.max():.1f} Å")
    print(f"SLD range: {sld_values.min():.2f} to {sld_values.max():.2f} ×10⁻⁶ Å⁻²")
    
    return z_values, sld_values


def calculate_sld_from_model(model_file_path, z_range=None, n_points=1000):
    """
    Calculate SLD profile from model.txt parameters using existing utilities.
    
    Args:
        model_file_path: Path to model.txt file
        z_range: Tuple of (min, max) depth values. If None, auto-determine
        n_points: Number of points for calculation
        
    Returns:
        Tuple of (z_values, sld_profile, params_dict) or (None, None, None) if parsing fails
    """
    print(f"Calculating SLD profile from model: {model_file_path}")
    
    # Parse true parameters using existing utility
    true_params_dict = parse_true_parameters_from_model_file(model_file_path)
    
    if not true_params_dict:
        print("Failed to parse model parameters")
        return None, None, None
    
    # Determine layer count and get parameters
    layer_key = '2_layer' if '2_layer' in true_params_dict else '1_layer'
    if layer_key not in true_params_dict:
        print("No valid layer interpretation found")
        return None, None, None
        
    params = true_params_dict[layer_key]['params']
    param_names = true_params_dict[layer_key]['param_names']
    
    print(f"Using {layer_key} model")
    print(f"Parameters: {dict(zip(param_names, params))}")
    
    # Extract parameters based on layer count
    if layer_key == '1_layer':
        # 1-layer: [thickness, amb_rough, sub_rough, layer_sld, sub_sld]
        thickness, amb_rough, sub_rough, layer_sld, sub_sld = params
        
        # Set up for sld_profile function
        # Convert SLD values back to Å^-2 units (they come as 10^-6 Å^-2 from parser)
        # SLDs: [ambient, layer, substrate]
        slds = [3.5e-6, layer_sld * 1e-6, sub_sld * 1e-6]  # Convert back to Å^-2
        interfaces = [0, thickness]  # Interface positions
        roughnesses = [amb_rough, sub_rough]  # Roughnesses at each interface
        
        total_thickness = thickness
        
    elif layer_key == '2_layer':
        # 2-layer: [thickness1, thickness2, amb_rough, int_rough, sub_rough, layer1_sld, layer2_sld, sub_sld]
        thickness1, thickness2, amb_rough, int_rough, sub_rough, layer1_sld, layer2_sld, sub_sld = params
        
        # Set up for sld_profile function
        # Convert SLD values back to Å^-2 units (they come as 10^-6 Å^-2 from parser)
        # SLDs: [ambient, layer1, layer2, substrate]
        slds = [3.5e-6, layer1_sld * 1e-6, layer2_sld * 1e-6, sub_sld * 1e-6]  # Convert back to Å^-2
        interfaces = [0, thickness1, thickness1 + thickness2]  # Interface positions
        roughnesses = [amb_rough, int_rough, sub_rough]  # Roughnesses at each interface
        
        total_thickness = thickness1 + thickness2
    
    # Set z_range if not provided
    if z_range is None:
        z_min = -total_thickness * 0.2  # Start before the structure
        z_max = total_thickness * 1.5   # Extend into substrate
        z_range = (z_min, z_max)
    
    # Create depth axis
    z_values = np.linspace(z_range[0], z_range[1], n_points)
    
    # Calculate SLD profile using existing utility
    sld_profile_calc = sld_profile(z_values, slds, interfaces, roughnesses)
    
    # The sld_profile function returns values in Å^-2, convert to 10^-6 Å^-2 units for plotting
    sld_profile_calc = sld_profile_calc * 1e6
    
    print(f"Calculated SLD profile over range {z_range[0]:.1f} to {z_range[1]:.1f} Å")
    print(f"SLD range: {sld_profile_calc.min():.2f} to {sld_profile_calc.max():.2f} ×10⁻⁶ Å⁻²")
    
    return z_values, sld_profile_calc, true_params_dict


def compare_sld_profiles(experiment_id, data_dir="./data/MARIA_VIPR_dataset/2", output_dir="./sld_plots/2", save=True, show=True):
    """
    Compare SLD profiles from .dat file and model.txt calculation.
    
    Args:
        experiment_id: Experiment identifier (e.g., 's000000')
        data_dir: Directory containing the data files
        output_dir: Directory to save output plots
        save: Whether to save the plot
        show: Whether to display the plot
        
    Returns:
        Figure object if successful, None otherwise
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # File paths
    dat_file = data_dir / f"{experiment_id}_sld_profile.dat"
    model_file = data_dir / f"{experiment_id}_model.txt"
    
    print(f"Comparing SLD profiles for experiment {experiment_id}")
    print("=" * 60)
    
    try:
        # Load data from .dat file
        z_dat, sld_dat = load_sld_dat_file(dat_file)
        
        # Calculate from model parameters
        z_calc, sld_calc, true_params_dict = calculate_sld_from_model(model_file, 
                                                                     z_range=(z_dat.min(), z_dat.max()),
                                                                     n_points=len(z_dat))
        
        if z_calc is None or sld_calc is None:
            print("Failed to calculate SLD profile from model")
            return None
            
        # Create comparison plot using existing utility
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot both profiles
        plot_sld_profile(z_dat, sld_dat, label='From sld_profile.dat', ax=ax, color='blue')
        plot_sld_profile(z_calc, sld_calc, label='Calculated from model.txt', ax=ax, color='red')
        
        # Customize plot
        ax.set_title(f'SLD Profile Comparison - {experiment_id}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add statistics and parameter information text
        info_text = ""
        
        # Add model parameters information
        if z_calc is not None and sld_calc is not None and true_params_dict:
            layer_key = '2_layer' if '2_layer' in true_params_dict else '1_layer'
            if layer_key in true_params_dict:
                params = true_params_dict[layer_key]['params']
                param_names = true_params_dict[layer_key]['param_names']
                
                # Format parameters for display
                info_text += f"Model Parameters ({layer_key}):\n"
                
                if layer_key == '1_layer':
                    thickness, amb_rough, sub_rough, layer_sld, sub_sld = params
                    info_text += f"  Ambient SLD: 3.50 ×10⁻⁶ Å⁻²\n"
                    info_text += f"  Layer thickness: {thickness:.1f} Å\n"
                    info_text += f"  Layer SLD: {layer_sld:.2f} ×10⁻⁶ Å⁻²\n"
                    info_text += f"  Substrate SLD: {sub_sld:.2f} ×10⁻⁶ Å⁻²\n"
                    info_text += f"  Amb roughness: {amb_rough:.1f} Å\n"
                    info_text += f"  Sub roughness: {sub_rough:.1f} Å\n"
                elif layer_key == '2_layer':
                    thickness1, thickness2, amb_rough, int_rough, sub_rough, layer1_sld, layer2_sld, sub_sld = params
                    info_text += f"  Ambient SLD: 3.50 ×10⁻⁶ Å⁻²\n"
                    info_text += f"  Layer1 thickness: {thickness1:.1f} Å\n"
                    info_text += f"  Layer1 SLD: {layer1_sld:.2f} ×10⁻⁶ Å⁻²\n"
                    info_text += f"  Layer2 thickness: {thickness2:.1f} Å\n"
                    info_text += f"  Layer2 SLD: {layer2_sld:.2f} ×10⁻⁶ Å⁻²\n"
                    info_text += f"  Substrate SLD: {sub_sld:.2f} ×10⁻⁶ Å⁻²\n"
                    info_text += f"  Amb roughness: {amb_rough:.1f} Å\n"
                    info_text += f"  Interface roughness: {int_rough:.1f} Å\n"
                    info_text += f"  Sub roughness: {sub_rough:.1f} Å\n"
        
        # Calculate RMS difference over overlapping range
        if len(z_dat) == len(z_calc):
            diff = sld_dat - sld_calc
            rms_diff = np.sqrt(np.mean(diff**2))
            max_diff = np.max(np.abs(diff))
            
            if info_text:
                info_text += "\n"
            info_text += f'RMS difference: {rms_diff:.3f} ×10⁻⁶ Å⁻²\n'
            info_text += f'Max difference: {max_diff:.3f} ×10⁻⁶ Å⁻²'
        
        # Add the combined text box in the right top corner
        if info_text:
            ax.text(0.98, 0.98, info_text, transform=ax.transAxes, 
                   ha='right', va='top', fontsize=9, 
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        
        # Save plot if requested
        if save:
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_path = output_dir / f"{experiment_id}_sld_comparison.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {plot_path}")
        
        # Show plot if requested
        if show:
            plt.show()
        
        print("SLD profile comparison completed successfully")
        return fig
        
    except Exception as e:
        print(f"Error during SLD profile comparison: {e}")
        return None


def discover_experiments_in_directory(data_dir):
    """
    Discover all experiments in a data directory by looking for _sld_profile.dat files.
    
    Args:
        data_dir: Directory to search for experiment files
        
    Returns:
        List of experiment IDs found
    """
    data_dir = Path(data_dir)
    experiment_ids = []
    
    if not data_dir.exists():
        print(f"Data directory does not exist: {data_dir}")
        return experiment_ids
    
    # Look for files matching pattern: {experiment_id}_sld_profile.dat
    for dat_file in data_dir.glob("*_sld_profile.dat"):
        # Extract experiment ID by removing the suffix
        exp_id = dat_file.stem.replace("_sld_profile", "")
        
        # Check if corresponding model file exists
        model_file = data_dir / f"{exp_id}_model.txt"
        if model_file.exists():
            experiment_ids.append(exp_id)
        else:
            print(f"Warning: Found {dat_file.name} but no corresponding {exp_id}_model.txt")
    
    experiment_ids.sort()  # Sort for consistent ordering
    print(f"Found {len(experiment_ids)} complete experiment pairs in {data_dir}")
    
    return experiment_ids


def batch_process_directory(data_dir="./data/MARIA_VIPR_dataset/1", output_dir="./sld_plots/1", 
                           show=False, max_experiments=None):
    """
    Batch process all experiments found in a data directory.
    
    Args:
        data_dir: Directory containing the experiment files
        output_dir: Directory to save output plots
        show: Whether to display the plots (not recommended for large batches)
        max_experiments: Maximum number of experiments to process (None for all)
        
    Returns:
        Dictionary of experiment_id -> figure object (or None if failed)
    """
    print(f"Batch processing experiments in: {data_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Discover all experiments in the directory
    experiment_ids = discover_experiments_in_directory(data_dir)
    
    if not experiment_ids:
        print("No experiments found to process")
        return {}
    
    # Limit number of experiments if specified
    if max_experiments is not None and max_experiments > 0:
        experiment_ids = experiment_ids[:max_experiments]
        print(f"Processing first {len(experiment_ids)} experiments (limited by max_experiments)")
    
    # Use the existing batch comparison function
    return batch_compare_sld_profiles(
        experiment_ids=experiment_ids,
        data_dir=data_dir,
        output_dir=output_dir,
        save=True,
        show=show
    )


def quick_compare(experiment_id, data_dir="./data/MARIA_VIPR_dataset/1"):
    """
    Quick comparison function for interactive use.
    
    Args:
        experiment_id: Experiment identifier (e.g., 's000000')
        data_dir: Directory containing the data files
        
    Returns:
        Figure object if successful, None otherwise
    """
    return compare_sld_profiles(experiment_id, data_dir, show=True, save=False)


def batch_compare_sld_profiles(experiment_ids, data_dir="./data/MARIA_VIPR_dataset/1", output_dir="./sld_plots/1", 
                              save=True, show=False):
    """
    Compare SLD profiles for multiple experiments.
    
    Args:
        experiment_ids: List of experiment identifiers
        data_dir: Directory containing the data files
        output_dir: Directory to save output plots
        save: Whether to save the plots
        show: Whether to display the plots
        
    Returns:
        Dictionary of experiment_id -> figure object (or None if failed)
    """
    results = {}
    
    print(f"Running batch SLD profile comparison for {len(experiment_ids)} experiments")
    print("=" * 80)
    
    for i, exp_id in enumerate(experiment_ids):
        print(f"\nProcessing {i+1}/{len(experiment_ids)}: {exp_id}")
        try:
            fig = compare_sld_profiles(exp_id, data_dir, output_dir, save, show)
            results[exp_id] = fig
        except Exception as e:
            print(f"Failed to process {exp_id}: {e}")
            results[exp_id] = None
    
    # Summary
    successful = sum(1 for fig in results.values() if fig is not None)
    print(f"\n" + "=" * 80)
    print(f"Batch comparison completed: {successful}/{len(experiment_ids)} successful")
    
    return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Compare SLD profiles from .dat file and model.txt parameters')
    parser.add_argument('experiment_ids', nargs='*', help='Experiment identifier(s) (e.g., s000000). If none provided, processes entire directory')
    parser.add_argument('--data-dir', default='./data/MARIA_VIPR_dataset/2', help='Directory containing data files')
    parser.add_argument('--output-dir', default='./sld_plots/2', help='Directory to save plots')
    parser.add_argument('--no-save', action='store_true', help='Do not save the plot')
    parser.add_argument('--no-show', action='store_true', help='Do not display the plot')
    parser.add_argument('--batch', action='store_true', help='Process multiple experiments (implied if multiple IDs given)')
    parser.add_argument('--max-experiments', type=int, help='Maximum number of experiments to process (for directory batch processing)')
    
    args = parser.parse_args()
    
    # If no experiment IDs provided, process entire directory
    if not args.experiment_ids:
        print("No experiment IDs provided. Processing entire directory...")
        results = batch_process_directory(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            show=not args.no_show,
            max_experiments=args.max_experiments
        )
        
        # Check if any failed
        failed = [exp_id for exp_id, fig in results.items() if fig is None]
        if failed:
            print(f"Failed experiments: {failed}")
            return 1
        return 0
    
    # Determine if batch processing
    is_batch = args.batch or len(args.experiment_ids) > 1
    
    if is_batch:
        # Run batch comparison
        results = batch_compare_sld_profiles(
            experiment_ids=args.experiment_ids,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            save=not args.no_save,
            show=not args.no_show
        )
        
        # Check if any failed
        failed = [exp_id for exp_id, fig in results.items() if fig is None]
        if failed:
            print(f"Failed experiments: {failed}")
            return 1
    else:
        # Single experiment
        fig = compare_sld_profiles(
            experiment_id=args.experiment_ids[0],
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            save=not args.no_save,
            show=not args.no_show
        )
        
        if fig is None:
            print("Failed to generate comparison plot")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())