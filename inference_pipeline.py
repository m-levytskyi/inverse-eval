#!/usr/bin/env python3
"""
Inference Pipeline for Testing Multiple ReflecTorch Models

This script tests different trained models on experimental neutron reflectometry data
and compares their performance and parameter predictions. The pipeline is configurable
through JSON configuration files that specify data paths, formats, and model parameters.

Usage:
    python inference_pipeline.py [config_file]

Configuration files:
    - configs/membrane_config.json: For membrane analysis with dQ/Q = 0.1
    - configs/s000000_config.json: For s000000 data analysis
    
The pipeline handles both 3-column (Q, R, dR) and 4-column (Q, R, dR, dQ) data formats
automatically based on the configuration.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import json
from datetime import datetime
import warnings
import sys
warnings.filterwarnings('ignore')

from reflectorch import EasyInferenceModel
from filter_error_bars import filter_and_truncate

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class InferencePipeline:
    """Pipeline for testing multiple models on experimental data."""
    
    def __init__(self, config_file=None, output_dir="inference_results", experiment_id=None, 
                 models_list=None, data_directory="data", priors_type="broad", layer_count=None,
                 error_bar_threshold=0.5, consecutive_threshold=3, remove_singles=False):
        """
        Initialize the inference pipeline.
        
        Args:
            config_file: Path to JSON configuration file (legacy mode)
            output_dir: Directory to save results
            experiment_id: Experiment ID (e.g., 's000000') for batch mode
            models_list: List of model names for batch mode
            data_directory: Base data directory for batch mode
            priors_type: Type of priors to use ('broad' or 'narrow') for batch mode
            layer_count: Number of layers (1 or 2). If None, will auto-detect.
            error_bar_threshold: Relative error threshold for filtering (default: 0.5)
            consecutive_threshold: Number of consecutive high-error points to trigger truncation (default: 3)
            remove_singles: Whether to remove isolated high-error points (default: False)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.layer_count_override = layer_count  # Store for override
        
        # Store preprocessing parameters (always enabled)
        self.error_bar_threshold = error_bar_threshold
        self.consecutive_threshold = consecutive_threshold
        self.remove_singles = remove_singles
        
        # Initialize results storage
        self.results = {}
        
        if experiment_id is not None:
            # Batch mode - initialize from experiment ID
            self.experiment_id = experiment_id
            self.data_directory = Path(data_directory)
            self.models_list = models_list or []
            self.priors_type = priors_type
            
            # Auto-discover experiment files
            self.discover_experiment_files()
            
            # Load experimental data
            self.load_experimental_data_from_files()
            
            # Load true parameters
            self.load_true_parameters_from_files()
            
            # Generate model configuration
            self.generate_model_configurations()
            
        else:
            # Legacy mode - initialize from config file
            if config_file is None:
                raise ValueError("Either config_file or experiment_id must be provided")
            
            self.config_file = Path(config_file)
            
            # Load configuration
            self.load_configuration()
            
            # Load experimental data
            self.load_experimental_data()
            
            # Load true parameters if available
            self.true_params_dict = None
            true_params_file = self.find_true_parameters_file()
            if true_params_file:
                print(f"Found true parameters file: {true_params_file}")
                self.true_params_dict = self.parse_true_parameters_from_model_file(true_params_file)
            elif 'true_parameters_file' in self.data_config:
                self.true_params_dict = self.parse_true_parameters_from_model_file(
                    self.data_config['true_parameters_file']
                )
    
    def discover_experiment_files(self):
        """Auto-discover experiment files based on experiment ID."""
        self.exp_data_file = None
        self.exp_model_file = None
        
        # Common file patterns
        exp_curve_pattern = f"{self.experiment_id}_experimental_curve.dat"
        model_pattern = f"{self.experiment_id}_model.txt"
        
        # Search in various locations
        search_paths = [
            self.data_directory,
            self.data_directory / "MARIA_VIPR_dataset" / "1",
            self.data_directory / "MARIA_VIPR_dataset" / "2",
            self.data_directory / self.experiment_id,
        ]
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
                
            # Look for files directly in the path
            exp_file = search_path / exp_curve_pattern
            model_file = search_path / model_pattern
            
            if exp_file.exists():
                self.exp_data_file = exp_file
            if model_file.exists():
                self.exp_model_file = model_file
                
            # If both found, we're done
            if self.exp_data_file and self.exp_model_file:
                break
                
            # Also search subdirectories
            for subdir in search_path.iterdir():
                if subdir.is_dir():
                    exp_file = subdir / exp_curve_pattern
                    model_file = subdir / model_pattern
                    
                    if exp_file.exists():
                        self.exp_data_file = exp_file
                    if model_file.exists():
                        self.exp_model_file = model_file
                        
                    if self.exp_data_file and self.exp_model_file:
                        break
            
            if self.exp_data_file and self.exp_model_file:
                break
        
        if not self.exp_data_file:
            raise FileNotFoundError(f"Could not find experimental data file for {self.experiment_id}")
        if not self.exp_model_file:
            raise FileNotFoundError(f"Could not find model file for {self.experiment_id}")
            
        print(f"Found experiment files:")
        print(f"  Data: {self.exp_data_file}")
        print(f"  Model: {self.exp_model_file}")
    
    def load_experimental_data_from_files(self):
        """Load experimental data from discovered files."""
        print(f"Loading experimental data from: {self.exp_data_file}")
        
        # Load the experimental data
        data = np.loadtxt(self.exp_data_file)
        
        # Determine format based on number of columns
        if data.shape[1] == 3:
            # 3-column format: Q, R, dR
            self.q_exp = data[:, 0]
            self.curve_exp = data[:, 1]
            self.sigmas_exp = data[:, 2]
            self.q_res_exp = None
        elif data.shape[1] == 4:
            # 4-column format: Q, R, dR, dQ
            self.q_exp = data[:, 0]
            self.curve_exp = data[:, 1]
            self.sigmas_exp = data[:, 2]
            self.q_res_exp = data[:, 3]
        else:
            raise ValueError(f"Unsupported data format: {data.shape[1]} columns")
        
        print(f"Loaded {len(self.q_exp)} data points")
        print(f"Q range: {self.q_exp.min():.4f} - {self.q_exp.max():.4f} Å⁻¹")
        
        # Apply preprocessing steps
        self.preprocess_experimental_data()
    
    def load_true_parameters_from_files(self):
        """Load true parameters from discovered model file."""
        if self.exp_model_file:
            print(f"Loading true parameters from: {self.exp_model_file}")
            self.true_params_dict = self.parse_true_parameters_from_model_file(str(self.exp_model_file))
        else:
            self.true_params_dict = None
    
    def preprocess_experimental_data(self, error_bar_threshold=None, 
                                   consecutive_threshold=None, 
                                   remove_singles=None):
        """
        Apply preprocessing steps to experimental neutron reflectometry data.
        
        This method implements the preprocessing steps described in preprocessing.md:
        1. Remove data points with negative intensity values
        2. Filter/truncate data points with high error bars
        
        Args:
            error_bar_threshold: float, relative error threshold for filtering (default: 0.5)
            consecutive_threshold: int, number of consecutive high-error points to trigger truncation (default: 3)
            remove_singles: bool, whether to remove isolated high-error points (default: False)
        """
        # Use provided parameters or sensible defaults
        threshold = 0.5 if error_bar_threshold is None else error_bar_threshold
        consecutive = 3 if consecutive_threshold is None else consecutive_threshold  
        singles = False if remove_singles is None else remove_singles
        
        print(f"Applying preprocessing to experimental data...")
        original_points = len(self.q_exp)
        
        # Step 1: Remove negative intensity values
        positive_mask = self.curve_exp > 0
        if not np.all(positive_mask):
            negative_count = np.sum(~positive_mask)
            print(f"  Removing {negative_count} data points with negative intensity values")
            
            self.q_exp = self.q_exp[positive_mask]
            self.curve_exp = self.curve_exp[positive_mask]
            self.sigmas_exp = self.sigmas_exp[positive_mask]
            if self.q_res_exp is not None:
                self.q_res_exp = self.q_res_exp[positive_mask]
        
        # Step 1.5: Remove any invalid data (NaN, inf, or zero values that could cause issues)
        # Check for valid finite values in all arrays
        finite_mask = (np.isfinite(self.q_exp) & 
                      np.isfinite(self.curve_exp) & 
                      np.isfinite(self.sigmas_exp) &
                      (self.curve_exp > 0) &  # Ensure positive intensities
                      (self.sigmas_exp > 0))  # Ensure positive error bars
        
        if self.q_res_exp is not None:
            finite_mask = finite_mask & np.isfinite(self.q_res_exp) & (self.q_res_exp > 0)
        
        if not np.all(finite_mask):
            invalid_count = np.sum(~finite_mask)
            print(f"  Removing {invalid_count} data points with invalid values (NaN, inf, or zero)")
            
            self.q_exp = self.q_exp[finite_mask]
            self.curve_exp = self.curve_exp[finite_mask]
            self.sigmas_exp = self.sigmas_exp[finite_mask]
            if self.q_res_exp is not None:
                self.q_res_exp = self.q_res_exp[finite_mask]
        
        # Step 1.5: Clean up invalid data (NaN, inf, or zero values that would cause division issues)
        finite_mask = np.isfinite(self.curve_exp) & np.isfinite(self.sigmas_exp) & (self.curve_exp > 0) & (self.sigmas_exp >= 0)
        if not np.all(finite_mask):
            invalid_count = np.sum(~finite_mask)
            print(f"  Removing {invalid_count} data points with invalid values (NaN, inf, or zero)")
            
            self.q_exp = self.q_exp[finite_mask]
            self.curve_exp = self.curve_exp[finite_mask]
            self.sigmas_exp = self.sigmas_exp[finite_mask]
            if self.q_res_exp is not None:
                self.q_res_exp = self.q_res_exp[finite_mask]
        
        # Step 1.5: Remove invalid data (NaN, inf, very small R values)
        # Check for NaN or inf values
        valid_mask = (np.isfinite(self.q_exp) & 
                     np.isfinite(self.curve_exp) & 
                     np.isfinite(self.sigmas_exp) &
                     (self.curve_exp > 1e-12))  # Remove very small R values that cause division issues
        
        if self.q_res_exp is not None:
            valid_mask = valid_mask & np.isfinite(self.q_res_exp)
        
        if not np.all(valid_mask):
            invalid_count = np.sum(~valid_mask)
            print(f"  Removing {invalid_count} data points with invalid values (NaN, inf, or very small R)")
            
            self.q_exp = self.q_exp[valid_mask]
            self.curve_exp = self.curve_exp[valid_mask]
            self.sigmas_exp = self.sigmas_exp[valid_mask]
            if self.q_res_exp is not None:
                self.q_res_exp = self.q_res_exp[valid_mask]
        
        # Step 2: Filter high error bars
        print(f"  Filtering high error bars (threshold: {threshold}, consecutive: {consecutive})")
        
        # Apply the filter_and_truncate function
        try:
            if self.q_res_exp is not None:
                # For 4-column data, also filter dQ
                q_filtered, curve_filtered, sigmas_filtered = filter_and_truncate(
                    self.q_exp, self.curve_exp, self.sigmas_exp,
                    threshold=threshold,
                    consecutive=consecutive,
                    remove_singles=singles
                )
                
                # Check if filtering was too aggressive
                original_q_range = self.q_exp.max() - self.q_exp.min()
                filtered_q_range = q_filtered.max() - q_filtered.min() if len(q_filtered) > 0 else 0
                points_retained = len(q_filtered) / len(self.q_exp)
                q_range_retained = filtered_q_range / original_q_range if original_q_range > 0 else 0
                
                # If filtering was too aggressive, use less stringent parameters
                if points_retained < 0.3 or q_range_retained < 0.1:
                    print(f"    Warning: Aggressive filtering detected (retained {points_retained:.1%} points, {q_range_retained:.1%} Q range)")
                    print(f"    Retrying with more permissive parameters...")
                    
                    # Try with more permissive parameters
                    q_filtered, curve_filtered, sigmas_filtered = filter_and_truncate(
                        self.q_exp, self.curve_exp, self.sigmas_exp,
                        threshold=max(threshold * 2, 1.0),  # Double threshold or use 1.0
                        consecutive=max(consecutive + 2, 5),  # Increase consecutive requirement
                        remove_singles=False  # Don't remove singles
                    )
                
                # Apply the filtered data
                self.q_exp = q_filtered
                self.curve_exp = curve_filtered
                self.sigmas_exp = sigmas_filtered
                
                # Filter dQ to match the filtered data
                if len(self.q_exp) < len(self.q_res_exp):
                    self.q_res_exp = self.q_res_exp[:len(self.q_exp)]
            else:
                # For 3-column data
                q_filtered, curve_filtered, sigmas_filtered = filter_and_truncate(
                    self.q_exp, self.curve_exp, self.sigmas_exp,
                    threshold=threshold,
                    consecutive=consecutive,
                    remove_singles=singles
                )
                
                # Check if filtering was too aggressive
                original_q_range = self.q_exp.max() - self.q_exp.min()
                filtered_q_range = q_filtered.max() - q_filtered.min() if len(q_filtered) > 0 else 0
                points_retained = len(q_filtered) / len(self.q_exp)
                q_range_retained = filtered_q_range / original_q_range if original_q_range > 0 else 0
                
                # If filtering was too aggressive, use less stringent parameters
                if points_retained < 0.3 or q_range_retained < 0.1:
                    print(f"    Warning: Aggressive filtering detected (retained {points_retained:.1%} points, {q_range_retained:.1%} Q range)")
                    print(f"    Retrying with more permissive parameters...")
                    
                    # Try with more permissive parameters
                    q_filtered, curve_filtered, sigmas_filtered = filter_and_truncate(
                        self.q_exp, self.curve_exp, self.sigmas_exp,
                        threshold=max(threshold * 2, 1.0),  # Double threshold or use 1.0
                        consecutive=max(consecutive + 2, 5),  # Increase consecutive requirement
                        remove_singles=False  # Don't remove singles
                    )
                
                # Apply the filtered data
                self.q_exp = q_filtered
                self.curve_exp = curve_filtered
                self.sigmas_exp = sigmas_filtered
        except Exception as e:
            print(f"  Warning: Error bar filtering failed ({e}), continuing with current data")
            # Continue with the data we have
        
        final_points = len(self.q_exp)
        removed_points = original_points - final_points
        final_q_range = self.q_exp.max() - self.q_exp.min()
        
        # Check if we have enough data points left
        if final_points < 10:  # Increased minimum from 3 to 10
            raise ValueError(f"Insufficient data points after preprocessing: {final_points} points remaining (minimum 10 required)")
        
        # Check if we have a reasonable Q range for model inference
        if final_q_range < 0.05:  # Minimum Q range of 0.05 Å⁻¹
            print(f"  Warning: Very narrow Q range after preprocessing: {final_q_range:.4f} Å⁻¹")
            # This might cause model inference issues, but we'll continue
        
        print(f"  Preprocessing complete:")
        print(f"    Original points: {original_points}")
        print(f"    Final points: {final_points}")
        print(f"    Removed points: {removed_points} ({100*removed_points/original_points:.1f}%)")
        print(f"    Final Q range: {self.q_exp.min():.4f} - {self.q_exp.max():.4f} Å⁻¹ (span: {final_q_range:.4f})")

    def generate_model_configurations(self):
        """Generate model configurations for batch processing."""
        if not self.models_list:
            raise ValueError("No models specified for batch processing")
        
        # Load MARIA bounds for priors
        maria_bounds = self.load_maria_bounds()
        
        # Determine layer count from true parameters
        layer_count = self.determine_layer_count()
        
        # Get appropriate priors
        prior_bounds = self.get_priors_for_layer_count(layer_count, maria_bounds)
        parameter_names = self.get_parameter_names_for_layer_count(layer_count)
        
        # Generate model configurations
        self.model_configs = {}
        for model_name in self.models_list:
            self.model_configs[model_name] = {
                "config_name": model_name,
                "description": f"{model_name} model for {layer_count}-layer system",
                "weights_format": "safetensors",
                "prior_bounds": prior_bounds,
                "parameter_names": parameter_names
            }
        
        print(f"Generated configurations for {len(self.model_configs)} models")
        print(f"Layer count: {layer_count}, Priors type: {self.priors_type}")
    
    def determine_layer_count(self):
        """Determine layer count from override or true parameters file."""
        # If layer count is explicitly provided, use it
        if hasattr(self, 'layer_count_override') and self.layer_count_override is not None:
            return self.layer_count_override
            
        if not self.true_params_dict:
            # Default assumption - could be improved
            return 2
            
        # Count actual material layers from the parsed model file
        # The true_params_dict might have multiple entries, but we want the natural layer count
        
        # Look for the most natural layer count based on available data
        # Priority: 2_layer (most common), then 1_layer
        for layer_key in ['2_layer', '1_layer']:
            if layer_key in self.true_params_dict:
                return int(layer_key.split('_')[0])
        
        return 2  # Default
    
    def load_maria_bounds(self):
        """Load MARIA dataset prior bounds."""
        bounds_file = Path("maria_dataset_prior_bounds.json")
        if not bounds_file.exists():
            print("Warning: MARIA bounds file not found. Using default bounds.")
            return None
            
        with open(bounds_file, 'r') as f:
            return json.load(f)
    
    def get_priors_for_layer_count(self, layer_count, maria_bounds):
        """Get appropriate priors for the layer count."""
        if not maria_bounds:
            return self.get_default_priors(layer_count)
        
        layer_key = f"{layer_count}_layers"
        if layer_key not in maria_bounds:
            return self.get_default_priors(layer_count)
        
        layer_data = maria_bounds[layer_key]
        parameters = layer_data['parameters']
        
        bounds = []
        
        if layer_count == 1:
            # 1-layer: [thickness, amb_rough, sub_rough, layer_sld, sub_sld]
            bounds = self.extract_bounds_1_layer(parameters)
        elif layer_count == 2:
            # 2-layer bounds
            bounds = self.extract_bounds_2_layer(parameters)
        
        # Apply priors type (broad vs narrow)
        if self.priors_type == "narrow":
            bounds = self.apply_narrow_priors(bounds, parameters, layer_count)
        
        # Validate all bounds
        validated_bounds = []
        for i, bound in enumerate(bounds):
            min_val, max_val = bound
            if min_val >= max_val:
                max_val = min_val + 0.1
            validated_bounds.append([min_val, max_val])
        
        return validated_bounds
    
    def get_default_priors(self, layer_count):
        """Get default prior bounds when MARIA bounds are not available."""
        if layer_count == 1:
            return [
                [10.0, 500.0],      # thickness
                [0.1, 10.0],        # amb_rough (min 0.1 to avoid negative)
                [0.1, 10.0],        # sub_rough (min 0.1 to avoid negative)
                [-5e-06, 5e-06],    # layer_sld
                [-5e-06, 5e-06]     # sub_sld
            ]
        elif layer_count == 2:
            return [
                [10.0, 500.0],      # L1 thickness
                [10.0, 500.0],      # L2 thickness
                [0.1, 10.0],        # amb_rough (min 0.1 to avoid negative)
                [0.1, 10.0],        # L1L2_rough (min 0.1 to avoid negative)
                [0.1, 10.0],        # L2sub_rough (min 0.1 to avoid negative)
                [-5e-06, 5e-06],    # L1_sld
                [-5e-06, 5e-06],    # L2_sld
                [-5e-06, 5e-06]     # sub_sld
            ]
    
    def extract_bounds_1_layer(self, parameters):
        """Extract bounds for 1-layer system from MARIA structure."""
        bounds = []
        
        # For 1-layer: [thickness, amb_rough, sub_rough, layer_sld, sub_sld]
        
        # Layer thickness - use overall thickness if available
        if 'overall' in parameters and 'thickness' in parameters['overall']:
            thickness_data = parameters['overall']['thickness']
            bounds.append([thickness_data['min'], thickness_data['max']])
        else:
            bounds.append([10.0, 500.0])
        
        # Roughness parameters
        if 'interface' in parameters and 'fronting_roughness' in parameters['interface']:
            fronting_data = parameters['interface']['fronting_roughness']
            min_val = max(0.1, fronting_data['min'])  # Ensure minimum of 0.1
            max_val = max(min_val + 0.1, fronting_data['max'])  # Ensure max > min
            bounds.append([min_val, max_val])
        else:
            bounds.append([1.0, 10.0])
        
        if 'interface' in parameters and 'backing_roughness' in parameters['interface']:
            backing_data = parameters['interface']['backing_roughness']
            min_val = max(0.1, backing_data['min'])  # Ensure minimum of 0.1
            max_val = max(min_val + 0.1, backing_data['max'])  # Ensure max > min
            bounds.append([min_val, max_val])
        else:
            bounds.append([0.1, 10.0])  # backing roughness minimum 0.1 to avoid negative values
        
        # SLD parameters
        if 'overall' in parameters and 'sld' in parameters['overall']:
            sld_data = parameters['overall']['sld']
            min_val = sld_data['min'] * 1e6  # Convert to 10^-6 units
            max_val = sld_data['max'] * 1e6  
            # Ensure valid range
            if min_val >= max_val:
                max_val = min_val + 0.1
            bounds.append([min_val, max_val])
        else:
            bounds.append([-5.0, 15.0])
            
        if 'interface' in parameters and 'backing_sld' in parameters['interface']:
            backing_sld = parameters['interface']['backing_sld']
            min_val = backing_sld['min'] * 1e6
            max_val = backing_sld['max'] * 1e6
            # Ensure valid range
            if min_val >= max_val:
                max_val = min_val + 0.1
            bounds.append([min_val, max_val])
        else:
            bounds.append([-5.0, 15.0])
        
        return bounds
    
    def extract_bounds_2_layer(self, parameters):
        """Extract bounds for 2-layer system from MARIA structure."""
        bounds = []
        
        # Extract bounds from the correct MARIA structure
        # For 2-layer: [L1_thick, L2_thick, amb_rough, L1L2_rough, L2sub_rough, L1_sld, L2_sld, sub_sld]
        
        # Layer thicknesses
        if 'by_position' in parameters and 'layer_1' in parameters['by_position']:
            layer1_data = parameters['by_position']['layer_1']
            bounds.append([layer1_data['thickness']['min'], layer1_data['thickness']['max']])
        else:
            bounds.append([10.0, 500.0])
            
        if 'by_position' in parameters and 'layer_2' in parameters['by_position']:
            layer2_data = parameters['by_position']['layer_2']
            bounds.append([layer2_data['thickness']['min'], layer2_data['thickness']['max']])
        else:
            bounds.append([10.0, 500.0])
        
        # Roughness parameters (fronting, inter-layer, backing)
        if 'interface' in parameters and 'fronting_roughness' in parameters['interface']:
            fronting_data = parameters['interface']['fronting_roughness']
            bounds.append([fronting_data['min'], fronting_data['max']])
        else:
            bounds.append([1.0, 10.0])
        
        # Inter-layer roughness - use average of layer roughnesses as approximation
        if 'by_position' in parameters:
            layer1_rough = parameters['by_position']['layer_1']['roughness']
            layer2_rough = parameters['by_position']['layer_2']['roughness']
            # Use the combined range of both layers for inter-layer roughness
            min_rough = min(layer1_rough['min'], layer2_rough['min'])
            max_rough = max(layer1_rough['max'], layer2_rough['max'])
            bounds.append([min_rough, max_rough])
            bounds.append([min_rough, max_rough])  # L2/substrate roughness
        else:
            bounds.append([1.0, 60.0])
            bounds.append([1.0, 60.0])
        
        # SLD parameters
        if 'by_position' in parameters:
            layer1_sld = parameters['by_position']['layer_1']['sld']
            layer2_sld = parameters['by_position']['layer_2']['sld']
            bounds.append([layer1_sld['min'] * 1e6, layer1_sld['max'] * 1e6])  # Convert to 10^-6 units
            bounds.append([layer2_sld['min'] * 1e6, layer2_sld['max'] * 1e6])
        else:
            bounds.append([-5.0, 15.0])
            bounds.append([-5.0, 15.0])
            
        if 'interface' in parameters and 'backing_sld' in parameters['interface']:
            backing_sld = parameters['interface']['backing_sld']
            bounds.append([backing_sld['min'] * 1e6, backing_sld['max'] * 1e6])
        else:
            bounds.append([-5.0, 15.0])
        
        return bounds
    
    def apply_narrow_priors(self, bounds, parameters, layer_count):
        """
        Apply narrow priors (75-125% of true values) without MARIA bounds constraints.
        
        This fixes the issue where narrow priors could exceed 25% MAPE due to 
        inappropriate constraining to MARIA dataset bounds.
        """
        # Use true parameters to create narrow priors around them
        if not self.true_params_dict:
            print("Warning: No true parameters available, using default narrow bounds")
            # Fallback to the old method if no true parameters
            narrow_bounds = []
            for i, bound in enumerate(bounds):
                range_size = bound[1] - bound[0]
                narrow_range = range_size * 0.5  # Use 50% of the range
                center = (bound[0] + bound[1]) / 2
                new_min = center - narrow_range / 2
                new_max = center + narrow_range / 2
                
                # Ensure roughness parameters never go below 0
                if i in [1, 2, 3, 4] and new_min < 0.0:
                    new_min = 0.0
                    
                narrow_bounds.append([new_min, new_max])
            return narrow_bounds
        
        # Get true parameters for this layer configuration
        layer_key = f"{layer_count}_layer"
        if layer_key not in self.true_params_dict:
            print(f"Warning: No true parameters for {layer_key}, using default narrow bounds")
            return bounds
        
        true_params = self.true_params_dict[layer_key]['params']
        param_names = self.true_params_dict[layer_key]['param_names']
        
        # Create narrow bounds: 75-125% of true values (FIXED VERSION)
        narrow_bounds = []
        for i, (param_name, true_val) in enumerate(zip(param_names, true_params)):
            if true_val is None:
                # Fallback for problematic true values
                narrow_bounds.append(bounds[i] if i < len(bounds) else [-5.0, 15.0])
                continue
            
            if abs(true_val) < 1e-10:  # Essentially zero
                # For near-zero values, use a small symmetric range
                narrow_bounds.append([-0.01, 0.01])
                continue
                
            # Calculate proper 75-125% bounds based on true value sign
            if true_val >= 0:
                # Positive values: straightforward 75-125%
                new_min = true_val * 0.75
                new_max = true_val * 1.25
            else:
                # Negative values: 75% is less negative, 125% is more negative
                new_min = true_val * 1.25  # More negative (larger absolute value)
                new_max = true_val * 0.75  # Less negative (smaller absolute value)
            
            # Apply ONLY essential physical constraints (not MARIA bounds!)
            if 'roughness' in param_name.lower() and new_min < 0:
                new_min = 0.0  # Roughness can't be negative
                
            if 'thickness' in param_name.lower() and new_min < 0:
                new_min = 0.01  # Thickness can't be negative or zero
            
            # Note: We do NOT constrain to original MARIA bounds anymore!
            # This was the root cause of the >25% MAPE issue.
            
            narrow_bounds.append([new_min, new_max])
            
        return narrow_bounds
    
    def get_parameter_names_for_layer_count(self, layer_count):
        """Get parameter names for the given layer count."""
        if layer_count == 1:
            return [
                "L1 thickness (Å)",
                "ambient/L1 roughness (Å)",
                "L1/substrate roughness (Å)",
                "L1 SLD (×10⁻⁶ Å⁻²)",
                "substrate SLD (×10⁻⁶ Å⁻²)"
            ]
        else:  # layer_count == 2
            return [
                "L1 thickness (Å)",
                "L2 thickness (Å)",
                "ambient/L1 roughness (Å)",
                "L1/L2 roughness (Å)",
                "L2/substrate roughness (Å)",
                "L1 SLD (×10⁻⁶ Å⁻²)",
                "L2 SLD (×10⁻⁶ Å⁻²)",
                "substrate SLD (×10⁻⁶ Å⁻²)"
            ]

    def load_configuration(self):
        """Load configuration from JSON file."""
        print(f"Loading configuration from: {self.config_file}")
        
        if not self.config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)
        
        # Extract data and model configurations
        self.data_config = self.config['data_config']
        self.model_configs = self.config['model_configurations']
        
        print(f"Loaded configuration for: {self.data_config['description']}")
        print(f"Data format: {self.data_config['data_format']}")
        print(f"Available models: {len(self.model_configs)}")
        
    def load_experimental_data(self):
        """Load experimental data from file based on configuration."""
        data_path = Path(self.data_config['data_path'])
        print(f"Loading experimental data from: {data_path}")
        
        # Load data
        data = np.loadtxt(data_path, skiprows=1)
        
        # Handle data format based on configuration
        if self.data_config['data_format'] == '3_column':
            # 3-column format: Q, R, dR
            print("Processing 3-column format (Q, R, dR)")
            self.q_exp = data[:, 0]
            self.curve_exp = data[:, 1]
            self.sigmas_exp = data[:, 2]
            
            # Calculate Q-resolution using configured dQ/Q ratio
            dq_over_q = self.data_config['dq_over_q']
            if dq_over_q is None:
                raise ValueError("dq_over_q must be specified for 3-column data format")
            
            self.q_res_exp = self.q_exp * dq_over_q
            print(f"Calculated Q-resolution using dQ/Q = {dq_over_q}")
            
        elif self.data_config['data_format'] == '4_column':
            # 4-column format: Q, R, dR, dQ
            print("Processing 4-column format (Q, R, dR, dQ)")
            
            # Apply max_points limit if specified
            max_points = self.data_config.get('max_points')
            if max_points is not None:
                max_points = min(max_points, len(data))
                data = data[:max_points]
                print(f"Trimmed data to {max_points} points")
            
            self.q_exp = data[:, 0]
            self.curve_exp = data[:, 1]
            self.sigmas_exp = data[:, 2]
            self.q_res_exp = data[:, 3]
        else:
            raise ValueError(f"Unsupported data format: {self.data_config['data_format']}")
        
        print(f"Loaded {len(self.q_exp)} data points")
        print(f"Q range: {self.q_exp.min():.4f} - {self.q_exp.max():.4f} Å⁻¹")
        print(f"dQ range: {self.q_res_exp.min():.6f} - {self.q_res_exp.max():.6f} Å⁻¹")
        
        # Apply preprocessing steps
        self.preprocess_experimental_data()
        
    def run_inference(self, model_config, model_name):
        """
        Run inference with a specific model configuration.
        
        Args:
            model_config: Dictionary containing model configuration
            model_name: Name identifier for the model
            
        Returns:
            Dictionary containing inference results
        """
        print(f"\n{'='*60}")
        print(f"Testing model: {model_name}")
        print(f"Description: {model_config['description']}")
        print(f"{'='*60}")
        
        try:
            # Initialize model
            inference_model = EasyInferenceModel(
                config_name=model_config['config_name'],
                device='cpu',
                weights_format=model_config.get('weights_format', 'pth')
            )
            
            # Interpolate data to model grid
            q_model, exp_curve_interp = inference_model.interpolate_data_to_model_q(
                self.q_exp, self.curve_exp
            )
            
            # Debug information for tensor concatenation issues
            print(f"Original data: {len(self.q_exp)} points, Q range: {self.q_exp.min():.4f} - {self.q_exp.max():.4f}")
            if self.q_res_exp is not None:
                print(f"Resolution data available")
            print(f"Interpolated data: {len(q_model)} points, Q range: {q_model.min():.4f} - {q_model.max():.4f}")
            
            # Check for potential issues that could cause tensor concatenation errors
            if len(q_model) == 0:
                raise ValueError("Model interpolation resulted in empty Q grid")
            if len(exp_curve_interp) == 0:
                raise ValueError("Model interpolation resulted in empty reflectivity curve")
            if len(q_model) != len(exp_curve_interp):
                raise ValueError(f"Q grid ({len(q_model)}) and curve ({len(exp_curve_interp)}) length mismatch")
            if np.any(~np.isfinite(exp_curve_interp)):
                print(f"  Warning: Non-finite values in interpolated curve")
            if np.any(exp_curve_interp <= 0):
                print(f"  Warning: Non-positive values in interpolated curve")
            
            # Ensure arrays have proper dimensions (1D) and correct data types
            q_model = np.asarray(q_model, dtype=np.float32).flatten()
            exp_curve_interp = np.asarray(exp_curve_interp, dtype=np.float32).flatten()
            
            if len(q_model) != len(exp_curve_interp):
                raise ValueError(f"After processing: Q grid ({len(q_model)}) and curve ({len(exp_curve_interp)}) length mismatch")
            
            # Interpolate resolution data
            if self.q_res_exp is not None:
                q_res_interp = np.interp(q_model, self.q_exp, self.q_res_exp)
            else:
                # For 3-column data, assume a constant dQ/Q = 0.05 (5%)
                q_res_interp = q_model * 0.05
                print(f"  No resolution data provided, using dQ/Q = 5%")
            
            # Ensure resolution array has proper dimensions and data type
            q_res_interp = np.asarray(q_res_interp, dtype=np.float32).flatten()
            
            print(f"Model Q grid: {len(q_model)} points, range: {q_model.min():.4f} - {q_model.max():.4f}")
            
            # Verify all arrays have the same length
            if not (len(q_model) == len(exp_curve_interp) == len(q_res_interp)):
                raise ValueError(f"Array length mismatch: Q={len(q_model)}, R={len(exp_curve_interp)}, dQ={len(q_res_interp)}")
            
            # Convert prior bounds from list format to tuple format
            prior_bounds = [tuple(bound) for bound in model_config['prior_bounds']]
            
            # Run prediction
            prediction_dict = inference_model.predict(
                reflectivity_curve=exp_curve_interp,
                prior_bounds=prior_bounds,
                q_values=q_model,
                q_resolution=q_res_interp,
                clip_prediction=True,
                polish_prediction=True,
                calc_pred_curve=True,
                calc_pred_sld_profile=True,
                calc_polished_sld_profile=True,
            )
            
            # Extract results
            result = {
                'model_name': model_name,
                'config_name': model_config['config_name'],
                'description': model_config['description'],
                'q_model': q_model,
                'predicted_params': prediction_dict['predicted_params_array'],
                'polished_params': prediction_dict['polished_params_array'],
                'param_names': prediction_dict['param_names'],
                'predicted_curve': prediction_dict['predicted_curve'],
                'polished_curve': prediction_dict['polished_curve'],
                'sld_profile_x': prediction_dict['predicted_sld_xaxis'],
                'sld_profile_predicted': prediction_dict['predicted_sld_profile'],
                'sld_profile_polished': prediction_dict['sld_profile_polished'],
                'success': True,
                'error': None
            }
            
            # Print parameter results using parameter names from config
            print(f"\nParameter Results for {model_name}:")
            print("-" * 50)
            param_names = model_config.get('parameter_names', prediction_dict["param_names"])
            for param_name, pred_val, polish_val in zip(
                param_names, 
                prediction_dict['predicted_params_array'],
                prediction_dict["polished_params_array"]
            ):
                print(f'{param_name.ljust(25)} -> Predicted: {pred_val:8.2f}   Polished: {polish_val:8.2f}')
            
            # Calculate fit quality metrics
            result['fit_metrics'] = self.calculate_fit_metrics(
                self.curve_exp, prediction_dict['polished_curve'], 
                self.sigmas_exp, q_model
            )
            
            # Calculate parameter metrics - get true parameters for this model
            true_params, true_param_names = self.get_true_params_for_model(model_config, self.true_params_dict)
            if true_params is not None:
                try:
                    result['parameter_metrics'] = self.calculate_parameter_metrics(
                        prediction_dict['polished_params_array'], 
                        true_params, 
                        param_names
                    )
                    
                    # Print parameter comparison
                    print(f"\nParameter Comparison (Predicted vs True):")
                    print("-" * 60)
                    print(f"{'Parameter':<25} {'Predicted':<12} {'True':<12} {'MSE':<12}")
                    print("-" * 60)
                    
                    param_metrics = result['parameter_metrics']['by_parameter']
                    for param_name, metrics in param_metrics.items():
                        pred_val = metrics['predicted']
                        true_val = metrics['true']
                        sq_error = metrics['squared_error']
                        
                        print(f"{param_name:<25} {pred_val:<12.2f} {true_val:<12.2f} "
                              f"{sq_error:<12.4f}")
                    
                    # Print aggregate metrics
                    overall_metrics = result['parameter_metrics']['overall']
                    by_type = result['parameter_metrics']['by_type']
                    print(f"\nParameter Metrics Summary:")
                    print(f"Overall MAPE: {overall_metrics['mape']:.2f}%, Overall MSE: {overall_metrics['mse']:.6f}")
                    if by_type['thickness_mape'] > 0:
                        print(f"Thickness - MAPE: {by_type['thickness_mape']:.2f}%, MSE: {by_type['thickness_mse']:.6f}")
                    if by_type['roughness_mape'] > 0:
                        print(f"Roughness - MAPE: {by_type['roughness_mape']:.2f}%, MSE: {by_type['roughness_mse']:.6f}")
                    if by_type['sld_mape'] > 0:
                        print(f"SLD - MAPE: {by_type['sld_mape']:.2f}%, MSE: {by_type['sld_mse']:.6f}")
                        
                except Exception as e:
                    print(f"Warning: Could not calculate parameter metrics: {e}")
                    result['parameter_metrics'] = None
            else:
                result['parameter_metrics'] = None
            
            print(f"\nFit Quality Metrics:")
            print(f"R-squared: {result['fit_metrics']['r_squared']:.4f}")
            print(f"MSE: {result['fit_metrics']['mse']:.6f}")
            print(f"L1 Loss: {result['fit_metrics']['l1_loss']:.6f}")
            
            return result
            
        except Exception as e:
            print(f"ERROR with model {model_name}: {str(e)}")
            return {
                'model_name': model_name,
                'success': False,
                'error': str(e)
            }
    
    def calculate_fit_metrics(self, y_exp, y_pred, sigma_exp, q_model):
        """Calculate fit quality metrics including MSE, R-squared, and L1 loss."""
        # Interpolate predicted curve to experimental Q points for comparison
        y_pred_interp = np.interp(self.q_exp, q_model, y_pred)
        
        # R-squared
        ss_res = np.sum((y_exp - y_pred_interp) ** 2)
        ss_tot = np.sum((y_exp - np.mean(y_exp)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # MSE (Mean Squared Error)
        mse = np.mean((y_exp - y_pred_interp) ** 2)
        
        # L1 Loss (Mean Absolute Error)
        l1_loss = np.mean(np.abs(y_exp - y_pred_interp))
        
        return {
            'r_squared': r_squared,
            'mse': mse,
            'l1_loss': l1_loss
        }
    
    def calculate_parameter_metrics(self, predicted_params, true_params, param_names):
        """
        Calculate parameter metrics comparing predicted vs true parameters.
        Individual parameters show MSE, but overall comparison uses MAPE for better interpretability.
        
        Args:
            predicted_params: Array of predicted parameter values
            true_params: Array of true parameter values  
            param_names: List of parameter names for detailed breakdown
            
        Returns:
            Dictionary containing parameter metrics
        """
        if len(predicted_params) != len(true_params):
            raise ValueError(f"Parameter arrays must have same length: {len(predicted_params)} vs {len(true_params)}")
        
        # Calculate overall MAPE for comparison
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_errors = np.abs((predicted_params - true_params) / (true_params + 1e-8))
            relative_errors = np.where(np.isfinite(relative_errors), relative_errors, 0)
        param_mape = np.mean(relative_errors) * 100
        
        # Per-parameter breakdown with MSE for individual parameters
        param_breakdown = {}
        for i, name in enumerate(param_names):
            if i < len(predicted_params):
                pred_val = predicted_params[i]
                true_val = true_params[i]
                squared_error = (pred_val - true_val) ** 2
                abs_error = abs(pred_val - true_val)
                rel_error = abs_error / abs(true_val) * 100 if abs(true_val) > 1e-10 else 0
                
                param_breakdown[name] = {
                    'predicted': float(pred_val),
                    'true': float(true_val),
                    'squared_error': float(squared_error),
                    'absolute_error': float(abs_error),
                    'relative_error_percent': float(rel_error)
                }
        
        # Group parameters by type for aggregate statistics (both MAPE and MSE)
        thickness_mapes = []
        thickness_mses = []
        roughness_mapes = []
        roughness_mses = []
        sld_mapes = []
        sld_mses = []
        
        # Calculate overall MSE
        overall_mse = np.mean([metrics['squared_error'] for metrics in param_breakdown.values()])
        
        for name, metrics in param_breakdown.items():
            rel_err = metrics['relative_error_percent']
            sq_err = metrics['squared_error']
            
            if 'thickness' in name.lower() or 'd_' in name.lower():
                thickness_mapes.append(rel_err)
                thickness_mses.append(sq_err)
            elif 'roughness' in name.lower() or 'sigma' in name.lower():
                roughness_mapes.append(rel_err)
                roughness_mses.append(sq_err)
            elif 'sld' in name.lower() or 'rho' in name.lower():
                sld_mapes.append(rel_err)
                sld_mses.append(sq_err)
        
        return {
            'overall': {
                'mape': float(param_mape),
                'mse': float(overall_mse)
            },
            'by_parameter': param_breakdown,
            'by_type': {
                'thickness_mape': float(np.mean(thickness_mapes)) if thickness_mapes else 0.0,
                'thickness_mse': float(np.mean(thickness_mses)) if thickness_mses else 0.0,
                'roughness_mape': float(np.mean(roughness_mapes)) if roughness_mapes else 0.0,
                'roughness_mse': float(np.mean(roughness_mses)) if roughness_mses else 0.0,
                'sld_mape': float(np.mean(sld_mapes)) if sld_mapes else 0.0,
                'sld_mse': float(np.mean(sld_mses)) if sld_mses else 0.0
            }
        }
    
    def run_all_models(self, show_plots=True):
        """
        Run inference on all models defined in the configuration.
        
        Args:
            show_plots: Whether to create and show comparison plots
        """
        print(f"Starting inference pipeline with {len(self.model_configs)} models...")
        print(f"Results will be saved to: {self.output_dir}")
        
        for model_name, model_config in self.model_configs.items():
            result = self.run_inference(model_config, model_name)
            self.results[model_name] = result
        
        # Save results
        self.save_results()
        
        # Generate comparison plots only if requested
        if show_plots:
            self.create_comparison_plots()
        
        # Print summary
        self.print_summary()
    
    def save_results(self):
        """Save results to JSON file."""
        # Prepare serializable results
        serializable_results = {}
        
        for model_name, result in self.results.items():
            if result['success']:
                serializable_result = {
                    'model_name': result['model_name'],
                    'config_name': result['config_name'],
                    'description': result['description'],
                    'predicted_params': result['predicted_params'].tolist(),
                    'polished_params': result['polished_params'].tolist(),
                    'param_names': result['param_names'],
                    'fit_metrics': result['fit_metrics'],
                    'success': result['success']
                }
                
                # Add parameter metrics if available
                if 'parameter_metrics' in result and result['parameter_metrics'] is not None:
                    serializable_result['parameter_metrics'] = result['parameter_metrics']
                
                serializable_results[model_name] = serializable_result
            else:
                serializable_results[model_name] = {
                    'model_name': result['model_name'],
                    'success': result['success'],
                    'error': result['error']
                }
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"inference_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
    
    def create_comparison_plots(self):
        """Create comparison plots for all successful models."""
        successful_models = {k: v for k, v in self.results.items() if v['success']}
        
        if not successful_models:
            print("No successful models to plot")
            return
        
        # Check if we have parameter metrics for plotting
        models_with_param_metrics = {
            k: v for k, v in successful_models.items() 
            if 'parameter_metrics' in v and v['parameter_metrics'] is not None
        }
        
        # Create plots with or without parameter loss subplot
        if models_with_param_metrics:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot experimental data
        ax1.errorbar(self.q_exp, self.curve_exp, yerr=self.sigmas_exp, 
                    xerr=self.q_res_exp, fmt='o', markersize=2, alpha=0.7,
                    color='black', label='Experimental', zorder=1)
        
        ax1.set_yscale('log')
        ax1.set_xlabel('q [Å⁻¹]', fontsize=12)
        ax1.set_ylabel('R(q)', fontsize=12)
        
        # Extract experiment name from data path or description
        experiment_name = self.get_experiment_name()
        ax1.set_title(f'Reflectivity Curves Comparison - {experiment_name}', fontsize=14)
        
        # Plot model predictions with metrics in labels
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        for i, (model_name, result) in enumerate(successful_models.items()):
            color = colors[i % len(colors)]
            metrics = result['fit_metrics']
            label = f"{model_name} (R²={metrics['r_squared']:.3f}, MSE={metrics['mse']:.4f})"
            ax1.plot(result['q_model'], result['polished_curve'], 
                    color=color, linewidth=2, label=label, 
                    alpha=0.8, zorder=2+i)
        
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot SLD profiles
        for i, (model_name, result) in enumerate(successful_models.items()):
            color = colors[i % len(colors)]
            ax2.plot(result['sld_profile_x'], result['sld_profile_polished'], 
                    color=color, linewidth=2, label=f"{model_name}", alpha=0.8)
        
        ax2.set_xlabel('z [Å]', fontsize=12)
        ax2.set_ylabel('SLD [10⁻⁶ Å⁻²]', fontsize=12)
        ax2.set_title(f'SLD Profiles Comparison - {experiment_name}', fontsize=14)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Plot parameter loss comparison if available
        if models_with_param_metrics:
            # Create parameter MAPE bar chart
            model_names = list(models_with_param_metrics.keys())
            param_mapes = []
            param_mses = []
            thickness_mapes = []
            thickness_mses = []
            roughness_mapes = []
            roughness_mses = []
            sld_mapes = []
            sld_mses = []
            
            for model_name in model_names:
                result = models_with_param_metrics[model_name]
                param_metrics = result['parameter_metrics']
                param_mapes.append(param_metrics['overall']['mape'])
                param_mses.append(param_metrics['overall']['mse'])
                thickness_mapes.append(param_metrics['by_type']['thickness_mape'])
                thickness_mses.append(param_metrics['by_type']['thickness_mse'])
                roughness_mapes.append(param_metrics['by_type']['roughness_mape'])
                roughness_mses.append(param_metrics['by_type']['roughness_mse'])
                sld_mapes.append(param_metrics['by_type']['sld_mape'])
                sld_mses.append(param_metrics['by_type']['sld_mse'])
            
            x = np.arange(len(model_names))
            width = 0.2
            
            ax3.bar(x - 1.5*width, param_mapes, width, label='Overall MAPE', alpha=0.8, color='gray')
            ax3.bar(x - 0.5*width, thickness_mapes, width, label='Thickness MAPE', alpha=0.8, color='blue')
            ax3.bar(x + 0.5*width, roughness_mapes, width, label='Roughness MAPE', alpha=0.8, color='green')
            ax3.bar(x + 1.5*width, sld_mapes, width, label='SLD MAPE', alpha=0.8, color='red')
            
            ax3.set_xlabel('Model', fontsize=12)
            ax3.set_ylabel('MAPE (%)', fontsize=12)
            ax3.set_title(f'Parameter Prediction Errors (MAPE) - {experiment_name}', fontsize=14)
            ax3.set_xticks(x)
            ax3.set_xticklabels([name[:15] + '...' if len(name) > 15 else name for name in model_names], 
                              rotation=45, ha='right')
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add values on top of bars for the overall MAPE
            for i, v in enumerate(param_mapes):
                ax3.text(i - 1.5*width, v + max(param_mapes) * 0.01, f'{v:.1f}%', 
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            # Create MSE text box
            mse_text = "Parameter MSE Values:\n"
            mse_text += "-" * 25 + "\n"
            for i, model_name in enumerate(model_names):
                short_name = model_name[:12] + "..." if len(model_name) > 12 else model_name
                mse_text += f"{short_name}:\n"
                mse_text += f"  Overall: {param_mses[i]:.4f}\n"
                if thickness_mses[i] > 0:
                    mse_text += f"  Thick: {thickness_mses[i]:.4f}\n"
                if roughness_mses[i] > 0:
                    mse_text += f"  Rough: {roughness_mses[i]:.4f}\n"
                if sld_mses[i] > 0:
                    mse_text += f"  SLD: {sld_mses[i]:.4f}\n"
                mse_text += "\n"
            
            # Add MSE text box to the parameter plot
            ax3.text(0.02, 0.98, mse_text, transform=ax3.transAxes, 
                    fontsize=7, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Add metrics summary text box on the plot
        if successful_models:
            # Sort models by MSE for the summary
            sorted_models = sorted(
                successful_models.items(), 
                key=lambda x: x[1]['fit_metrics']['mse']
            )
            
            # Create metrics summary text
            summary_text = "Best Models (by MSE):\n"
            for i, (model_name, result) in enumerate(sorted_models[:3]):  # Top 3
                metrics = result['fit_metrics']
                summary_text += f"{i+1}. {model_name[:15]}...\n"
                summary_text += f"   R²={metrics['r_squared']:.3f}, MSE={metrics['mse']:.4f}\n"
            
            # Add text box to the reflectivity plot
            ax1.text(0.02, 0.02, summary_text, transform=ax1.transAxes, 
                    fontsize=8, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.output_dir / f"model_comparison_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Comparison plot saved to: {plot_file}")
    
    def print_summary(self):
        """Print summary of all model results."""
        print(f"\n{'='*80}")
        print("INFERENCE PIPELINE SUMMARY")
        print(f"{'='*80}")
        
        successful_models = {k: v for k, v in self.results.items() if v['success']}
        failed_models = {k: v for k, v in self.results.items() if not v['success']}
        
        print(f"Total models tested: {len(self.results)}")
        print(f"Successful: {len(successful_models)}")
        print(f"Failed: {len(failed_models)}")
        
        if failed_models:
            print(f"\nFailed models:")
            for model_name, result in failed_models.items():
                print(f"  - {model_name}: {result['error']}")
        
        if successful_models:
            print(f"\nFit Quality Comparison:")
            print("-" * 80)
            print(f"{'Model':<20} {'R²':<10} {'MSE':<12} {'L1 Loss':<12}")
            print("-" * 80)
            
            # Sort by MSE (lower is better)
            sorted_models = sorted(
                successful_models.items(), 
                key=lambda x: x[1]['fit_metrics']['mse']
            )
            
            for model_name, result in sorted_models:
                metrics = result['fit_metrics']
                print(f"{model_name:<20} {metrics['r_squared']:<10.4f} "
                      f"{metrics['mse']:<12.6f} {metrics['l1_loss']:<12.6f}")
            
            print("\nBest performing model (lowest MSE):")
            best_model_name, best_result = sorted_models[0]
            print(f"  {best_model_name}: {best_result['description']}")
            print(f"  MSE = {best_result['fit_metrics']['mse']:.6f}")
            
            # Print parameter metrics summary if available
            models_with_param_metrics = [
                (name, result) for name, result in successful_models.items() 
                if 'parameter_metrics' in result and result['parameter_metrics'] is not None
            ]
            
            if models_with_param_metrics:
                print(f"\nParameter Quality Comparison:")
                print("-" * 110)
                print(f"{'Model':<20} {'MAPE (%)':<12} {'MSE':<12} {'Thick MAPE':<12} {'Rough MAPE':<12} {'SLD MAPE':<12}")
                print("-" * 110)
                
                # Sort by parameter MAPE
                sorted_param_models = sorted(
                    models_with_param_metrics,
                    key=lambda x: x[1]['parameter_metrics']['overall']['mape']
                )
                
                for model_name, result in sorted_param_models:
                    param_metrics = result['parameter_metrics']['overall']
                    by_type = result['parameter_metrics']['by_type']
                    print(f"{model_name:<20} {param_metrics['mape']:<12.2f} "
                          f"{param_metrics['mse']:<12.6f} {by_type['thickness_mape']:<12.2f} "
                          f"{by_type['roughness_mape']:<12.2f} {by_type['sld_mape']:<12.2f}")
                
                print("\nBest parameter prediction (lowest Parameter MAPE):")
                best_param_model_name, best_param_result = sorted_param_models[0]
                print(f"  {best_param_model_name}: {best_param_result['description']}")
                print(f"  Parameter MAPE = {best_param_result['parameter_metrics']['overall']['mape']:.2f}%")
                print(f"  Parameter MSE = {best_param_result['parameter_metrics']['overall']['mse']:.6f}")
        
    def parse_true_parameters_from_model_file(self, model_file_path):
        """
        Parse true parameters from s000000_model.txt file format.
        
        Args:
            model_file_path: Path to the model.txt file
            
        Returns:
            Dictionary containing true parameters organized by layer count and parameter names
        """
        print(f"Parsing true parameters from: {model_file_path}")
        
        if not Path(model_file_path).exists():
            print(f"Warning: Model file not found: {model_file_path}")
            return None
            
        # Read the model file
        with open(model_file_path, 'r') as f:
            lines = f.readlines()
        
        # Parse the data
        layers = []
        for line in lines:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
                
            parts = line.split()
            if len(parts) >= 4:
                layer_name = parts[0]
                sld = float(parts[1])  # Already in 10^-6 Å^-2 units
                thickness = float(parts[2]) if parts[2] != 'inf' else None
                roughness = float(parts[3]) if parts[3] != 'none' else None
                
                layers.append({
                    'name': layer_name,
                    'sld': sld,
                    'thickness': thickness,
                    'roughness': roughness
                })
        
        print(f"Parsed {len(layers)} layers from model file:")
        for layer in layers:
            print(f"  {layer['name']}: SLD={layer['sld']:.2e}, thickness={layer['thickness']}, roughness={layer['roughness']}")
        
        # Convert to parameter arrays for different layer configurations
        # Based on the model file structure:
        # fronting (ambient): SLD=3.50000e-06, roughness=8.72
        # layer1: SLD=1.14195e-05, thickness=242.09, roughness=24.16  
        # layer2: SLD=9.79919e-06, thickness=959.62, roughness=75.06
        # backing (substrate): SLD=6.45461e-06
        
        true_params_dict = {}
        
        # Handle different layer configurations
        if len(layers) >= 4:  # fronting + layer1 + layer2 + backing (2-layer system)
            # 2-layer configuration
            true_params_2layer = [
                layers[1]['thickness'],  # L1 thickness
                layers[2]['thickness'],  # L2 thickness
                layers[0]['roughness'],  # ambient/L1 roughness (fronting roughness)
                layers[1]['roughness'],  # L1/L2 roughness 
                layers[2]['roughness'],  # L2/substrate roughness
                layers[1]['sld'] * 1e6,  # L1 SLD (convert to 10^-6 units for display)
                layers[2]['sld'] * 1e6,  # L2 SLD
                layers[3]['sld'] * 1e6   # substrate SLD
            ]
            
            param_names_2layer = [
                "L1 thickness (Å)",
                "L2 thickness (Å)", 
                "ambient/L1 roughness (Å)",
                "L1/L2 roughness (Å)",
                "L2/substrate roughness (Å)",
                "L1 SLD (×10⁻⁶ Å⁻²)",
                "L2 SLD (×10⁻⁶ Å⁻²)",
                "substrate SLD (×10⁻⁶ Å⁻²)"
            ]
            
            true_params_dict['2_layer'] = {
                'params': true_params_2layer,
                'param_names': param_names_2layer
            }
            
            # 1-layer configuration - combine both physical layers
            combined_thickness = layers[1]['thickness'] + layers[2]['thickness']
            # Weighted average SLD
            total_volume = layers[1]['thickness'] + layers[2]['thickness']
            weighted_sld = (layers[1]['sld'] * layers[1]['thickness'] + 
                          layers[2]['sld'] * layers[2]['thickness']) / total_volume
            
            true_params_1layer = [
                combined_thickness,        # L1 thickness (combined)
                layers[0]['roughness'],    # ambient/L1 roughness
                layers[2]['roughness'],    # L1/substrate roughness
                weighted_sld * 1e6,        # L1 SLD (weighted average)
                layers[3]['sld'] * 1e6     # substrate SLD
            ]
            
            param_names_1layer = [
                "L1 thickness (Å)",
                "ambient/L1 roughness (Å)",
                "L1/substrate roughness (Å)",
                "L1 SLD (×10⁻⁶ Å⁻²)",
                "substrate SLD (×10⁻⁶ Å⁻²)"
            ]
            
            true_params_dict['1_layer'] = {
                'params': true_params_1layer,
                'param_names': param_names_1layer
            }
        
        elif len(layers) == 3:  # fronting + layer1 + backing (1-layer system)
            # 1-layer configuration with single physical layer
            true_params_1layer = [
                layers[1]['thickness'],    # L1 thickness
                layers[0]['roughness'],    # ambient/L1 roughness
                layers[1]['roughness'],    # L1/substrate roughness  
                layers[1]['sld'] * 1e6,    # L1 SLD
                layers[2]['sld'] * 1e6     # substrate SLD
            ]
            
            param_names_1layer = [
                "L1 thickness (Å)",
                "ambient/L1 roughness (Å)",
                "L1/substrate roughness (Å)",
                "L1 SLD (×10⁻⁶ Å⁻²)",
                "substrate SLD (×10⁻⁶ Å⁻²)"
            ]
            
            true_params_dict['1_layer'] = {
                'params': true_params_1layer,
                'param_names': param_names_1layer
            }
        
        return true_params_dict

    def get_true_params_for_model(self, model_config, true_params_dict):
        """
        Get the appropriate true parameters for a specific model configuration.
        
        Args:
            model_config: Model configuration dictionary
            true_params_dict: Dictionary of true parameters by layer count
            
        Returns:
            Tuple of (true_params_array, param_names) or (None, None) if not found
        """
        if true_params_dict is None:
            return None, None
        
        # Determine layer count from parameter names
        param_names = model_config.get('parameter_names', [])
        layer_count = None
        
        # Count layers based on parameter names
        if any('L2' in name for name in param_names):
            layer_count = '2_layer'
        elif any('L1' in name for name in param_names):
            layer_count = '1_layer'
        
        if layer_count and layer_count in true_params_dict:
            true_data = true_params_dict[layer_count]
            return np.array(true_data['params']), true_data['param_names']
        
        return None, None

    def get_experiment_name(self):
        """Extract experiment name from data path or description."""
        # Try to extract from data path first
        data_path = Path(self.data_config['data_path'])
        
        # Look for experiment ID patterns (like s123456)
        if '_experimental_curve.dat' in data_path.name:
            exp_id = data_path.name.replace('_experimental_curve.dat', '')
            return exp_id
        elif '_exp_curve.dat' in data_path.name:
            exp_id = data_path.name.replace('_exp_curve.dat', '')
            return exp_id
        elif data_path.stem.startswith('s') and data_path.stem[1:].isdigit():
            return data_path.stem
        
        # Try to extract from description
        description = self.data_config.get('description', '')
        if 'for ' in description:
            # Extract part after 'for '
            parts = description.split('for ')
            if len(parts) > 1:
                exp_part = parts[1].split(' ')[0]  # Get first word after 'for '
                if exp_part:
                    return exp_part
        
        # Fallback to filename stem
        return data_path.stem
    
    def find_true_parameters_file(self):
        """
        Automatically find the true parameters file based on naming convention.
        Looks for a file ending with '_model.txt' in the same directory as the data file.
        
        Returns:
            str: Path to the true parameters file if found, None otherwise
        """
        data_path = Path(self.data_config['data_path'])
        data_dir = data_path.parent
        
        # Extract the experiment ID (e.g., 's000000' from 's000000_experimental_curve.dat')
        data_filename = data_path.stem
        
        # Extract experiment ID by removing '_experimental_curve' suffix
        if '_experimental_curve' in data_filename:
            experiment_id = data_filename.replace('_experimental_curve', '')
        else:
            # Fallback: assume the whole filename is the experiment ID
            experiment_id = data_filename
        
        # Look for corresponding model file
        model_filename = f"{experiment_id}_model.txt"
        model_path = data_dir / model_filename
        
        if model_path.exists():
            return str(model_path)
        
        # Also check if there's a subdirectory with the experiment ID
        subdir_path = data_dir / experiment_id / model_filename
        if subdir_path.exists():
            return str(subdir_path)
        
        return None

    @staticmethod
    def run_experiment_inference(experiment_id, models_list, data_directory="data", priors_type="broad", output_dir="inference_results", layer_count=None):
        """
        Static method to run inference on a single experiment for batch processing.
        
        Args:
            experiment_id: Experiment ID (e.g., 's000000')
            models_list: List of model names to test
            data_directory: Base data directory
            layer_count: Number of layers (1 or 2). If None, will auto-detect.
            priors_type: Type of priors ('broad' or 'narrow')
            output_dir: Output directory for results
        
        Returns:
            dict: Results dictionary with experiment info and model results
        """
        try:
            # Create pipeline instance in batch mode
            pipeline = InferencePipeline(
                experiment_id=experiment_id,
                models_list=models_list,
                data_directory=data_directory,
                priors_type=priors_type,
                output_dir=output_dir,
                layer_count=layer_count
            )
            
            # Run inference on all models
            pipeline.run_all_models(show_plots=False)  # Don't show plots in batch mode
            
            # Extract and format results
            results = {
                'exp_id': experiment_id,
                'layer_count': pipeline.determine_layer_count(),
                'priors_type': priors_type,
                'models_results': {},
                'success': False,  # Will be set to True if any model succeeds
                'error': None
            }
            
            # Track if any model succeeded
            any_model_succeeded = False
            
            # Extract results from each model
            for model_name, model_result in pipeline.results.items():
                if model_result['success']:
                    any_model_succeeded = True
                    results['models_results'][model_name] = {
                        'success': True,
                        'fit_metrics': model_result['fit_metrics'],
                        'predicted_params': model_result['predicted_params'].tolist() if hasattr(model_result['predicted_params'], 'tolist') else model_result['predicted_params'],
                        'polished_params': model_result['polished_params'].tolist() if hasattr(model_result['polished_params'], 'tolist') else model_result['polished_params'],
                        'param_names': model_result['param_names']
                    }
                    
                    # Add parameter metrics if available
                    if 'parameter_metrics' in model_result and model_result['parameter_metrics'] is not None:
                        results['models_results'][model_name]['parameter_metrics'] = model_result['parameter_metrics']
                else:
                    results['models_results'][model_name] = {
                        'success': False,
                        'error': model_result.get('error', 'Unknown error')
                    }
            
            # Set experiment success based on whether any model succeeded
            results['success'] = any_model_succeeded
            
            return results
            
        except Exception as e:
            return {
                'exp_id': experiment_id,
                'success': False,
                'error': str(e),
                'models_results': {}
            }

def main():
    """Main execution function."""
    # Parse command line arguments
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        # Default to membrane configuration
        config_file = "configs/membrane_config.json"
    
    print("ReflecTorch Multi-Model Inference Pipeline")
    print(f"Using configuration: {config_file}")
    print("=" * 50)
    
    try:
        # Initialize and run pipeline
        pipeline = InferencePipeline(config_file)
        pipeline.run_all_models()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nAvailable configurations:")
        print("  - configs/membrane_config.json (membrane analysis with dQ/Q=0.1)")
        print("  - configs/s000000_config.json (s000000 data analysis)")
        print("  - configs/s000004_config.json (s000004 data analysis - single layer)")
        print("\nUsage: python inference_pipeline.py [config_file]")
        sys.exit(1)
    except Exception as e:
        print(f"Error running pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
