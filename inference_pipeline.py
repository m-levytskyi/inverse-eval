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

# Configuration Constants
NARROW_PRIORS_DEVIATION = 0.05  # e.g 0.25 for 25% deviation from true values (0.75-1.25 range)

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
                 error_bar_threshold=0.5, consecutive_threshold=3, remove_singles=False, preprocess=True,
                 custom_priors=None):
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
            preprocess: Whether to apply preprocessing to the experimental data (default: True)
        """
        print("DEBUG: Initializing InferencePipeline")
        print(f"  - config_file: {config_file}")
        print(f"  - output_dir: {output_dir}")
        print(f"  - experiment_id: {experiment_id}")
        print(f"  - models_list: {models_list}")
        print(f"  - data_directory: {data_directory}")
        print(f"  - priors_type: {priors_type}")
        print(f"  - layer_count: {layer_count}")
        print(f"  - error_bar_threshold: {error_bar_threshold}")
        print(f"  - consecutive_threshold: {consecutive_threshold}")
        print(f"  - remove_singles: {remove_singles}")
        print(f"  - preprocess: {preprocess}")
        print(f"  - custom_priors: {'Provided' if custom_priors is not None else 'Not provided'}")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.layer_count_override = layer_count  # Store for override
        self.custom_priors = custom_priors

        # Store preprocessing parameters
        self.preprocess = preprocess
        self.error_bar_threshold = error_bar_threshold
        self.consecutive_threshold = consecutive_threshold
        self.remove_singles = remove_singles

        # Initialize results storage
        self.results = {}
        
        if experiment_id is not None:
            print("DEBUG: Running in batch mode (initialized with experiment_id)")
            # Batch mode - initialize from experiment ID
            self.experiment_id = experiment_id
            self.data_directory = Path(data_directory)
            self.models_list = models_list or []
            self.priors_type = priors_type
            
            # Auto-discover experiment files
            print("DEBUG: Starting experiment file discovery...")
            self.discover_experiment_files()
            
            # Load experimental data
            print("DEBUG: Loading experimental data from discovered files...")
            self.load_experimental_data_from_files()
            
            # Load true parameters
            print("DEBUG: Loading true parameters from discovered files...")
            self.load_true_parameters_from_files()
            
            # Generate model configuration
            print("DEBUG: Generating model configurations...")
            self.generate_model_configurations()
            
        else:
            # Legacy mode - initialize from config file
            print("DEBUG: Running in legacy mode (initialized with config_file)")
            if config_file is None:
                raise ValueError("Either config_file or experiment_id must be provided")
            
            self.config_file = Path(config_file)
            
            # Load configuration
            print("DEBUG: Loading configuration from JSON file...")
            self.load_configuration()
            
            # Load experimental data
            print("DEBUG: Loading experimental data from configuration...")
            self.load_experimental_data()
            
            # Load true parameters if available
            print("DEBUG: Attempting to load true parameters...")
            self.true_params_dict = None
            true_params_file = self.find_true_parameters_file()
            if true_params_file:
                print(f"DEBUG: Found true parameters file via find_true_parameters_file: {true_params_file}")
                self.true_params_dict = self.parse_true_parameters_from_model_file(true_params_file)
            elif 'true_parameters_file' in self.data_config:
                print(f"DEBUG: Found true parameters file in data_config: {self.data_config['true_parameters_file']}")
                self.true_params_dict = self.parse_true_parameters_from_model_file(
                    self.data_config['true_parameters_file']
                )
            else:
                print("DEBUG: No true parameters file found or specified.")
    
    def discover_experiment_files(self):
        """Auto-discover experiment files based on experiment ID."""
        print(f"DEBUG: Discovering files for experiment_id: {self.experiment_id}")
        self.exp_data_file = None
        self.exp_model_file = None
        
        # Common file patterns
        exp_curve_pattern = f"{self.experiment_id}_experimental_curve.dat"
        model_pattern = f"{self.experiment_id}_model.txt"
        print(f"DEBUG: Using patterns: data='{exp_curve_pattern}', model='{model_pattern}'")
        
        # Search in various locations
        search_paths = [
            self.data_directory,
            self.data_directory / "MARIA_VIPR_dataset" / "1",
            self.data_directory / "MARIA_VIPR_dataset" / "2",
            self.data_directory / self.experiment_id,
        ]
        print(f"DEBUG: Search paths: {search_paths}")
        
        for search_path in search_paths:
            if not search_path.exists():
                print(f"DEBUG: Search path does not exist, skipping: {search_path}")
                continue
                
            # Look for files directly in the path
            print(f"DEBUG: Searching in: {search_path}")
            exp_file = search_path / exp_curve_pattern
            model_file = search_path / model_pattern
            
            if exp_file.exists():
                print(f"DEBUG: Found potential data file: {exp_file}")
                self.exp_data_file = exp_file
            if model_file.exists():
                print(f"DEBUG: Found potential model file: {model_file}")
                self.exp_model_file = model_file
                
            # If both found, we're done
            if self.exp_data_file and self.exp_model_file:
                print("DEBUG: Both data and model files found. Stopping search.")
                break
                
            # Also search subdirectories
            print(f"DEBUG: Searching subdirectories of {search_path}")
            for subdir in search_path.iterdir():
                if subdir.is_dir():
                    exp_file = subdir / exp_curve_pattern
                    model_file = subdir / model_pattern
                    
                    if exp_file.exists() and not self.exp_data_file:
                        print(f"DEBUG: Found data file in subdirectory: {exp_file}")
                        self.exp_data_file = exp_file
                    if model_file.exists() and not self.exp_model_file:
                        print(f"DEBUG: Found model file in subdirectory: {model_file}")
                        self.exp_model_file = model_file
                        
                    if self.exp_data_file and self.exp_model_file:
                        break
            
            if self.exp_data_file and self.exp_model_file:
                print("DEBUG: Both files found in subdirectories. Stopping search.")
                break
        
        if not self.exp_data_file:
            print(f"DEBUG: ERROR - Experimental data file for {self.experiment_id} not found after search.")
            raise FileNotFoundError(f"Could not find experimental data file for {self.experiment_id}")
        if not self.exp_model_file:
            print(f"DEBUG: ERROR - Model file for {self.experiment_id} not found after search.")
            raise FileNotFoundError(f"Could not find model file for {self.experiment_id}")
            
        print(f"Found experiment files:")
        print(f"  Data: {self.exp_data_file}")
        print(f"  Model: {self.exp_model_file}")
    
    def load_experimental_data_from_files(self):
        """Load experimental data from discovered files."""
        print(f"DEBUG: Loading experimental data from: {self.exp_data_file}")
        
        # Load the experimental data
        data = np.loadtxt(self.exp_data_file)
        
        # Determine format based on number of columns
        print(f"DEBUG: Data shape: {data.shape}")
        if data.shape[1] == 3:
            # 3-column format: Q, R, dR
            print("DEBUG: Detected 3-column format (Q, R, dR)")
            self.q_exp = data[:, 0]
            self.curve_exp = data[:, 1]
            self.sigmas_exp = data[:, 2]
            self.q_res_exp = None
        elif data.shape[1] == 4:
            # 4-column format: Q, R, dR, dQ
            print("DEBUG: Detected 4-column format (Q, R, dR, dQ)")
            self.q_exp = data[:, 0]
            self.curve_exp = data[:, 1]
            self.sigmas_exp = data[:, 2]
            self.q_res_exp = data[:, 3]
        else:
            print(f"DEBUG: ERROR - Unsupported data format with {data.shape[1]} columns.")
            raise ValueError(f"Unsupported data format: {data.shape[1]} columns")
        
        print(f"Loaded {len(self.q_exp)} data points")
        print(f"Q range: {self.q_exp.min():.4f} - {self.q_exp.max():.4f} Å⁻¹")
        
        # Apply preprocessing steps
        if self.preprocess:
            print("DEBUG: Preprocessing is enabled. Calling preprocess_experimental_data.")
            self.preprocess_experimental_data()
    
    def load_true_parameters_from_files(self):
        """Load true parameters from discovered model file."""
        if self.exp_model_file:
            print(f"DEBUG: Loading true parameters from: {self.exp_model_file}")
            self.true_params_dict = self.parse_true_parameters_from_model_file(str(self.exp_model_file))
        else:
            self.true_params_dict = None
            print("DEBUG: No experiment model file found, cannot load true parameters.")
    
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
        threshold = self.error_bar_threshold if error_bar_threshold is None else error_bar_threshold
        consecutive = self.consecutive_threshold if consecutive_threshold is None else consecutive_threshold
        singles = self.remove_singles if remove_singles is None else remove_singles
        
        print(f"DEBUG: Starting preprocessing with parameters:")
        print(f"  - error_bar_threshold: {threshold}")
        print(f"  - consecutive_threshold: {consecutive}")
        print(f"  - remove_singles: {singles}")

        print(f"Applying preprocessing to experimental data...")
        original_points = len(self.q_exp)
        
        # Step 1: Remove negative intensity values
        print("DEBUG: Step 1 - Removing negative intensity values.")
        positive_mask = self.curve_exp > 0
        if not np.all(positive_mask):
            negative_count = np.sum(~positive_mask)
            print(f"DEBUG: Found {negative_count} points with negative intensity. Removing them.")
            
            self.q_exp = self.q_exp[positive_mask]
            self.curve_exp = self.curve_exp[positive_mask]
            self.sigmas_exp = self.sigmas_exp[positive_mask]
            if self.q_res_exp is not None:
                self.q_res_exp = self.q_res_exp[positive_mask]
        else:
            print("DEBUG: No negative intensity values found.")
        
        # Step 1.5: Remove any invalid data (NaN, inf, or zero values that could cause issues)
        print("DEBUG: Step 1.5 - Removing invalid data (NaN, inf, zero).")
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
            print(f"DEBUG: Found {invalid_count} points with invalid values. Removing them.")
            
            self.q_exp = self.q_exp[finite_mask]
            self.curve_exp = self.curve_exp[finite_mask]
            self.sigmas_exp = self.sigmas_exp[finite_mask]
            if self.q_res_exp is not None:
                self.q_res_exp = self.q_res_exp[finite_mask]
        else:
            print("DEBUG: No invalid values found in this pass.")

        # Step 1.5: Clean up invalid data (NaN, inf, or zero values that would cause division issues)
        print("DEBUG: Step 1.5 - Cleaning up invalid data (NaN, inf, zero).")
        finite_mask = np.isfinite(self.curve_exp) & np.isfinite(self.sigmas_exp) & (self.curve_exp > 0) & (self.sigmas_exp >= 0)
        if not np.all(finite_mask):
            invalid_count = np.sum(~finite_mask)
            print(f"DEBUG: Found {invalid_count} points with invalid values. Removing them.")
            
            self.q_exp = self.q_exp[finite_mask]
            self.curve_exp = self.curve_exp[finite_mask]
            self.sigmas_exp = self.sigmas_exp[finite_mask]
            if self.q_res_exp is not None:
                self.q_res_exp = self.q_res_exp[finite_mask]
        else:
            print("DEBUG: No invalid values found in this pass.")

        # Step 1.5: Remove invalid data (NaN, inf, very small R values)
        # Check for NaN or inf values
        print("DEBUG: Step 1.5 - Removing invalid data (NaN, inf, very small R).")
        valid_mask = (np.isfinite(self.q_exp) & 
                     np.isfinite(self.curve_exp) & 
                     np.isfinite(self.sigmas_exp) &
                     (self.curve_exp > 1e-12))  # Remove very small R values that cause division issues
        
        if self.q_res_exp is not None:
            valid_mask = valid_mask & np.isfinite(self.q_res_exp)
        
        if not np.all(valid_mask):
            invalid_count = np.sum(~valid_mask)
            print(f"DEBUG: Found {invalid_count} points with invalid values. Removing them.")
            
            self.q_exp = self.q_exp[valid_mask]
            self.curve_exp = self.curve_exp[valid_mask]
            self.sigmas_exp = self.sigmas_exp[valid_mask]
            if self.q_res_exp is not None:
                self.q_res_exp = self.q_res_exp[valid_mask]
        else:
            print("DEBUG: No invalid values found in this pass.")
        
        # Step 2: Filter high error bars
        print(f"DEBUG: Step 2 - Filtering high error bars.")
        print(f"  Filtering high error bars (threshold: {threshold}, consecutive: {consecutive})")
        
        # Apply the filter_and_truncate function
        try:
            print("DEBUG: Applying filter_and_truncate function.")
            if self.q_res_exp is not None:
                # For 4-column data, also filter dQ
                print("DEBUG: 4-column data detected for filtering.")
                q_filtered, curve_filtered, sigmas_filtered = filter_and_truncate(
                    self.q_exp, self.curve_exp, self.sigmas_exp,
                    threshold=threshold,
                    consecutive=consecutive,
                    remove_singles=singles
                )
                
                # Check if filtering was too aggressive
                print("DEBUG: Checking if filtering was too aggressive.")
                original_q_range = self.q_exp.max() - self.q_exp.min()
                filtered_q_range = q_filtered.max() - q_filtered.min() if len(q_filtered) > 0 else 0
                points_retained = len(q_filtered) / len(self.q_exp) if len(self.q_exp) > 0 else 0
                q_range_retained = filtered_q_range / original_q_range if original_q_range > 0 else 0
                
                # If filtering was too aggressive, use less stringent parameters
                if points_retained < 0.3 or q_range_retained < 0.1:
                    print(f"DEBUG: WARNING - Aggressive filtering detected. Retrying with more permissive parameters.")
                    print(f"    - Retained points: {points_retained:.1%}, Q-range: {q_range_retained:.1%}")
                    
                    # Try with more permissive parameters
                    new_threshold = max(threshold * 2, 1.0)
                    new_consecutive = max(consecutive + 2, 5)
                    print(f"DEBUG: Retrying with threshold={new_threshold}, consecutive={new_consecutive}, remove_singles=False")
                    q_filtered, curve_filtered, sigmas_filtered = filter_and_truncate(
                        self.q_exp, self.curve_exp, self.sigmas_exp,
                        threshold=new_threshold,  # Double threshold or use 1.0
                        consecutive=new_consecutive,  # Increase consecutive requirement
                        remove_singles=False  # Don't remove singles
                    )
                
                # Apply the filtered data
                print(f"DEBUG: Applying filtered data. Points before: {len(self.q_exp)}, after: {len(q_filtered)}")
                self.q_exp = q_filtered
                self.curve_exp = curve_filtered
                self.sigmas_exp = sigmas_filtered
                
                # Filter dQ to match the filtered data
                if self.q_res_exp is not None and len(self.q_exp) < len(self.q_res_exp):
                    print(f"DEBUG: Truncating q_res_exp from {len(self.q_res_exp)} to {len(self.q_exp)} points.")
                    self.q_res_exp = self.q_res_exp[:len(self.q_exp)]
            else:
                # For 3-column data
                print("DEBUG: 3-column data detected for filtering.")
                q_filtered, curve_filtered, sigmas_filtered = filter_and_truncate(
                    self.q_exp, self.curve_exp, self.sigmas_exp,
                    threshold=threshold,
                    consecutive=consecutive,
                    remove_singles=singles
                )
                
                # Check if filtering was too aggressive
                print("DEBUG: Checking if filtering was too aggressive.")
                original_q_range = self.q_exp.max() - self.q_exp.min()
                filtered_q_range = q_filtered.max() - q_filtered.min() if len(q_filtered) > 0 else 0
                points_retained = len(q_filtered) / len(self.q_exp) if len(self.q_exp) > 0 else 0
                q_range_retained = filtered_q_range / original_q_range if original_q_range > 0 else 0
                
                # If filtering was too aggressive, use less stringent parameters
                if points_retained < 0.3 or q_range_retained < 0.1:
                    print(f"DEBUG: WARNING - Aggressive filtering detected. Retrying with more permissive parameters.")
                    print(f"    - Retained points: {points_retained:.1%}, Q-range: {q_range_retained:.1%}")
                    
                    # Try with more permissive parameters
                    new_threshold = max(threshold * 2, 1.0)
                    new_consecutive = max(consecutive + 2, 5)
                    print(f"DEBUG: Retrying with threshold={new_threshold}, consecutive={new_consecutive}, remove_singles=False")
                    q_filtered, curve_filtered, sigmas_filtered = filter_and_truncate(
                        self.q_exp, self.curve_exp, self.sigmas_exp,
                        threshold=new_threshold,  # Double threshold or use 1.0
                        consecutive=new_consecutive,  # Increase consecutive requirement
                        remove_singles=False  # Don't remove singles
                    )
                
                # Apply the filtered data
                print(f"DEBUG: Applying filtered data. Points before: {len(self.q_exp)}, after: {len(q_filtered)}")
                self.q_exp = q_filtered
                self.curve_exp = curve_filtered
                self.sigmas_exp = sigmas_filtered
        except Exception as e:
            print(f"DEBUG: ERROR during error bar filtering: {e}. Continuing with current data.")
            # Continue with the data we have
        
        final_points = len(self.q_exp)
        removed_points = original_points - final_points
        final_q_range = self.q_exp.max() - self.q_exp.min() if final_points > 0 else 0
        
        # Check if we have enough data points left
        print(f"DEBUG: Final check on preprocessed data. Points remaining: {final_points}")
        if final_points < 10:  # Increased minimum from 3 to 10
            print(f"DEBUG: ERROR - Insufficient data points after preprocessing: {final_points}")
            raise ValueError(f"Insufficient data points after preprocessing: {final_points} points remaining (minimum 10 required)")
        
        # Check if we have a reasonable Q range for model inference
        if final_q_range < 0.05:  # Minimum Q range of 0.05 Å⁻¹
            print(f"DEBUG: WARNING - Very narrow Q range after preprocessing: {final_q_range:.4f} Å⁻¹")
            # This might cause model inference issues, but we'll continue
        
        print(f"  Preprocessing complete:")
        print(f"    Original points: {original_points}")
        print(f"    Final points: {final_points}")
        print(f"    Removed points: {removed_points} ({100*removed_points/original_points if original_points > 0 else 0:.1f}%)")
        print(f"    Final Q range: {self.q_exp.min():.4f} - {self.q_exp.max():.4f} Å⁻¹ (span: {final_q_range:.4f})")

    def generate_model_configurations(self):
        """Generate model configurations for batch processing, with optional custom priors."""
        print("DEBUG: Generating model configurations for batch processing.")
        if not self.models_list:
            raise ValueError("No models specified for batch processing")

        # Load MARIA bounds for priors
        print("DEBUG: Loading MARIA bounds for priors.")
        maria_bounds = self.load_maria_bounds()

        # Determine layer count from true parameters
        print("DEBUG: Determining layer count.")
        layer_count = self.determine_layer_count()
        print(f"DEBUG: Determined layer count: {layer_count}")

        # Get parameter names for this layer count
        print("DEBUG: Getting parameter names for the layer count.")
        parameter_names = self.get_parameter_names_for_layer_count(layer_count)
        print(f"DEBUG: Parameter names: {parameter_names}")

        # Use custom priors if provided, else use MARIA or default
        if self.custom_priors is not None:
            prior_bounds = self.custom_priors
            print("DEBUG: Using provided custom priors.")
        else:
            print("DEBUG: No custom priors provided, getting priors for layer count.")
            prior_bounds = self.get_priors_for_layer_count(layer_count, maria_bounds)
        
        print(f"DEBUG: Final prior bounds to be used: {prior_bounds}")

        # Generate model configurations
        self.model_configs = {}
        for model_name in self.models_list:
            print(f"DEBUG: Generating config for model: {model_name}")
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
        print("DEBUG: Determining layer count...")
        # If layer count is explicitly provided, use it
        if hasattr(self, 'layer_count_override') and self.layer_count_override is not None:
            print(f"DEBUG: Using layer_count_override: {self.layer_count_override}")
            return self.layer_count_override
            
        if not self.true_params_dict:
            # Default assumption - could be improved
            print("DEBUG: No true_params_dict found. Defaulting to 2 layers.")
            return 2
            
        # Count actual material layers from the parsed model file
        # The true_params_dict might have multiple entries, but we want the natural layer count
        
        # Look for the most natural layer count based on available data
        # Priority: 2_layer (most common), then 1_layer
        print("DEBUG: Searching for layer count in true_params_dict keys.")
        for layer_key in ['2_layer', '1_layer']:
            if layer_key in self.true_params_dict:
                determined_layers = int(layer_key.split('_')[0])
                print(f"DEBUG: Found '{layer_key}' in true_params_dict. Setting layer count to {determined_layers}.")
                return determined_layers
        
        print("DEBUG: No layer-specific keys found in true_params_dict. Defaulting to 2 layers.")
        return 2  # Default
    
    def load_maria_bounds(self):
        """Load MARIA dataset prior bounds."""
        bounds_file = Path("maria_dataset_prior_bounds.json")
        print(f"DEBUG: Attempting to load MARIA bounds from {bounds_file}")
        
        if not bounds_file.exists():
            print("DEBUG: WARNING - MARIA bounds file not found. Will use default bounds.")
            return None
            
        with open(bounds_file, 'r') as f:
            print("DEBUG: MARIA bounds file found and loaded.")
            return json.load(f)
    
    def get_priors_for_layer_count(self, layer_count, maria_bounds):
        """Get appropriate priors for the layer count."""
        print(f"DEBUG: Getting priors for layer_count={layer_count}.")
        if not maria_bounds:
            print("DEBUG: No MARIA bounds provided. Falling back to default priors.")
            return self.get_default_priors(layer_count)
        
        layer_key = f"{layer_count}_layers"
        if layer_key not in maria_bounds:
            print(f"DEBUG: '{layer_key}' not in MARIA bounds. Falling back to default priors.")
            return self.get_default_priors(layer_count)
        
        print(f"DEBUG: Using priors from MARIA bounds for '{layer_key}'.")
        layer_data = maria_bounds[layer_key]
        parameters = layer_data['parameters']
        
        bounds = []
        
        if layer_count == 1:
            # 1-layer: [thickness, amb_rough, sub_rough, layer_sld, sub_sld]
            print("DEBUG: Extracting bounds for 1-layer system.")
            bounds = self.extract_bounds_1_layer(parameters)
        elif layer_count == 2:
            # 2-layer bounds
            print("DEBUG: Extracting bounds for 2-layer system.")
            bounds = self.extract_bounds_2_layer(parameters)
        
        print(f"DEBUG: Extracted broad prior bounds: {bounds}")
        
        # Apply priors type (broad vs narrow)
        if self.priors_type == "narrow":
            print("DEBUG: Priors type is 'narrow'. Applying narrow priors.")
            bounds = self.apply_narrow_priors(bounds, parameters, layer_count)
            print(f"DEBUG: Applied narrow prior bounds: {bounds}")
        else:
            print("DEBUG: Priors type is 'broad'. Using extracted bounds directly.")
        
        # Validate all bounds
        validated_bounds = []
        for i, bound in enumerate(bounds):
            min_val, max_val = bound
            if min_val >= max_val:
                print(f"DEBUG: WARNING - Invalid bound at index {i}: min ({min_val}) >= max ({max_val}). Adjusting max.")
                max_val = min_val + 0.1
            validated_bounds.append([min_val, max_val])
        
        if validated_bounds != bounds:
            print(f"DEBUG: Validated bounds: {validated_bounds}")
        
        return validated_bounds
    
    def get_default_priors(self, layer_count):
        """Get default prior bounds when MARIA bounds are not available."""
        print(f"DEBUG: Using default priors for layer_count={layer_count}.")
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
        print("DEBUG: Extracting 1-layer bounds from MARIA structure.")
        
        # Layer thickness - use overall thickness if available
        if 'overall' in parameters and 'thickness' in parameters['overall']:
            thickness_data = parameters['overall']['thickness']
            bounds.append([thickness_data['min'], thickness_data['max']])
        else:
            print("DEBUG: Fallback for 1-layer thickness.")
            bounds.append([10.0, 500.0])
        
        # Roughness parameters
        if 'interface' in parameters and 'fronting_roughness' in parameters['interface']:
            fronting_data = parameters['interface']['fronting_roughness']
            min_val = max(0.1, fronting_data['min'])  # Ensure minimum of 0.1
            max_val = max(min_val + 0.1, fronting_data['max'])  # Ensure max > min
            bounds.append([min_val, max_val])
        else:
            print("DEBUG: Fallback for 1-layer fronting roughness.")
            bounds.append([1.0, 10.0])
        
        if 'interface' in parameters and 'backing_roughness' in parameters['interface']:
            backing_data = parameters['interface']['backing_roughness']
            min_val = max(0.1, backing_data['min'])  # Ensure minimum of 0.1
            max_val = max(min_val + 0.1, backing_data['max'])  # Ensure max > min
            bounds.append([min_val, max_val])
        else:
            print("DEBUG: Fallback for 1-layer backing roughness.")
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
            print("DEBUG: Fallback for 1-layer SLD.")
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
            print("DEBUG: Fallback for 1-layer backing SLD.")
            bounds.append([-5.0, 15.0])
        
        return bounds
    
    def extract_bounds_2_layer(self, parameters):
        """Extract bounds for 2-layer system from MARIA structure."""
        bounds = []
        print("DEBUG: Extracting 2-layer bounds from MARIA structure.")
        
        # Extract bounds from the correct MARIA structure
        # For 2-layer: [L1_thick, L2_thick, amb_rough, L1L2_rough, L2sub_rough, L1_sld, L2_sld, sub_sld]
        # Layer thicknesses
        if 'by_position' in parameters and 'layer_1' in parameters['by_position']:
            layer1_data = parameters['by_position']['layer_1']
            bounds.append([layer1_data['thickness']['min'], layer1_data['thickness']['max']])
        else:
            print("DEBUG: Fallback for 2-layer L1 thickness.")
            bounds.append([10.0, 500.0])
            
        if 'by_position' in parameters and 'layer_2' in parameters['by_position']:
            layer2_data = parameters['by_position']['layer_2']
            bounds.append([layer2_data['thickness']['min'], layer2_data['thickness']['max']])
        else:
            print("DEBUG: Fallback for 2-layer L2 thickness.")
            bounds.append([10.0, 500.0])
        
        # Roughness parameters (fronting, inter-layer, backing)
        if 'interface' in parameters and 'fronting_roughness' in parameters['interface']:
            fronting_data = parameters['interface']['fronting_roughness']
            bounds.append([fronting_data['min'], fronting_data['max']])
        else:
            print("DEBUG: Fallback for 2-layer fronting roughness.")
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
            print("DEBUG: Fallback for 2-layer inter-layer and backing roughness.")
            bounds.append([1.0, 60.0])
            bounds.append([1.0, 60.0])
        
        # SLD parameters
        if 'by_position' in parameters:
            layer1_sld = parameters['by_position']['layer_1']['sld']
            layer2_sld = parameters['by_position']['layer_2']['sld']
            bounds.append([layer1_sld['min'] * 1e6, layer1_sld['max'] * 1e6])  # Convert to 10^-6 units
            bounds.append([layer2_sld['min'] * 1e6, layer2_sld['max'] * 1e6])
        else:
            print("DEBUG: Fallback for 2-layer L1 and L2 SLD.")
            bounds.append([-5.0, 15.0])
            bounds.append([-5.0, 15.0])
            
        if 'interface' in parameters and 'backing_sld' in parameters['interface']:
            backing_sld = parameters['interface']['backing_sld']
            bounds.append([backing_sld['min'] * 1e6, backing_sld['max'] * 1e6])
        else:
            print("DEBUG: Fallback for 2-layer backing SLD.")
            bounds.append([-5.0, 15.0])
        
        return bounds
    
    def apply_narrow_priors(self, bounds, parameters, layer_count):
        """
        Apply narrow priors based on configurable deviation from true values.
        
        Uses NARROW_PRIORS_DEVIATION constant to determine the range around true values.
        Default 25% deviation creates 75-125% range around true values.
        
        Note: NARROW_PRIORS_DEVIATION ≥ 1.0 will set minimum bounds to 0 for positive 
        parameters, which may not be physically meaningful for thickness/roughness.
        
        This fixes the issue where narrow priors could exceed 25% MAPE due to 
        inappropriate constraining to MARIA dataset bounds.
        """
        print("DEBUG: Applying narrow priors.")
        # Use true parameters to create narrow priors around them
        if not self.true_params_dict:
            print("DEBUG: WARNING - No true parameters available for narrow priors. Falling back to old method.")
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
                    print(f"DEBUG: Clamping narrow prior min for roughness at index {i} to 0.0.")
                    new_min = 0.0
                    
                narrow_bounds.append([new_min, new_max])
            return narrow_bounds
        
        # Get true parameters for this layer configuration
        layer_key = f"{layer_count}_layer"
        if layer_key not in self.true_params_dict:
            print(f"DEBUG: WARNING - No true parameters for '{layer_key}'. Using broad bounds as fallback.")
            return bounds
        
        true_params = self.true_params_dict[layer_key]['params']
        param_names = self.true_params_dict[layer_key]['param_names']
        print(f"DEBUG: Using true parameters for narrow priors from '{layer_key}': {true_params}")
        
        # Create narrow bounds using configurable deviation
        narrow_bounds = []
        for i, (param_name, true_val) in enumerate(zip(param_names, true_params)):
            if true_val is None:
                # Fallback for problematic true values
                fallback_bound = bounds[i] if i < len(bounds) else [-5.0, 15.0]
                print(f"DEBUG: True value for '{param_name}' is None. Falling back to bound: {fallback_bound}")
                narrow_bounds.append(fallback_bound)
                continue
            
            if abs(true_val) < 1e-10:  # Essentially zero
                # For near-zero values, use a small symmetric range
                print(f"DEBUG: True value for '{param_name}' is near zero. Using fixed range [-0.01, 0.01].")
                narrow_bounds.append([-0.01, 0.01])
                continue
                
            # Calculate bounds using configurable deviation
            min_factor = 1.0 - NARROW_PRIORS_DEVIATION  # e.g., 0.75 for 25% deviation
            max_factor = 1.0 + NARROW_PRIORS_DEVIATION  # e.g., 1.25 for 25% deviation
            
            if true_val >= 0:
                # Positive values: straightforward scaling
                new_min = true_val * min_factor
                new_max = true_val * max_factor
            else:
                # Negative values: more negative is larger factor, less negative is smaller factor
                new_min = true_val * max_factor  # More negative (larger absolute value)
                new_max = true_val * min_factor  # Less negative (smaller absolute value)
            
            # Apply ONLY essential physical constraints (not MARIA bounds!)
            if 'roughness' in param_name.lower() and new_min < 0:
                print(f"DEBUG: Clamping negative min bound for roughness '{param_name}' to 0.0.")
                new_min = 0.0  # Roughness can't be negative
                
            if 'thickness' in param_name.lower() and new_min < 0:
                print(f"DEBUG: Clamping negative min bound for thickness '{param_name}' to 0.01.")
                new_min = 0.01  # Thickness can't be negative or zero
            
            # Note: We do NOT constrain to original MARIA bounds anymore!
            # This was the root cause of the >25% MAPE issue.
            # The deviation range is controlled by NARROW_PRIORS_DEVIATION constant.
            
            narrow_bounds.append([new_min, new_max])
            
        return narrow_bounds
    
    def get_parameter_names_for_layer_count(self, layer_count):
        """Get parameter names for the given layer count."""
        print(f"DEBUG: Getting parameter names for layer_count={layer_count}.")
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
        print(f"DEBUG: Loading legacy configuration from: {self.config_file}")
        
        if not self.config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)
        
        # Extract data and model configurations
        print("DEBUG: Extracting data_config and model_configurations from JSON.")
        self.data_config = self.config['data_config']
        self.model_configs = self.config['model_configurations']
        
        print(f"Loaded configuration for: {self.data_config['description']}")
        print(f"Data format: {self.data_config['data_format']}")
        print(f"Available models: {len(self.model_configs)}")
        
    def load_experimental_data(self):
        """Load experimental data from file based on configuration."""
        data_path = Path(self.data_config['data_path'])
        print(f"DEBUG: Loading legacy experimental data from: {data_path}")
        
        # Load data
        data = np.loadtxt(data_path, skiprows=1)
        print(f"DEBUG: Loaded legacy data with shape: {data.shape}")
        
        # Handle data format based on configuration
        if self.data_config['data_format'] == '3_column':
            # 3-column format: Q, R, dR
            print("DEBUG: Processing legacy 3-column format (Q, R, dR)")
            self.q_exp = data[:, 0]
            self.curve_exp = data[:, 1]
            self.sigmas_exp = data[:, 2]
            
            # Calculate Q-resolution using configured dQ/Q ratio
            dq_over_q = self.data_config.get('dq_over_q')
            if dq_over_q is None:
                print("DEBUG: ERROR - dq_over_q not specified for 3-column data.")
                raise ValueError("dq_over_q must be specified for 3-column data format")
            
            self.q_res_exp = self.q_exp * dq_over_q
            print(f"DEBUG: Calculated Q-resolution using dQ/Q = {dq_over_q}")
            
        elif self.data_config['data_format'] == '4_column':
            # 4-column format: Q, R, dR, dQ
            print("DEBUG: Processing legacy 4-column format (Q, R, dR, dQ)")
            
            # Apply max_points limit if specified
            max_points = self.data_config.get('max_points')
            if max_points is not None:
                print(f"DEBUG: Applying max_points limit: {max_points}")
                max_points = min(max_points, len(data))
                data = data[:max_points]
                print(f"Trimmed data to {max_points} points")
            
            self.q_exp = data[:, 0]
            self.curve_exp = data[:, 1]
            self.sigmas_exp = data[:, 2]
            self.q_res_exp = data[:, 3]
        else:
            print(f"DEBUG: ERROR - Unsupported legacy data format: {self.data_config['data_format']}")
            raise ValueError(f"Unsupported data format: {self.data_config['data_format']}")
        
        print(f"Loaded {len(self.q_exp)} data points")
        print(f"Q range: {self.q_exp.min():.4f} - {self.q_exp.max():.4f} Å⁻¹")
        if self.q_res_exp is not None:
            print(f"dQ range: {self.q_res_exp.min():.6f} - {self.q_res_exp.max():.6f} Å⁻¹")
        
        # Apply preprocessing steps
        if self.preprocess:
            print("DEBUG: Preprocessing is enabled for legacy data. Calling preprocess_experimental_data.")
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
        print(f"DEBUG: Running inference for model: {model_name}")
        print(f"  - Config name: {model_config['config_name']}")
        print(f"  - Weights format: {model_config.get('weights_format', 'pth')}")
        
        try:
            # Initialize model
            print("DEBUG: Initializing EasyInferenceModel.")
            inference_model = EasyInferenceModel(
                config_name=model_config['config_name'],
                device='cpu',
                weights_format=model_config.get('weights_format', 'pth')
            )
            
            # Interpolate data to model grid
            print("DEBUG: Interpolating experimental data to model's Q grid.")
            q_model, exp_curve_interp = inference_model.interpolate_data_to_model_q(
                self.q_exp, self.curve_exp
            )
            
            # Debug information for tensor concatenation issues
            print("DEBUG: Post-interpolation data summary:")
            print(f"Original data: {len(self.q_exp)} points, Q range: {self.q_exp.min():.4f} - {self.q_exp.max():.4f}")
            if self.q_res_exp is not None:
                print(f"Resolution data available")
            else:
                print(f"DEBUG: No resolution data (q_res_exp is None).")
            print(f"Interpolated data: {len(q_model)} points, Q range: {q_model.min():.4f} - {q_model.max():.4f}")
            
            # Check for potential issues that could cause tensor concatenation errors
            print("DEBUG: Checking for potential interpolation issues.")
            if len(q_model) == 0:
                raise ValueError("Model interpolation resulted in empty Q grid")
            if len(exp_curve_interp) == 0:
                raise ValueError("Model interpolation resulted in empty reflectivity curve")
            if len(q_model) != len(exp_curve_interp):
                raise ValueError(f"Q grid ({len(q_model)}) and curve ({len(exp_curve_interp)}) length mismatch")
            if np.any(~np.isfinite(exp_curve_interp)):
                print(f"  DEBUG: WARNING - Non-finite values found in interpolated curve.")
            if np.any(exp_curve_interp <= 0):
                print(f"  DEBUG: WARNING - Non-positive values found in interpolated curve.")
            
            # Ensure arrays have proper dimensions (1D) and correct data types
            print("DEBUG: Ensuring interpolated arrays have correct dimensions and types.")
            q_model = np.asarray(q_model, dtype=np.float32).flatten()
            exp_curve_interp = np.asarray(exp_curve_interp, dtype=np.float32).flatten()
            
            if len(q_model) != len(exp_curve_interp):
                raise ValueError(f"After processing: Q grid ({len(q_model)}) and curve ({len(exp_curve_interp)}) length mismatch")
            
            # Interpolate resolution data
            print("DEBUG: Interpolating resolution data.")
            if self.q_res_exp is not None:
                q_res_interp = np.interp(q_model, self.q_exp, self.q_res_exp)
                print("DEBUG: Interpolated q_res_exp to model Q grid.")
            else:
                # For 3-column data, assume a constant dQ/Q = 0.05 (5%)
                q_res_interp = q_model * 0.05
                print(f"  DEBUG: FALLBACK - No resolution data provided, using default dQ/Q = 5%.")
            
            # Ensure resolution array has proper dimensions and data type
            q_res_interp = np.asarray(q_res_interp, dtype=np.float32).flatten()
            
            print(f"Model Q grid: {len(q_model)} points, range: {q_model.min():.4f} - {q_model.max():.4f}")
            
            # Verify all arrays have the same length
            print("DEBUG: Verifying all interpolated arrays have the same length.")
            if not (len(q_model) == len(exp_curve_interp) == len(q_res_interp)):
                raise ValueError(f"Array length mismatch: Q={len(q_model)}, R={len(exp_curve_interp)}, dQ={len(q_res_interp)}")
            
            # Convert prior bounds from list format to tuple format
            prior_bounds = [tuple(bound) for bound in model_config['prior_bounds']]
            print(f"DEBUG: Converted prior bounds to tuples: {prior_bounds}")
            
            # Run prediction
            print("DEBUG: Calling inference_model.predict with the following parameters:")
            print(f"  - reflectivity_curve shape: {exp_curve_interp.shape}")
            print(f"  - prior_bounds: {prior_bounds}")
            print(f"  - q_values shape: {q_model.shape}")
            print(f"  - q_resolution shape: {q_res_interp.shape}")
            print(f"  - clip_prediction: True")
            print(f"  - polish_prediction: True")
            print(f"  - calc_pred_curve: True")
            print(f"  - calc_pred_sld_profile: True")
            print(f"  - calc_polished_sld_profile: True")

            prediction_dict = inference_model.predict(
                reflectivity_curve=exp_curve_interp,
                prior_bounds=prior_bounds,
                q_values=q_model,
                q_resolution=q_res_interp,
                clip_prediction=True,  # False - allows predictions outside prior bounds
                polish_prediction=True,
                calc_pred_curve=True,
                calc_pred_sld_profile=True,
                calc_polished_sld_profile=True,
            )
            
            print("DEBUG: `predict` call finished. Extracting results.")
            
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
                # Convert SLD values to proper units for display (×10⁻⁶ Å⁻²)
                if 'sld' in param_name.lower():
                    pred_display = pred_val * 1e6
                    polish_display = polish_val * 1e6
                    print(f'{param_name.ljust(25)} -> Predicted: {pred_display:8.2f}   Polished: {polish_display:8.2f}')
                    print(f"DEBUG: SLD converted for display - Original pred: {pred_val:.6e}, Display: {pred_display:.4f}")
                else:
                    print(f'{param_name.ljust(25)} -> Predicted: {pred_val:8.2f}   Polished: {polish_val:8.2f}')
            
            # Calculate fit quality metrics
            print("DEBUG: Calculating fit quality metrics.")
            result['fit_metrics'] = self.calculate_fit_metrics(
                self.curve_exp, prediction_dict['polished_curve'], 
                self.sigmas_exp, q_model
            )
            
            # Calculate parameter metrics - get true parameters for this model
            print("DEBUG: Getting true parameters for this model to calculate parameter metrics.")
            true_params, true_param_names = self.get_true_params_for_model(model_config, self.true_params_dict)
            if true_params is not None:
                print(f"DEBUG: True parameters found: {true_params}")
                try:
                    print("DEBUG: Calculating parameter metrics.")
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
                        
                        # Values are already converted to proper units in calculate_parameter_metrics
                        print(f"{param_name:<25} {pred_val:<12.2f} {true_val:<12.2f} "
                              f"{sq_error:<12.4f}")
                        
                        # Debug info for SLD parameters
                        if 'sld' in param_name.lower():
                            print(f"DEBUG: SLD comparison - Already converted values in ×10⁻⁶ units")
                    
                    # Print aggregate metrics
                    overall_metrics = result['parameter_metrics']['overall']
                    by_type = result['parameter_metrics']['by_type']
                    print(f"\nParameter Metrics Summary:")
                    print(f"Overall MAPE: {overall_metrics['mape']:.2f}%, Overall MSE: {overall_metrics['mse']:.6f}")
                    if by_type.get('thickness_mape', 0) > 0:
                        print(f"Thickness - MAPE: {by_type['thickness_mape']:.2f}%, MSE: {by_type['thickness_mse']:.6f}")
                    if by_type.get('roughness_mape', 0) > 0:
                        print(f"Roughness - MAPE: {by_type['roughness_mape']:.2f}%, MSE: {by_type['roughness_mse']:.6f}")
                    if by_type.get('sld_mape', 0) > 0:
                        print(f"SLD - MAPE: {by_type['sld_mape']:.2f}%, MSE: {by_type['sld_mse']:.6f}")
                        
                except Exception as e:
                    print(f"DEBUG: WARNING - Could not calculate parameter metrics: {e}")
                    result['parameter_metrics'] = None
            else:
                print("DEBUG: No true parameters available for this model. Skipping parameter metrics.")
                result['parameter_metrics'] = None
            
            print(f"\nFit Quality Metrics:")
            print(f"R-squared: {result['fit_metrics']['r_squared']:.4f}")
            print(f"MSE: {result['fit_metrics']['mse']:.6f}")
            print(f"L1 Loss: {result['fit_metrics']['l1_loss']:.6f}")
            
            print(f"DEBUG: Inference for model {model_name} succeeded.")
            return result
            
        except Exception as e:
            print(f"DEBUG: ERROR during inference for model {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'model_name': model_name,
                'success': False,
                'error': str(e)
            }
    
    def calculate_fit_metrics(self, y_exp, y_pred, sigma_exp, q_model):
        """Calculate fit quality metrics including MSE, R-squared, and L1 loss."""
        print("DEBUG: Calculating fit metrics (R-squared, MSE, L1 loss).")
        # Interpolate predicted curve to experimental Q points for comparison
        print(f"DEBUG: Interpolating predicted curve ({len(y_pred)} pts on q_model) to experimental Q points ({len(self.q_exp)} pts).")
        y_pred_interp = np.interp(self.q_exp, q_model, y_pred)
        
        # R-squared
        ss_res = np.sum((y_exp - y_pred_interp) ** 2)
        ss_tot = np.sum((y_exp - np.mean(y_exp)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # MSE (Mean Squared Error)
        mse = np.mean((y_exp - y_pred_interp) ** 2)
        
        # L1 Loss (Mean Absolute Error)
        l1_loss = np.mean(np.abs(y_exp - y_pred_interp))
        
        metrics = {
            'r_squared': r_squared,
            'mse': mse,
            'l1_loss': l1_loss
        }
        print(f"DEBUG: Calculated fit metrics: {metrics}")
        return metrics

    def calculate_parameter_metrics(self, pred_params, true_params, param_names):
        """Calculate parameter metrics: MAPE and MSE for different parameter types."""
        print("DEBUG: Calculating parameter metrics (MAPE, MSE).")
        print(f"  - Predicted params: {pred_params}")
        print(f"  - True params: {true_params}")
        print(f"  - Param names: {param_names}")

        if len(pred_params) != len(true_params):
            print(f"DEBUG: WARNING - Mismatch in parameter count. Predicted: {len(pred_params)}, True: {len(true_params)}. Cannot calculate metrics.")
            return {
                'overall': {'mape': -1, 'mse': -1},
                'by_type': {},
                'by_parameter': {}
            }

        # Convert arrays and handle unit conversions for SLD parameters
        pred_params_converted = []
        true_params_converted = []
        
        for i, param_name in enumerate(param_names):
            pred_val = pred_params[i]
            true_val = true_params[i]
            
            # Convert SLD values to same units for proper comparison
            if 'sld' in param_name.lower():
                # Both predicted and true SLD values need to be in same units
                # The model predicts in scientific notation, true values are in 10^-6 units
                # Convert both to 10^-6 units for comparison
                pred_converted = pred_val * 1e6
                true_converted = true_val  # Already in 10^-6 units from parsing
                print(f"DEBUG: SLD unit conversion for {param_name} - Pred: {pred_val:.6e} -> {pred_converted:.4f}, True: {true_val:.4f}")
            else:
                pred_converted = pred_val
                true_converted = true_val
                
            pred_params_converted.append(pred_converted)
            true_params_converted.append(true_converted)

        errors = np.array(pred_params_converted) - np.array(true_params_converted)
        squared_errors = errors ** 2
        
        # Avoid division by zero for MAPE calculation
        # Replace true zeros with a small number to avoid infinity, or handle them separately
        true_params_mape = np.array(true_params_converted)
        zero_mask = true_params_mape == 0
        
        # Calculate percentage error, handling true zeros
        percentage_errors = np.zeros_like(errors)
        non_zero_mask = ~zero_mask
        
        if np.any(non_zero_mask):
            percentage_errors[non_zero_mask] = np.abs(errors[non_zero_mask] / true_params_mape[non_zero_mask]) * 100
        
        # For true values that were zero, MAPE contribution is complex.
        # If prediction is also zero, error is 0. If not, it's infinite.
        # We'll cap the contribution for stability.
        percentage_errors[zero_mask] = np.where(np.abs(errors[zero_mask]) < 1e-9, 0, 100.0) # 100% error if pred is not zero

        # Overall metrics
        overall_mape = np.mean(percentage_errors)
        overall_mse = np.mean(squared_errors)
        
        # Metrics by parameter type
        thickness_mape, thickness_mse = [], []
        roughness_mape, roughness_mse = [], []
        sld_mape, sld_mse = [], []
        
        by_parameter_metrics = {}

        for i, name in enumerate(param_names):
            param_metrics = {
                'predicted': pred_params_converted[i],  # Use converted values
                'true': true_params_converted[i],       # Use converted values
                'error': errors[i],
                'squared_error': squared_errors[i],
                'percentage_error': percentage_errors[i]
            }
            by_parameter_metrics[name] = param_metrics

            if 'thickness' in name.lower():
                thickness_mape.append(percentage_errors[i])
                thickness_mse.append(squared_errors[i])
            elif 'roughness' in name.lower():
                roughness_mape.append(percentage_errors[i])
                roughness_mse.append(squared_errors[i])
            elif 'sld' in name.lower():
                sld_mape.append(percentage_errors[i])
                sld_mse.append(squared_errors[i])
        
        by_type_metrics = {
            'thickness_mape': np.mean(thickness_mape) if thickness_mape else 0,
            'thickness_mse': np.mean(thickness_mse) if thickness_mse else 0,
            'roughness_mape': np.mean(roughness_mape) if roughness_mape else 0,
            'roughness_mse': np.mean(roughness_mse) if roughness_mse else 0,
            'sld_mape': np.mean(sld_mape) if sld_mape else 0,
            'sld_mse': np.mean(sld_mse) if sld_mse else 0,
        }

        metrics = {
            'overall': {'mape': overall_mape, 'mse': overall_mse},
            'by_type': by_type_metrics,
            'by_parameter': by_parameter_metrics
        }
        print(f"DEBUG: Calculated parameter metrics: {metrics}")
        return metrics

    def get_true_params_for_model(self, model_config, true_params_dict):
        """Get true parameters corresponding to a given model configuration."""
        print(f"DEBUG: Getting true parameters for model '{model_config['config_name']}'.")
        if not true_params_dict:
            print("DEBUG: No true_params_dict available.")
            return None, None
        
        # Determine layer count from parameter names in model_config
        num_params = len(model_config.get('parameter_names', []))
        layer_count = 1 if num_params == 5 else (2 if num_params == 8 else None)
        
        if layer_count is None:
            print(f"DEBUG: Could not determine layer count from param count ({num_params}).")
            return None, None
            
        layer_key = f"{layer_count}_layer"
        print(f"DEBUG: Determined layer key: '{layer_key}'")
        
        if layer_key in true_params_dict:
            true_params = true_params_dict[layer_key]['params']
            true_param_names = true_params_dict[layer_key]['param_names']
            print(f"DEBUG: Found true params for '{layer_key}': {true_params}")
            return true_params, true_param_names
        else:
            print(f"DEBUG: No true params found for key '{layer_key}'.")
            return None, None

    def find_true_parameters_file(self):
        """Find the true parameters file based on the data path."""
        data_path = Path(self.data_config['data_path'])
        print(f"DEBUG: Finding true parameters file based on data path: {data_path}")
        
        # Example: s000000_experimental_curve.dat -> s000000_model.txt
        base_name = data_path.stem.replace('_experimental_curve', '')
        model_file = data_path.parent / f"{base_name}_model.txt"
        
        if model_file.exists():
            print(f"DEBUG: Found potential true parameters file: {model_file}")
            return str(model_file)
        else:
            print(f"DEBUG: Did not find a model file named '{model_file.name}' in '{data_path.parent}'.")
            return None

    def parse_true_parameters_from_model_file(self, model_file_path):
        """
        Parse true parameters from a Motofit-style model file.
        
        This function parses tabular model files where each row represents a layer:
        - fronting: ambient medium (usually air)
        - layer1, layer2, etc: material layers
        - backing: substrate
        
        Columns are: layer_name, sld(A^-2), thickness(A), roughness(A)
        """
        print(f"DEBUG: Parsing true parameters from model file: {model_file_path}")
        if not Path(model_file_path).exists():
            print(f"DEBUG: ERROR - Model file does not exist: {model_file_path}")
            return {}

        with open(model_file_path, 'r') as f:
            lines = f.readlines()

        # Parse the tabular format
        layers = {}
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 4:
                layer_name = parts[0]
                try:
                    sld = float(parts[1])
                    thickness_str = parts[2]
                    roughness_str = parts[3]
                    
                    # Handle special values for thickness
                    if thickness_str.lower() in ['inf', 'none', 'nan']:
                        thickness = float('inf')
                    else:
                        thickness = float(thickness_str)
                    
                    # Handle special values for roughness
                    if roughness_str.lower() in ['none', 'nan']:
                        roughness = 0.0
                    elif roughness_str.lower() == 'inf':
                        roughness = float('inf')
                    else:
                        roughness = float(roughness_str)
                    
                    layers[layer_name] = {
                        'sld': sld,
                        'thickness': thickness,
                        'roughness': roughness
                    }
                    print(f"DEBUG: Parsed layer '{layer_name}': sld={sld}, thickness={thickness}, roughness={roughness}")
                    
                except (ValueError, IndexError) as e:
                    print(f"DEBUG: Failed to parse line '{line}': {e}")
                    continue
        
        print(f"DEBUG: Parsed layers: {list(layers.keys())}")

        true_params_dict = {}

        # Count actual material layers (exclude fronting and backing)
        material_layers = [name for name in layers.keys() if name.startswith('layer')]
        num_material_layers = len(material_layers)
        print(f"DEBUG: Found {num_material_layers} material layers: {material_layers}")

        # --- Parse as 1-layer model (if we have 1 material layer) ---
        if num_material_layers == 1 and 'layer1' in layers:
            try:
                print("DEBUG: Attempting to parse as a 1-layer model.")
                
                fronting = layers.get('fronting', {})
                layer1 = layers.get('layer1', {})
                backing = layers.get('backing', {})
                
                # 1-layer parameters: [thickness, amb_rough, sub_rough, layer_sld, sub_sld]
                thickness = layer1.get('thickness', 0.0)
                amb_rough = fronting.get('roughness', 0.0)  # fronting roughness
                sub_rough = layer1.get('roughness', 0.0)   # layer1 roughness (interface with substrate)
                layer_sld = layer1.get('sld', 0.0) * 1e6   # Convert to 10^-6 units
                sub_sld = backing.get('sld', 0.0) * 1e6    # Convert to 10^-6 units
                
                params_1_layer = [thickness, amb_rough, sub_rough, layer_sld, sub_sld]
                names_1_layer = self.get_parameter_names_for_layer_count(1)
                
                true_params_dict['1_layer'] = {
                    'params': params_1_layer,
                    'param_names': names_1_layer
                }
                print(f"DEBUG: Successfully parsed as 1-layer model: {params_1_layer}")
                print(f"DEBUG: SLD values converted to 10^-6 units - layer_sld: {layer_sld:.2f}, sub_sld: {sub_sld:.2f}")
                
            except Exception as e:
                print(f"DEBUG: Failed to parse as 1-layer model: {e}")

        # --- Parse as 2-layer model (if we have 2 material layers) ---
        if num_material_layers >= 2 and 'layer1' in layers and 'layer2' in layers:
            try:
                print("DEBUG: Attempting to parse as a 2-layer model.")
                
                fronting = layers.get('fronting', {})
                layer1 = layers.get('layer1', {})
                layer2 = layers.get('layer2', {})
                backing = layers.get('backing', {})
                
                # 2-layer parameters: [L1_thick, L2_thick, amb_rough, L1L2_rough, L2sub_rough, L1_sld, L2_sld, sub_sld]
                l1_thick = layer1.get('thickness', 0.0)
                l2_thick = layer2.get('thickness', 0.0)
                amb_rough = fronting.get('roughness', 0.0)    # fronting roughness (ambient/L1 interface)
                l1l2_rough = layer1.get('roughness', 0.0)     # layer1 roughness (L1/L2 interface)
                l2sub_rough = layer2.get('roughness', 0.0)    # layer2 roughness (L2/substrate interface)
                l1_sld = layer1.get('sld', 0.0) * 1e6         # Convert to 10^-6 units
                l2_sld = layer2.get('sld', 0.0) * 1e6         # Convert to 10^-6 units
                sub_sld = backing.get('sld', 0.0) * 1e6       # Convert to 10^-6 units

                params_2_layer = [l1_thick, l2_thick, amb_rough, l1l2_rough, l2sub_rough, l1_sld, l2_sld, sub_sld]
                names_2_layer = self.get_parameter_names_for_layer_count(2)
                
                true_params_dict['2_layer'] = {
                    'params': params_2_layer,
                    'param_names': names_2_layer
                }
                print(f"DEBUG: Successfully parsed as 2-layer model: {params_2_layer}")
                print(f"DEBUG: SLD values converted to 10^-6 units - L1_sld: {l1_sld:.2f}, L2_sld: {l2_sld:.2f}, sub_sld: {sub_sld:.2f}")
                
            except Exception as e:
                print(f"DEBUG: Failed to parse as 2-layer model: {e}")

        # --- If we have more than 2 layers, we can still try to create a 2-layer equivalent ---
        # This would involve combining layers appropriately, but for now we'll skip this

        if not true_params_dict:
            print("DEBUG: WARNING - Could not parse model file for either 1 or 2 layers.")
            
        return true_params_dict

    def save_results(self):
        """Save all results to a JSON file."""
        timestamp = datetime.now().strftime("%d%b%Y_%H_%M")
        
        if self.experiment_id:
            output_filename = f"{self.experiment_id}_{self.priors_type}_{timestamp}.json"
        else:
            config_name = self.config_file.stem
            output_filename = f"{config_name}_results_{timestamp}.json"
        
        output_path = self.output_dir / output_filename
        print(f"DEBUG: Saving results to {output_path}")

        # Convert numpy arrays to lists for JSON serialization
        results_to_save = {}
        for model_name, result in self.results.items():
            if result['success']:
                results_to_save[model_name] = {
                    'model_name': result['model_name'],
                    'config_name': result['config_name'],
                    'description': result['description'],
                    'predicted_params': result['predicted_params'].tolist(),
                    'polished_params': result['polished_params'].tolist(),
                    'param_names': result['param_names'],
                    'fit_metrics': result['fit_metrics'],
                    'parameter_metrics': result['parameter_metrics'],
                    'success': True
                }
            else:
                results_to_save[model_name] = {
                    'model_name': result['model_name'],
                    'success': False,
                    'error': result['error']
                }

        with open(output_path, 'w') as f:
            json.dump(results_to_save, f, indent=4)
            
        print(f"Results saved to {output_path}")

    def plot_results(self):
        """Plot and save all results."""
        timestamp = datetime.now().strftime("%d%b%Y_%H_%M")
        
        if self.experiment_id:
            base_filename = f"{self.experiment_id}_{self.priors_type}_{timestamp}"
        else:
            config_name = self.config_file.stem
            base_filename = f"{config_name}_results_{timestamp}"
            
        plot_path_reflectivity = self.output_dir / f"{base_filename}_reflectivity.png"
        plot_path_sld = self.output_dir / f"{base_filename}_sld.png"
        
        print(f"DEBUG: Generating plots. Base filename: {base_filename}")

        # --- Plot Reflectivity Curves ---
        plt.figure(figsize=(10, 8))
        plt.errorbar(self.q_exp, self.curve_exp, yerr=self.sigmas_exp, fmt='o', 
                     label='Experimental Data', color='black', markersize=4, capsize=2)
        
        for model_name, result in self.results.items():
            if result['success']:
                plt.plot(result['q_model'], result['polished_curve'], 
                         label=f'{model_name} (Polished Fit)')
        
        plt.yscale('log')
        plt.xlabel('Q (Å⁻¹)')
        plt.ylabel('Reflectivity')
        plt.title(f'Model Comparison - Reflectivity ({self.experiment_id or self.config_file.stem})')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(plot_path_reflectivity)
        plt.close()
        print(f"Reflectivity plot saved to {plot_path_reflectivity}")

        # --- Plot SLD Profiles ---
        plt.figure(figsize=(10, 8))
        
        # Plot true SLD profile if available
        true_sld_x, true_sld_y = self.get_true_sld_profile()
        if true_sld_x is not None:
            plt.plot(true_sld_x, true_sld_y, label='True SLD Profile', color='black', linestyle='--', linewidth=2)

        for model_name, result in self.results.items():
            if result['success']:
                plt.plot(result['sld_profile_x'], result['sld_profile_polished'], 
                         label=f'{model_name} (Polished SLD)')
        
        plt.xlabel('Depth (Å)')
        plt.ylabel('SLD (×10⁻⁶ Å⁻²)')
        plt.title(f'Model Comparison - SLD Profile ({self.experiment_id or self.config_file.stem})')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(plot_path_sld)
        plt.close()
        print(f"SLD profile plot saved to {plot_path_sld}")

    def get_true_sld_profile(self):
        """Generate a true SLD profile from the parsed parameters."""
        print("DEBUG: Attempting to generate true SLD profile.")
        if not self.true_params_dict:
            print("DEBUG: No true parameters available to generate SLD profile.")
            return None, None

        # Prefer 2-layer interpretation if available
        layer_key = '2_layer' if '2_layer' in self.true_params_dict else '1_layer'
        if layer_key not in self.true_params_dict:
            print("DEBUG: No valid layer interpretation in true_params_dict.")
            return None, None
            
        print(f"DEBUG: Using '{layer_key}' interpretation for true SLD profile.")
        true_params = self.true_params_dict[layer_key]['params']
        
        # Ambient SLD is assumed to be 0
        ambient_sld = 0.0
        
        depth = [0]
        sld = [ambient_sld]

        if layer_key == '1_layer':
            # [thickness, amb_rough, sub_rough, layer_sld, sub_sld]
            thickness, _, _, layer_sld_val, sub_sld_val = true_params
            
            depth.extend([0, thickness, thickness])
            sld.extend([layer_sld_val, layer_sld_val, sub_sld_val])
            
            # Extend substrate for plotting
            depth.append(thickness + 50)
            sld.append(sub_sld_val)

        elif layer_key == '2_layer':
            # [L1_thick, L2_thick, amb_rough, L1L2_rough, L2sub_rough, L1_sld, L2_sld, sub_sld]
            l1_thick, l2_thick, _, _, _, l1_sld, l2_sld, sub_sld_val = true_params
            
            depth.extend([0, l1_thick, l1_thick, l1_thick + l2_thick, l1_thick + l2_thick])
            sld.extend([l1_sld, l1_sld, l2_sld, l2_sld, sub_sld_val])

            # Extend substrate for plotting
            depth.append(l1_thick + l2_thick + 50)
            sld.append(sub_sld_val)
            
        # Sort by depth for correct plotting
        sorted_points = sorted(zip(depth, sld))
        x = [p[0] for p in sorted_points]
        y = [p[1] for p in sorted_points]
        
        print(f"DEBUG: Generated true SLD profile with {len(x)} points.")
        return np.array(x), np.array(y)

    def run_all_models(self):
        """Run inference for all configured models."""
        print("DEBUG: Starting to run inference for all configured models.")
        for model_name, model_config in self.model_configs.items():
            result = self.run_inference(model_config, model_name)
            self.results[model_name] = result
        print("DEBUG: Finished running all models.")

def main():
    """Main function to run the inference pipeline."""
    print("DEBUG: main() function started.")
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        print(f"DEBUG: Running with config file from command line: {config_file}")
        pipeline = InferencePipeline(config_file=config_file)
        pipeline.run_all_models()
        pipeline.save_results()
        pipeline.plot_results()
    else:
        print("DEBUG: No config file provided. Running a default example (s000000).")
        # Example of running in batch mode for a specific experiment
        experiment_id = 's000000'
        models = ['alpha', 'beta', 'gamma', 'delta'] # Example model names
        
        pipeline = InferencePipeline(
            experiment_id=experiment_id,
            models_list=models,
            data_directory="data",
            priors_type="broad", # or "narrow"
            output_dir="inference_results/s000000_test"
        )
        pipeline.run_all_models()
        pipeline.save_results()
        pipeline.plot_results()

if __name__ == "__main__":
    main()
