#!/usr/bin/env python3
"""
Parameter discovery utilities for reflectometry analysis.

This module contains functions to discover and parse true parameters from 
experimental data files and model files.
"""

import numpy as np
from pathlib import Path


def discover_experiment_files(experiment_id, data_directory, layer_count=None):
    """
    Discover experimental data and model files for a given experiment ID.
    
    Args:
        experiment_id: Experiment identifier (e.g., 's000000')
        data_directory: Base data directory to search
        layer_count: Number of layers (1 or 2). If None, searches both
        
    Returns:
        Tuple of (data_file_path, model_file_path, detected_layer_count) or (None, None, None) if not found
    """
    data_dir = Path(data_directory)
    exp_data_file = None
    exp_model_file = None
    detected_layer_count = None
    
    print(f"Searching for experiment files for {experiment_id}")
    print(f"Base directory: {data_dir}")
    if layer_count:
        print(f"Looking specifically for {layer_count}-layer data")
    
    # Define search patterns
    data_patterns = [
        f"{experiment_id}_experimental_curve.dat"
    ]
    
    model_patterns = [
        f"{experiment_id}_model.txt",
        f"{experiment_id}_model.dat"
    ]
    
    # Define layer directories to search
    layer_dirs_to_search = []
    if layer_count is not None:
        layer_dirs_to_search = [str(layer_count)]
    else:
        layer_dirs_to_search = ['1', '2']  # Search both if not specified
    
    # Search in subdirectories recursively
    def search_directory(directory, depth=0):
        nonlocal exp_data_file, exp_model_file, detected_layer_count
        
        if depth > 3:  # Prevent infinite recursion
            return
            
        if not directory.is_dir():
            return
            
        print(f"Searching in: {directory}")
        
        # Check if this is a layer directory (contains '1' or '2')
        dir_name = directory.name
        current_layer_count = None
        
        # Check if directory name indicates layer count
        if dir_name in layer_dirs_to_search:
            current_layer_count = int(dir_name)
            print(f"Found layer directory: {directory} (layer count: {current_layer_count})")
        
        # Look for data files in current directory
        for pattern in data_patterns:
            data_file = directory / pattern
            if data_file.exists():
                print(f"Found experimental data: {data_file}")
                exp_data_file = data_file
                if current_layer_count:
                    detected_layer_count = current_layer_count
                break
        
        # Look for model files in current directory
        for pattern in model_patterns:
            model_file = directory / pattern
            if model_file.exists():
                print(f"Found model file: {model_file}")
                exp_model_file = model_file
                if current_layer_count and detected_layer_count is None:
                    detected_layer_count = current_layer_count
                break
        
        # If both files found, stop searching
        if exp_data_file and exp_model_file:
            return
        
        # Recursively search subdirectories
        try:
            for subdir in directory.iterdir():
                if subdir.is_dir():
                    search_directory(subdir, depth + 1)
                    if exp_data_file and exp_model_file:
                        return
        except PermissionError:
            print(f"Permission denied accessing: {directory}")
    
    # Start search from data directory
    search_directory(data_dir)
    
    if not exp_data_file:
        print(f"ERROR: Experimental data file for {experiment_id} not found")
        print(f"Searched patterns: {data_patterns}")
        print(f"In layer directories: {layer_dirs_to_search}")
    if not exp_model_file:
        print(f"ERROR: Model file for {experiment_id} not found")
        print(f"Searched patterns: {model_patterns}")
        print(f"In layer directories: {layer_dirs_to_search}")
    
    if detected_layer_count:
        print(f"Detected layer count: {detected_layer_count}")
    
    return exp_data_file, exp_model_file, detected_layer_count


def parse_true_parameters_from_model_file(model_file_path):
    """
    Parse true parameters from a Motofit-style model file.
    
    This function parses tabular model files where each row represents a layer:
    - fronting: ambient medium (usually air)
    - layer1, layer2, etc: material layers  
    - backing: substrate
    
    Columns are: layer_name, sld(A^-2), thickness(A), roughness(A)
    
    Args:
        model_file_path: Path to model file
        
    Returns:
        Dictionary with parsed parameters for different layer interpretations
    """
    print(f"Parsing true parameters from: {model_file_path}")
    
    if not Path(model_file_path).exists():
        print(f"ERROR: Model file does not exist: {model_file_path}")
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
                print(f"Parsed layer '{layer_name}': sld={sld}, thickness={thickness}, roughness={roughness}")
                
            except (ValueError, IndexError) as e:
                print(f"Failed to parse line '{line}': {e}")
                continue
    
    print(f"Parsed layers: {list(layers.keys())}")
    
    true_params_dict = {}
    
    # Count actual material layers (exclude fronting and backing)
    material_layers = [name for name in layers.keys() if name.startswith('layer')]
    num_material_layers = len(material_layers)
    print(f"Found {num_material_layers} material layers: {material_layers}")
    
    # Parse as 1-layer model (if we have 1 material layer)
    if num_material_layers == 1 and 'layer1' in layers:
        try:
            print("Parsing as 1-layer model")
            
            fronting = layers.get('fronting', {})
            layer1 = layers.get('layer1', {})
            backing = layers.get('backing', {})
            
            # 1-layer parameters: [thickness, amb_rough, sub_rough, layer_sld, sub_sld]
            thickness = layer1.get('thickness', 0.0)
            amb_rough = fronting.get('roughness', 0.0)  # fronting roughness
            sub_rough = layer1.get('roughness', 0.0)   # layer1 roughness
            layer_sld = layer1.get('sld', 0.0) * 1e6   # Convert to 10^-6 units
            sub_sld = backing.get('sld', 0.0) * 1e6    # Convert to 10^-6 units
            
            params_1_layer = [thickness, amb_rough, sub_rough, layer_sld, sub_sld]
            names_1_layer = get_parameter_names_for_layer_count(1)
            
            true_params_dict['1_layer'] = {
                'params': params_1_layer,
                'param_names': names_1_layer
            }
            print(f"1-layer parameters: {params_1_layer}")
            print(f"SLD values converted to 10^-6 units - layer_sld: {layer_sld:.2f}, sub_sld: {sub_sld:.2f}")
            
        except Exception as e:
            print(f"Failed to parse as 1-layer model: {e}")
    
    # Parse as 2-layer model (if we have 2 material layers)
    if num_material_layers == 2 and 'layer1' in layers and 'layer2' in layers:
        try:
            print("Parsing as 2-layer model")
            
            fronting = layers.get('fronting', {})
            layer1 = layers.get('layer1', {})
            layer2 = layers.get('layer2', {})
            backing = layers.get('backing', {})
            
            # 2-layer parameters: [thickness1, thickness2, amb_rough, int_rough, sub_rough, 
            #                      layer1_sld, layer2_sld, sub_sld]
            thickness1 = layer1.get('thickness', 0.0)
            thickness2 = layer2.get('thickness', 0.0)
            amb_rough = fronting.get('roughness', 0.0)  # fronting roughness
            int_rough = layer1.get('roughness', 0.0)   # interface between layer1 and layer2
            sub_rough = layer2.get('roughness', 0.0)   # interface between layer2 and substrate
            layer1_sld = layer1.get('sld', 0.0) * 1e6  # Convert to 10^-6 units
            layer2_sld = layer2.get('sld', 0.0) * 1e6  # Convert to 10^-6 units
            sub_sld = backing.get('sld', 0.0) * 1e6    # Convert to 10^-6 units
            
            params_2_layer = [thickness1, thickness2, amb_rough, int_rough, sub_rough,
                             layer1_sld, layer2_sld, sub_sld]
            names_2_layer = get_parameter_names_for_layer_count(2)
            
            true_params_dict['2_layer'] = {
                'params': params_2_layer,
                'param_names': names_2_layer
            }
            print(f"2-layer parameters: {params_2_layer}")
            print(f"SLD values converted to 10^-6 units - layer1_sld: {layer1_sld:.2f}, layer2_sld: {layer2_sld:.2f}, sub_sld: {sub_sld:.2f}")
            
        except Exception as e:
            print(f"Failed to parse as 2-layer model: {e}")
    
    return true_params_dict


def get_parameter_names_for_layer_count(layer_count):
    """
    Get parameter names for a given layer count.
    
    Args:
        layer_count: Number of layers (1 or 2)
        
    Returns:
        List of parameter names
    """
    if layer_count == 1:
        return ['thickness', 'amb_rough', 'sub_rough', 'layer_sld', 'sub_sld']
    elif layer_count == 2:
        return ['thickness1', 'thickness2', 'amb_rough', 'int_rough', 'sub_rough',
                'layer1_sld', 'layer2_sld', 'sub_sld']
    else:
        raise ValueError(f"Unsupported layer count: {layer_count}")


def generate_true_sld_profile(true_params_dict, x_range=(0, 1000), n_points=1000):
    """
    Generate a true SLD profile from parsed parameters.
    
    Args:
        true_params_dict: Dictionary with parsed true parameters
        x_range: Tuple of (min, max) depth values in Angstroms
        n_points: Number of points in the profile
        
    Returns:
        Tuple of (x_axis, sld_profile) or (None, None) if no valid data
    """
    print("Generating true SLD profile")
    
    if not true_params_dict:
        print("No true parameters available")
        return None, None
    
    # Prefer 2-layer interpretation if available
    layer_key = '2_layer' if '2_layer' in true_params_dict else '1_layer'
    if layer_key not in true_params_dict:
        print("No valid layer interpretation found")
        return None, None
    
    params = true_params_dict[layer_key]['params']
    param_names = true_params_dict[layer_key]['param_names']
    
    print(f"Using {layer_key} interpretation")
    print(f"Parameters: {dict(zip(param_names, params))}")
    
    x_axis = np.linspace(x_range[0], x_range[1], n_points)
    sld_profile = np.zeros_like(x_axis)
    
    if layer_key == '1_layer':
        # 1-layer: [thickness, amb_rough, sub_rough, layer_sld, sub_sld]
        thickness, amb_rough, sub_rough, layer_sld, sub_sld = params
        
        # Simple step function model
        layer_mask = (x_axis >= 0) & (x_axis <= thickness)
        substrate_mask = x_axis > thickness
        
        sld_profile[layer_mask] = layer_sld
        sld_profile[substrate_mask] = sub_sld
        
    elif layer_key == '2_layer':
        # 2-layer: [thickness1, thickness2, amb_rough, int_rough, sub_rough,
        #           layer1_sld, layer2_sld, sub_sld]
        thickness1, thickness2, amb_rough, int_rough, sub_rough, layer1_sld, layer2_sld, sub_sld = params
        
        # Simple step function model
        layer1_mask = (x_axis >= 0) & (x_axis <= thickness1)
        layer2_mask = (x_axis > thickness1) & (x_axis <= thickness1 + thickness2)
        substrate_mask = x_axis > thickness1 + thickness2
        
        sld_profile[layer1_mask] = layer1_sld
        sld_profile[layer2_mask] = layer2_sld
        sld_profile[substrate_mask] = sub_sld
    
    print(f"Generated SLD profile with {n_points} points over range {x_range}")
    return x_axis, sld_profile


def auto_detect_layer_count(model_file_path):
    """
    Automatically detect the number of layers from a model file.
    
    Args:
        model_file_path: Path to model file
        
    Returns:
        Number of layers (1 or 2) or None if cannot determine
    """
    true_params = parse_true_parameters_from_model_file(model_file_path)
    
    if '2_layer' in true_params:
        print("Auto-detected: 2 layers")
        return 2
    elif '1_layer' in true_params:
        print("Auto-detected: 1 layer")
        return 1
    else:
        print("Could not auto-detect layer count")
        return None


def get_prior_bounds_for_experiment(experiment_id, true_params_dict=None, 
                                   priors_type="broad", deviation=0.5, layer_count=1):
    """
    Generate appropriate prior bounds for an experiment.
    
    Args:
        experiment_id: Experiment identifier
        true_params_dict: Dictionary with true parameters (for narrow priors)
        priors_type: "broad" or "narrow"
        deviation: Relative deviation for narrow priors (e.g., 0.5 for 50%)
        layer_count: Number of layers (1 or 2)
        
    Returns:
        List of (min, max) tuples for prior bounds
    """
    print(f"Generating {priors_type} prior bounds for {experiment_id} ({layer_count} layer{'s' if layer_count > 1 else ''})")
    
    if priors_type == "narrow" and true_params_dict:
        # Use true parameters to generate narrow priors
        layer_key = f'{layer_count}_layer'
        if layer_key in true_params_dict:
            true_params = true_params_dict[layer_key]['params']
            bounds = []
            
            for param in true_params:
                if param > 0:
                    min_val = param * (1 - deviation)
                    max_val = param * (1 + deviation)
                    bounds.append((min_val, max_val))
                else:
                    # Handle zero or negative parameters
                    bounds.append((-abs(param) - 1, abs(param) + 1))
            
            print(f"Generated narrow priors with {deviation*100}% deviation")
            return bounds
    
    # Default broad priors for common structures
    if layer_count == 1:
        broad_priors = [
            (50.0, 500.0),     # layer thickness (Å)
            (1.0, 15.0),       # ambient roughness (Å)
            (1.0, 50.0),       # substrate roughness (Å)
            (-5.0, 20.0),      # layer SLD (×10^-6 Å^-2)
            (0.0, 25.0)        # substrate SLD (×10^-6 Å^-2)
        ]
    elif layer_count == 2:
        broad_priors = [
            (20.0, 300.0),     # layer1 thickness (Å)
            (20.0, 300.0),     # layer2 thickness (Å)
            (1.0, 15.0),       # ambient roughness (Å)
            (1.0, 30.0),       # interface roughness (Å)
            (1.0, 50.0),       # substrate roughness (Å)
            (-5.0, 20.0),      # layer1 SLD (×10^-6 Å^-2)
            (-5.0, 20.0),      # layer2 SLD (×10^-6 Å^-2)
            (0.0, 25.0)        # substrate SLD (×10^-6 Å^-2)
        ]
    else:
        raise ValueError(f"Unsupported layer count: {layer_count}")
    
    print(f"Using default broad {layer_count}-layer priors")
    return broad_priors
