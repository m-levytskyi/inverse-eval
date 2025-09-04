#!/usr/bin/env python3
"""
Deep debug analysis to understand why all experiments are getting 40%+ MAPE
even with ±50% narrow priors on synthetic data.
"""

import numpy as np
import json
from pathlib import Path
import sys
import os

# Add debug folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'debug'))

# Import from debug folder
from debug.debug_synthetic_inference import debug_single_experiment
import inference_pipeline

def analyze_multiple_experiments(data_directory, layer_count, num_to_analyze=5):
    """Analyze multiple experiments to find patterns in the poor performance"""
    
    print(f"{'='*80}")
    print(f"ANALYZING MULTIPLE SYNTHETIC EXPERIMENTS")
    print(f"{'='*80}")
    
    # Find available experiments
    base_path = Path(data_directory)
    maria_dir = base_path / "MARIA_VIPR_dataset" / str(layer_count)
    
    if not maria_dir.exists():
        print(f"ERROR: MARIA directory not found: {maria_dir}")
        return
    
    # Find experiment files
    model_files = list(maria_dir.glob("synth_*_model.txt"))
    if not model_files:
        print(f"ERROR: No synthetic model files found in {maria_dir}")
        return
    
    # Extract experiment IDs
    exp_ids = []
    for model_file in model_files:
        exp_id = model_file.stem.replace("_model", "")
        exp_ids.append(exp_id)
    
    exp_ids = sorted(exp_ids)[:num_to_analyze]
    print(f"Analyzing {len(exp_ids)} experiments: {exp_ids}")
    
    # Collect detailed results for analysis
    analysis_results = []
    
    for i, exp_id in enumerate(exp_ids):
        print(f"\n{'='*20} EXPERIMENT {i+1}/{len(exp_ids)}: {exp_id} {'='*20}")
        
        # Load true parameters
        model_file = maria_dir / f"{exp_id}_model.txt"
        curve_file = maria_dir / f"{exp_id}_experimental_curve.dat"
        
        # Parse true parameters
        true_params = {}
        if model_file.exists():
            with open(model_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) >= 4:
                    layer_name = parts[0]
                    sld_str = parts[1]
                    thickness_str = parts[2]
                    roughness_str = parts[3]
                    
                    try:
                        if sld_str != 'none' and sld_str != 'inf':
                            true_params[f"{layer_name}_sld"] = float(sld_str)
                        if thickness_str != 'inf' and thickness_str != 'none':
                            true_params[f"{layer_name}_thickness"] = float(thickness_str)
                        if roughness_str != 'none' and roughness_str != 'inf':
                            true_params[f"{layer_name}_roughness"] = float(roughness_str)
                    except ValueError:
                        continue
        
        # Load experimental curve data
        curve_data = {}
        if curve_file.exists():
            exp_data = np.loadtxt(curve_file)
            if exp_data.shape[1] >= 3:
                q_exp = exp_data[:, 0]
                r_exp = exp_data[:, 1]
                sigma_exp = exp_data[:, 2]
                
                curve_data = {
                    'q_range': [float(np.min(q_exp)), float(np.max(q_exp))],
                    'r_range': [float(np.min(r_exp)), float(np.max(r_exp))],
                    'num_points': len(q_exp),
                    'negative_r_count': int(np.sum(r_exp < 0)),
                    'zero_r_count': int(np.sum(r_exp == 0))
                }
        
        print(f"True parameters: {true_params}")
        print(f"Curve data: {curve_data}")
        
        # Check if this experiment has the data quality issues
        if curve_data.get('negative_r_count', 0) > 0:
            print(f"⚠️  WARNING: {curve_data['negative_r_count']} negative reflectivity values!")
        
        if curve_data.get('num_points', 0) < 100:
            print(f"⚠️  WARNING: Only {curve_data.get('num_points', 0)} data points!")
        
        # Run a quick inference test
        try:
            print(f"Running inference for {exp_id}...")
            
            # Run inference but capture just the key metrics
            models = ["b_mc_point_neutron_conv_standard_L1_comp"] if layer_count == 1 else ["b_mc_point_neutron_conv_standard_L2_comp"]
            
            from inference_pipeline import InferencePipeline
            result = InferencePipeline.run_experiment_inference(
                experiment_id=exp_id,
                models_list=models,
                data_directory=data_directory,
                priors_type='narrow',
                output_dir='temp_debug',
                layer_count=layer_count
            )
            
            analysis_data = {
                'exp_id': exp_id,
                'true_params': true_params,
                'curve_data': curve_data,
                'inference_success': result.get('success', False)
            }
            
            if result.get('success', False):
                for model_name, model_result in result.get('models_results', {}).items():
                    if model_result.get('success', False):
                        param_metrics = model_result.get('parameter_metrics', {})
                        if param_metrics and 'overall' in param_metrics:
                            mape = param_metrics['overall'].get('mape', None)
                            analysis_data['mape'] = mape
                            print(f"  MAPE: {mape:.1f}%")
                            
                            # Get predicted parameters
                            if 'predicted_params' in model_result and 'param_names' in model_result:
                                predicted = model_result['predicted_params']
                                param_names = model_result['param_names']
                                analysis_data['predicted_params'] = dict(zip(param_names, predicted))
                                
                                # Calculate individual parameter errors
                                param_errors = {}
                                for name, pred_val in zip(param_names, predicted):
                                    # Match to true values
                                    if 'thickness' in name.lower() and 'l1' in name.lower():
                                        true_val = true_params.get('layer1_thickness')
                                    elif 'sld' in name.lower() and 'l1' in name.lower():
                                        true_val = true_params.get('layer1_sld')
                                        if true_val is not None:
                                            true_val = true_val * 1e6  # Convert to ×10⁻⁶ units
                                    elif 'sld' in name.lower() and ('sub' in name.lower() or 'backing' in name.lower()):
                                        true_val = true_params.get('backing_sld')
                                        if true_val is not None:
                                            true_val = true_val * 1e6
                                    elif 'roughness' in name.lower():
                                        if 'sub' in name.lower():
                                            true_val = true_params.get('backing_roughness') or true_params.get('layer1_roughness')
                                        elif 'l1' in name.lower() or 'ambient' in name.lower():
                                            true_val = true_params.get('fronting_roughness') or true_params.get('layer1_roughness')
                                        else:
                                            true_val = true_params.get('layer1_roughness')
                                    else:
                                        true_val = None
                                    
                                    if true_val is not None:
                                        error_percent = abs(pred_val - true_val) / abs(true_val) * 100
                                        param_errors[name] = {
                                            'true': true_val,
                                            'predicted': pred_val,
                                            'error_percent': error_percent
                                        }
                                        print(f"    {name}: True={true_val:.3f}, Pred={pred_val:.3f}, Error={error_percent:.1f}%")
                                
                                analysis_data['param_errors'] = param_errors
            
            analysis_results.append(analysis_data)
            
        except Exception as e:
            print(f"  ERROR during inference: {e}")
            analysis_results.append({
                'exp_id': exp_id,
                'true_params': true_params,
                'curve_data': curve_data,
                'error': str(e)
            })
        
        print("-" * 80)
    
    # Analyze patterns across all experiments
    print(f"\n{'='*80}")
    print("CROSS-EXPERIMENT ANALYSIS")
    print(f"{'='*80}")
    
    mape_values = [r.get('mape') for r in analysis_results if r.get('mape') is not None]
    negative_counts = [r['curve_data'].get('negative_r_count', 0) for r in analysis_results if 'curve_data' in r]
    
    if mape_values:
        print(f"MAPE Distribution:")
        print(f"  Min MAPE: {min(mape_values):.1f}%")
        print(f"  Max MAPE: {max(mape_values):.1f}%")
        print(f"  Mean MAPE: {np.mean(mape_values):.1f}%")
        print(f"  Median MAPE: {np.median(mape_values):.1f}%")
    
    if negative_counts:
        print(f"\nData Quality Issues:")
        print(f"  Experiments with negative values: {sum(1 for x in negative_counts if x > 0)}/{len(negative_counts)}")
        print(f"  Average negative values per experiment: {np.mean(negative_counts):.1f}")
    
    # Look for patterns in parameter errors
    print(f"\nParameter Error Patterns:")
    thickness_errors = []
    roughness_errors = []
    sld_errors = []
    
    for result in analysis_results:
        if 'param_errors' in result:
            for param_name, error_data in result['param_errors'].items():
                error_pct = error_data['error_percent']
                if 'thickness' in param_name.lower():
                    thickness_errors.append(error_pct)
                elif 'roughness' in param_name.lower():
                    roughness_errors.append(error_pct)
                elif 'sld' in param_name.lower():
                    sld_errors.append(error_pct)
    
    for param_type, errors in [('Thickness', thickness_errors), ('Roughness', roughness_errors), ('SLD', sld_errors)]:
        if errors:
            print(f"  {param_type} errors: min={min(errors):.1f}%, max={max(errors):.1f}%, mean={np.mean(errors):.1f}%")
    
    # Save detailed analysis
    analysis_file = Path('synthetic_data_analysis.json')
    with open(analysis_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    print(f"\nDetailed analysis saved to: {analysis_file}")
    
    return analysis_results

if __name__ == "__main__":
    # Set narrow priors to 50%
    inference_pipeline.NARROW_PRIORS_DEVIATION = 0.5
    
    # Analyze the most recent synthetic data
    data_dirs = [
        "synthetic_data/b_mc_point_neutron_conv_standard_L1_comp",
        "corrected_synthetic_data/b_mc_point_neutron_conv_standard_L1_comp"
    ]
    
    for data_dir in data_dirs:
        if Path(data_dir).exists():
            print(f"ANALYZING: {data_dir}")
            analyze_multiple_experiments(data_dir, 1, 3)
            print("\n" + "="*80 + "\n")
        else:
            print(f"Directory not found: {data_dir}")
