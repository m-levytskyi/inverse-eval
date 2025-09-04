#!/usr/bin/env python3
"""
Generate SLD profiles for specific experiments identified as edge cases or good cases
"""
import json
from pathlib import Path
from batch_inference_pipeline import BatchInferencePipeline
from inference_pipeline import InferencePipeline
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def run_specific_experiments(exp_ids, layer_count=1, priors_types=['broad', 'narrow']):
    """Run inference on specific experiments to generate SLD profiles."""
    models = {
        1: [
            "b_mc_point_neutron_conv_standard_L1_comp",
            "b_mc_point_neutron_conv_standard_L1_InputQDq",
            "b_mc_point_xray_conv_standard_L2"
        ],
        2: [
            "b_mc_point_neutron_conv_standard_L2_comp",
            "b_mc_point_neutron_conv_standard_L2_InputQDq", 
            "b_mc_point_xray_conv_standard_L2"
        ]
    }
    
    results = {}
    
    for exp_id in exp_ids:
        print(f"\nProcessing {exp_id}...")
        exp_results = {
            'experiment_id': exp_id,
            'layer_count': layer_count,
            'priors': {}
        }
        
        for priors_type in priors_types:
            print(f"  Running {priors_type} priors...")
            try:
                # Create inference pipeline instance
                pipeline = InferencePipeline(
                    experiment_id=exp_id,
                    models_list=models[layer_count],
                    data_directory="data",
                    priors_type=priors_type,
                    output_dir="inference_results",
                    layer_count=layer_count
                )
                
                # Load experimental data and true parameters
                pipeline.discover_experiment_files()
                pipeline.load_experimental_data_from_files()
                pipeline.load_true_parameters_from_files()
                pipeline.generate_model_configurations()
                
                # Run inference for all models (don't show plots to save time)
                pipeline.run_all_models(show_plots=False)
                
                # Extract results with SLD profiles
                models_results = {}
                best_model_name = None
                best_mape = float('inf')
                
                for model_name, model_result in pipeline.results.items():
                    if model_result['success']:
                        # Extract essential data including SLD profiles
                        models_results[model_name] = {
                            'success': True,
                            'parameter_metrics': model_result['parameter_metrics'],
                            'fit_metrics': model_result['fit_metrics'],
                            # Include SLD profiles for plotting, convert numpy arrays to lists
                            'sld_profile_x': model_result.get('sld_profile_x', []).tolist() if hasattr(model_result.get('sld_profile_x', []), 'tolist') else model_result.get('sld_profile_x', []),
                            'sld_profile_predicted': model_result.get('sld_profile_predicted', []).tolist() if hasattr(model_result.get('sld_profile_predicted', []), 'tolist') else model_result.get('sld_profile_predicted', []),
                            'sld_profile_polished': model_result.get('sld_profile_polished', []).tolist() if hasattr(model_result.get('sld_profile_polished', []), 'tolist') else model_result.get('sld_profile_polished', [])
                        }
                        
                        # Track best model (lowest overall MAPE)
                        if model_result['parameter_metrics'] and 'overall' in model_result['parameter_metrics']:
                            overall_mape = model_result['parameter_metrics']['overall']['mape']
                            if overall_mape < best_mape:
                                best_mape = overall_mape
                                best_model_name = model_name
                    else:
                        models_results[model_name] = {'success': False}
                
                exp_results['priors'][priors_type] = {
                    'success': len([r for r in models_results.values() if r.get('success', False)]) > 0,
                    'models_results': models_results,
                    'best_model_name': best_model_name,
                    'best_mape': best_mape
                }
                
            except Exception as e:
                print(f"    ✗ Error in {exp_id} ({priors_type}): {e}")
                exp_results['priors'][priors_type] = {
                    'success': False,
                    'error': str(e),
                    'models_results': {}
                }
        
        results[exp_id] = exp_results
    
    return results

def main():
    # Edge cases and good cases identified from the June 20th results
    edge_cases = ['s002156', 's008834', 's005700']  # Top 3 worst
    good_cases = ['s007854', 's001185', 's002485', 's004359', 's009393']  # Top 5 best
    
    print("Generating SLD profiles for edge cases and good cases...")
    print("=" * 60)
    
    # Run inference on edge cases
    print("Processing edge cases...")
    edge_results = run_specific_experiments(edge_cases, layer_count=1, priors_types=['broad'])
    
    # Run inference on good cases  
    print("\nProcessing good cases...")
    good_results = run_specific_experiments(good_cases, layer_count=1, priors_types=['narrow'])
    
    # Combine results
    all_results = {**edge_results, **good_results}
    
    # Create a summary structure compatible with our plotting script
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        'timestamp': timestamp,
        'layer_count': 1,
        'total_experiments': len(all_results),
        'all_results': all_results
    }
    
    # Save results
    output_file = f"batch_inference_results/edge_good_cases_summary_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print(f"You can now run:")
    print(f"python generate_edge_case_plots.py --json-file {output_file}")

if __name__ == "__main__":
    main()
