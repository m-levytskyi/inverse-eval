import pickle
import json
import numpy as np

# Load all data
print("Loading data...")
manifest_data = pickle.load(open('manifest_exp_1L.pkl', 'rb'))
manifest_samples = manifest_data['samples']

pkl_data = pickle.load(open('results_exp_1L_fitconstraints0_width0.3_simple.pkl', 'rb'))
pkl_targets = pkl_data[0]
pkl_predictions = pkl_data[1]
pkl_indices = pkl_data[2]
pkl_bounds = pkl_data[5]

json_data = json.load(open('batch_results_143.json', 'r'))

# Parameter names
param_names = [
    'sld_fronting',
    'roughness_fronting',
    'sld_1',
    'thickness_1',
    'roughness_1',
    'sld_backing'
]

param_units = [
    'Å⁻² × 10⁻⁶',
    'Å',
    'Å⁻² × 10⁻⁶',
    'Å',
    'Å',
    'Å⁻² × 10⁻⁶'
]

# Build combined results
combined_results = {}

print(f"Processing {len(pkl_indices)} experiments...")

for pkl_pos in range(len(pkl_indices)):
    manifest_idx = pkl_indices[pkl_pos]
    
    if manifest_idx >= len(manifest_samples):
        continue
    
    # Get experiment ID
    experiment_id = manifest_samples[manifest_idx]['base_id']
    
    # Get true values and pickle predictions
    true_vals = pkl_targets[pkl_pos].tolist()
    pkl_pred = pkl_predictions[pkl_pos].tolist()
    pkl_out_of_bounds = bool(pkl_bounds[pkl_pos] == 1)
    
    # Get JSON predictions if available
    json_pred = None
    json_success = False
    
    if experiment_id in json_data:
        json_result = json_data[experiment_id]
        json_success = json_result.get('success', False)
        
        if json_success and 'prediction_dict' in json_result:
            pred_dict = json_result['prediction_dict']
            if 'predicted_params_array' in pred_dict:
                json_pred_vals = pred_dict['predicted_params_array']
                
                # Map JSON params to our parameter order
                # JSON order: Thickness L1, Roughness L1, Roughness sub, SLD L1, SLD sub
                json_pred = [
                    None,  # sld_fronting - not predicted by JSON
                    json_pred_vals[2] if len(json_pred_vals) > 2 else None,  # roughness_fronting (sub roughness)
                    json_pred_vals[3] if len(json_pred_vals) > 3 else None,  # sld_1
                    json_pred_vals[0] if len(json_pred_vals) > 0 else None,  # thickness_1
                    json_pred_vals[1] if len(json_pred_vals) > 1 else None,  # roughness_1
                    json_pred_vals[4] if len(json_pred_vals) > 4 else None,  # sld_backing (sub SLD)
                ]
    
    # Build result entry
    combined_results[experiment_id] = {
        'experiment_id': experiment_id,
        'manifest_index': int(manifest_idx),
        'pickle_position': int(pkl_pos),
        'parameter_names': param_names,
        'parameter_units': param_units,
        'true_values': true_vals,
        'pickle_predictions': pkl_pred,
        'pickle_out_of_bounds': pkl_out_of_bounds,
        'json_predictions': json_pred,
        'json_success': json_success
    }
    
    if (pkl_pos + 1) % 500 == 0:
        print(f"  Processed {pkl_pos + 1}/{len(pkl_indices)} experiments...")

print(f"\nTotal experiments: {len(combined_results)}")
print(f"JSON successful: {sum(1 for v in combined_results.values() if v['json_success'])}")
print(f"PKL out of bounds: {sum(1 for v in combined_results.values() if v['pickle_out_of_bounds'])}")

# Save to JSON file
output_file = 'combined_predictions.json'
print(f"\nSaving to {output_file}...")

with open(output_file, 'w') as f:
    json.dump(combined_results, f, indent=2)

print(f"Done! Saved {len(combined_results)} experiments to {output_file}")

# Show example
print("\nExample entry (first experiment):")
first_id = list(combined_results.keys())[0]
import pprint
pprint.pprint(combined_results[first_id], width=100)
