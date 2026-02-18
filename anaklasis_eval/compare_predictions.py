import pickle
import json
import numpy as np

# Load all data
manifest_data = pickle.load(open('manifest_exp_1L.pkl', 'rb'))
manifest_samples = manifest_data['samples']

pkl_data = pickle.load(open('results_exp_1L_fitconstraints0_width0.3_simple.pkl', 'rb'))
pkl_targets = pkl_data[0]
pkl_predictions = pkl_data[1]
pkl_indices = pkl_data[2]

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

def show_experiment(base_id):
    """Display true values and predictions for an experiment by base_id (e.g., 's000798')"""
    
    # Find index in manifest
    manifest_idx = None
    for i, sample in enumerate(manifest_samples):
        if sample['base_id'] == base_id:
            manifest_idx = i
            break
    
    if manifest_idx is None:
        print(f"Error: {base_id} not found in manifest")
        return
    
    # Find position in pickle data
    pkl_position = np.where(pkl_indices == manifest_idx)[0]
    
    if len(pkl_position) == 0:
        print(f"Error: manifest index {manifest_idx} not found in pickle indices")
        return
    
    pkl_pos = pkl_position[0]
    
    # Get values
    true_vals = pkl_targets[pkl_pos]
    pkl_pred = pkl_predictions[pkl_pos]
    pkl_bounds = pkl_data[5][pkl_pos]  # Array B - out of bounds indicator
    
    # Get JSON prediction if available
    json_pred_vals = None
    json_success = False
    if base_id in json_data:
        json_result = json_data[base_id]
        json_success = json_result.get('success', False)
        if json_success and 'prediction_dict' in json_result:
            pred_dict = json_result['prediction_dict']
            if 'predicted_params_array' in pred_dict:
                json_pred_vals = pred_dict['predicted_params_array']
    
    # Display
    print(f"\n{'='*90}")
    print(f"Experiment: {base_id} (manifest index: {manifest_idx}, pickle position: {pkl_pos})")
    print(f"PKL out of bounds: {'YES' if pkl_bounds == 1 else 'NO'}")
    print(f"JSON success: {'YES' if json_success else 'NO'}")
    print(f"{'='*90}\n")
    
    # Map JSON params (different order) to our params
    json_map = {}
    if json_pred_vals and len(json_pred_vals) >= 5:
        # JSON order: Thickness L1, Roughness L1, Roughness sub, SLD L1, SLD sub, r_scale, log10_background
        json_map = {
            'thickness_1': json_pred_vals[0],
            'roughness_1': json_pred_vals[1],
            'roughness_fronting': json_pred_vals[2],  # sub roughness
            'sld_1': json_pred_vals[3],
            'sld_backing': json_pred_vals[4],  # sub SLD
        }
    
    print(f"{'Parameter':<20} {'Unit':<15} {'True Value':<15} {'PKL Pred':<15} {'JSON Pred':<15}")
    print('-'*90)
    
    for i, (name, unit) in enumerate(zip(param_names, param_units)):
        true_val = true_vals[i]
        pkl_val = pkl_pred[i]
        json_val = json_map.get(name, 'N/A')
        
        if json_val == 'N/A':
            json_str = 'N/A'
        else:
            json_str = f"{json_val:14.6e}"
        
        print(f"{name:<20} {unit:<15} {true_val:14.6e} {pkl_val:14.6e} {json_str:<15}")
    
    print()

# Example usage
if __name__ == "__main__":
    # Show first experiment from pickle
    first_idx = pkl_indices[0]
    first_base_id = manifest_samples[first_idx]['base_id']
    
    print(f"Example: Showing first experiment in pickle file")
    show_experiment(first_base_id)
    
    # You can call with any base_id:
    # show_experiment('s000798')
    # show_experiment('s009977')
