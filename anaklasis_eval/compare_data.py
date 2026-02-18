import pickle
import json
import numpy as np

# Load pickle data
with open('results_exp_1L_fitconstraints0_width0.3_simple.pkl', 'rb') as f:
    pkl_data = pickle.load(f)

targets = pkl_data[0]
predictions = pkl_data[1]
indices = pkl_data[2]

# Load JSON data
print("Loading JSON file (this may take a moment)...")
with open('batch_results_143.json', 'r') as f:
    json_data = json.load(f)

print(f"JSON loaded. Number of experiments: {len(json_data)}")

# Test the mapping: pickle index -> s00XXXX format
test_indices = [254, 1234, 75, 3162, indices[0], indices[1], indices[2]]

print("\nTesting mapping: pickle_index -> s00XXXX")
print("="*80)

for test_idx in test_indices:
    # Find position in pickle data
    pkl_positions = np.where(indices == test_idx)[0]
    
    if len(pkl_positions) == 0:
        print(f"\nIndex {test_idx} NOT FOUND in pickle indices")
        continue
        
    pos = pkl_positions[0]
    pkl_targets = targets[pos]
    
    # Try the s00XXXX format
    sample_key = f"s{test_idx:06d}"
    
    print(f"\n--- Pickle index {test_idx} (position {pos}) -> {sample_key} ---")
    
    if sample_key in json_data:
        json_sample = json_data[sample_key]
        
        if json_sample.get('success') and 'true_params_dict' in json_sample:
            layer_data = json_sample['true_params_dict']['1_layer']
            param_names = layer_data['param_names']
            param_values = layer_data['params']
            
            print(f"Pickle (6 params): {pkl_targets}")
            print(f"JSON   (5 params): {param_values}")
            print(f"JSON param names:  {param_names}")
            
            # Try to find which pickle columns match JSON
            print("\nMatching analysis:")
            for i, (name, json_val) in enumerate(zip(param_names, param_values)):
                # Find closest match in pickle
                diffs = [abs(pkl_targets[j] - json_val) for j in range(6)]
                best_match_idx = np.argmin(diffs)
                best_diff = diffs[best_match_idx]
                
                if best_diff < 0.01:  # Good match
                    print(f"  {name:15s} = {json_val:12.6f} -> PKL[{best_match_idx}] = {pkl_targets[best_match_idx]:12.6f} ✓")
                else:
                    print(f"  {name:15s} = {json_val:12.6f} -> PKL[{best_match_idx}] = {pkl_targets[best_match_idx]:12.6f} (diff: {best_diff:.2e})")
        else:
            print(f"  Sample not successful or missing params")
    else:
        print(f"  Key {sample_key} NOT FOUND in JSON")

# Determine parameter mapping from first successful match
print("\n" + "="*80)
print("Summary: Determining parameter order in pickle file")
print("="*80)

# Find a successful match
for test_idx in indices[:20]:
    sample_key = f"s{test_idx:06d}"
    if sample_key in json_data and json_data[sample_key].get('success'):
        pos = np.where(indices == test_idx)[0][0]
        pkl_targets = targets[pos]
        
        layer_data = json_data[sample_key]['true_params_dict']['1_layer']
        param_names = layer_data['param_names']
        param_values = layer_data['params']
        
        print(f"\nUsing sample {sample_key} for mapping:")
        print(f"Pickle: {pkl_targets}")
        print(f"JSON:   {param_values}")
        
        # Create mapping
        mapping = {}
        for name, json_val in zip(param_names, param_values):
            for pkl_idx in range(6):
                if abs(pkl_targets[pkl_idx] - json_val) < 0.01:
                    mapping[pkl_idx] = name
                    break
        
        print(f"\nParameter mapping (pickle column -> name):")
        for pkl_idx in range(6):
            if pkl_idx in mapping:
                print(f"  Column {pkl_idx}: {mapping[pkl_idx]}")
            else:
                print(f"  Column {pkl_idx}: UNKNOWN (extra parameter)")
        
        break
