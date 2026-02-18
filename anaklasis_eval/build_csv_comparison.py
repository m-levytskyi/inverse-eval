import json
import csv
import numpy as np

# Load combined data
print("Loading combined predictions...")
with open('combined_predictions.json', 'r') as f:
    data = json.load(f)

# Parameter names for headers
param_names = [
    'SLD Fronting',
    'Roughness Fronting',
    'SLD Layer 1',
    'Thickness Layer 1',
    'Roughness Layer 1',
    'SLD Backing'
]

# Build CSV headers
headers = ['Experiment ID']

for param in param_names:
    headers.extend([
        f'{param} - True',
        f'{param} - PKL Pred',
        f'{param} - JSON Pred',
        f'{param} - PKL Diff %',
        f'{param} - JSON Diff %',
        f'{param} - PKL vs JSON Diff %'
    ])

# Add status columns
headers.extend(['PKL Out of Bounds', 'JSON Success'])

# Write CSV
output_file = 'predictions_comparison.csv'
print(f"Writing to {output_file}...")

with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(headers)
    
    count = 0
    for exp_id, exp_data in sorted(data.items()):
        row = [exp_id]
        
        true_vals = exp_data['true_values']
        pkl_preds = exp_data['pickle_predictions']
        json_preds = exp_data['json_predictions']
        
        for i in range(6):
            true_val = true_vals[i]
            pkl_pred = pkl_preds[i]
            json_pred = json_preds[i] if json_preds else None
            
            # True value
            row.append(true_val)
            
            # PKL prediction
            row.append(pkl_pred)
            
            # JSON prediction
            row.append(json_pred if json_pred is not None else '')
            
            # PKL difference %
            if true_val != 0:
                pkl_diff = ((pkl_pred - true_val) / abs(true_val)) * 100
                row.append(f'{pkl_diff:.4f}')
            else:
                row.append('N/A')
            
            # JSON difference %
            if json_pred is not None and true_val != 0:
                json_diff = ((json_pred - true_val) / abs(true_val)) * 100
                row.append(f'{json_diff:.4f}')
            else:
                row.append('N/A')
            
            # PKL vs JSON difference %
            if json_pred is not None and pkl_pred != 0:
                pkl_json_diff = ((json_pred - pkl_pred) / abs(pkl_pred)) * 100
                row.append(f'{pkl_json_diff:.4f}')
            else:
                row.append('N/A')
        
        # Status columns
        row.append('YES' if exp_data['pickle_out_of_bounds'] else 'NO')
        row.append('YES' if exp_data['json_success'] else 'NO')
        
        writer.writerow(row)
        
        count += 1
        if count % 500 == 0:
            print(f"  Processed {count}/{len(data)} experiments...")

print(f"\nDone! Saved {count} experiments to {output_file}")
print(f"\nColumns per parameter:")
print("  - True value")
print("  - PKL prediction")
print("  - JSON prediction")
print("  - PKL difference % (from true)")
print("  - JSON difference % (from true)")
print("  - PKL vs JSON difference %")
