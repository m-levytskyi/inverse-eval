import pickle
import json

# Load manifest
manifest_data = pickle.load(open('manifest_exp_1L.pkl', 'rb'))
manifest_samples = manifest_data['samples']

print(f'Manifest samples: {len(manifest_samples)}')
print(f'First 10 manifest base_ids: {[s["base_id"] for s in manifest_samples[:10]]}')

# Load JSON
print('\nLoading JSON...')
json_data = json.load(open('batch_results_143.json', 'r'))
print(f'JSON experiments: {len(json_data)}')
print(f'First 10 JSON keys: {list(json_data.keys())[:10]}')

# Check overlap
manifest_ids = set(s['base_id'] for s in manifest_samples)
json_ids = set(json_data.keys())

overlap = manifest_ids & json_ids
print(f'\nOverlap: {len(overlap)} experiments')
print(f'Only in manifest: {len(manifest_ids - json_ids)}')
print(f'Only in JSON: {len(json_ids - manifest_ids)}')

if len(overlap) > 0:
    print(f'\nFirst 10 overlapping IDs: {list(overlap)[:10]}')

# Check the pickle indices
pkl_data = pickle.load(open('results_exp_1L_fitconstraints0_width0.3_simple.pkl', 'rb'))
pkl_indices = pkl_data[2]

print(f'\n\nPickle indices range: {pkl_indices.min()} to {pkl_indices.max()}')
print(f'First 10 pickle indices: {pkl_indices[:10].tolist()}')

# Try to map pickle indices to manifest samples
print('\n\nMapping pickle indices to manifest:')
print('Pickle idx | Manifest base_id')
print('-'*40)

for i in range(min(10, len(pkl_indices))):
    pkl_idx = pkl_indices[i]
    # Pickle index should correspond to position in manifest
    if pkl_idx < len(manifest_samples):
        manifest_sample = manifest_samples[pkl_idx]
        base_id = manifest_sample['base_id']
        in_json = 'YES' if base_id in json_data else 'NO'
        print(f'{pkl_idx:10d} | {base_id:16s} | In JSON: {in_json}')
    else:
        print(f'{pkl_idx:10d} | OUT OF RANGE')
