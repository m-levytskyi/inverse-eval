import pickle

data = pickle.load(open('manifest_exp_1L.pkl', 'rb'))

print('Keys:', list(data.keys()))
print('\nroot:', data['root'])
print('n:', data['n'])

samples = data['samples']
print('\nsamples type:', type(samples))
print('samples length:', len(samples))

# Find index 254
sample_254 = None
for s in samples:
    if s.get('base_id') == 's000254':
        sample_254 = s
        break

if sample_254:
    print('\n=== Found s000254 in manifest ===')
    print('Keys:', list(sample_254.keys()))
    
    mp = sample_254.get('model_preview', [])
    print('\nmodel_preview length:', len(mp))
    if len(mp) > 0:
        print('\nFirst few items in model_preview:')
        for i, item in enumerate(mp[:5]):
            print(f'{i}: {item}')
            
        # Check if there's parameter info
        for item in mp:
            if isinstance(item, dict) and ('param' in str(item).lower() or 'sld' in str(item).lower()):
                print('\nFound parameter-related item:')
                print(item)
                break
else:
    print('\ns000254 not found in manifest')
    print('Sample base_ids (first 10):', [s['base_id'] for s in samples[:10]])
