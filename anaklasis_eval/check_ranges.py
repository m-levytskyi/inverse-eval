import json
import numpy as np

data = json.load(open('batch_results_143.json'))
successful = [(k,v) for k,v in data.items() if v.get('success')]

print(f'Successful: {len(successful)}/{len(data)}')

samples = [v['true_params_dict']['1_layer']['params'] for k,v in successful[:100]]
samples_arr = np.array(samples)

print('\nJSON parameter ranges (first 100 successful):')
names = successful[0][1]['true_params_dict']['1_layer']['param_names']

for i, name in enumerate(names):
    print(f'{name:15s}: min={samples_arr[:,i].min():10.4f}, max={samples_arr[:,i].max():10.4f}, mean={samples_arr[:,i].mean():10.4f}')

print('\n\nPickle parameter ranges:')
import pickle
pkl_data = pickle.load(open('results_exp_1L_fitconstraints0_width0.3_simple.pkl', 'rb'))
targets = pkl_data[0]

param_labels = ['Param 0', 'Param 1', 'Param 2', 'Param 3', 'Param 4', 'Param 5']
for i in range(6):
    print(f'{param_labels[i]:15s}: min={targets[:,i].min():10.4e}, max={targets[:,i].max():10.4e}, mean={targets[:,i].mean():10.4e}')
