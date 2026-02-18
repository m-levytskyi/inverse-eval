import pickle
import numpy as np

data = pickle.load(open('results_exp_1L_fitconstraints0_width0.3_simple.pkl', 'rb'))
targets = data[0]
indices = data[2]

print('Target parameters array - statistics for each column:\n')
print('Column | Min          | Max          | Mean         | Std          | Parameter Name')
print('-'*100)

param_names = [
    'sld_fronting (Å⁻² × 10⁻⁶)',
    'roughness_fronting (Å)',
    'sld_1 (Å⁻² × 10⁻⁶)',
    'thickness_1 (Å)',
    'roughness_1 (Å)',
    'sld_backing (Å⁻² × 10⁻⁶)'
]

for i in range(6):
    print(f'Col {i}  | {np.min(targets[:,i]):12.6e} | {np.max(targets[:,i]):12.6e} | {np.mean(targets[:,i]):12.6e} | {np.std(targets[:,i]):12.6e} | {param_names[i]}')

print('\n\nFirst 10 samples with their indices and parameter values:\n')
print('Index | sld_front    | rough_front  | sld_1        | thickness_1  | rough_1      | sld_back')
print('-'*105)

for i in range(10):
    print(f'{indices[i]:5d} | {targets[i,0]:12.6e} | {targets[i,1]:12.6f} | {targets[i,2]:12.6e} | {targets[i,3]:12.6f} | {targets[i,4]:12.6f} | {targets[i,5]:12.6e}')
