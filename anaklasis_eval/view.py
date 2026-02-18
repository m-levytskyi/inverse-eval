import pickle
import pprint # Optional, for a more readable output
import numpy as np

file_path = 'results_exp_1L_fitconstraints0_width0.3_simple.pkl'

# Open the file in binary mode ('rb' for read binary) and load the data
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Now 'data' contains the deserialized Python object
print("Data structure:")
print(f"Number of elements: {len(data)}")
print(f"Type: {type(data)}")
print()

# Check if data has parameter names
if hasattr(data, 'keys'):
    print("Keys:", data.keys())
elif isinstance(data, tuple):
    print("Tuple with elements:")
    for i, item in enumerate(data):
        if hasattr(item, 'shape'):
            print(f"  [{i}] array, shape: {item.shape}, dtype: {item.dtype}")
        else:
            print(f"  [{i}] {type(item).__name__}: {item}")

print("\n" + "="*60)
print("Target parameters (T) - mean values:")
targets = data[0]
for i in range(targets.shape[1]):
    print(f"  Parameter {i}: mean = {np.mean(targets[:, i]):.6e}, std = {np.std(targets[:, i]):.6e}")

print("\nPredictions (O) - mean values:")
predictions = data[1]
for i in range(predictions.shape[1]):
    print(f"  Parameter {i}: mean = {np.mean(predictions[:, i]):.6e}, std = {np.std(predictions[:, i]):.6e}")
