import pickle
import pprint

data = pickle.load(open('manifest_exp_1L.pkl', 'rb'))

print('Type:', type(data))

if isinstance(data, dict):
    print('Dict keys:', list(data.keys()))
    first_key = list(data.keys())[0]
    print(f'\nFirst key: {first_key}')
    print(f'Value type: {type(data[first_key])}')
    
    if isinstance(data[first_key], list):
        print(f'List length: {len(data[first_key])}')
        print('\nFirst item in list:')
        item = data[first_key][0]
        if hasattr(item, 'keys'):
            print('Item keys:', list(item.keys()))
            print(f"base_id: {item.get('base_id')}")
            
            # Check for model_preview
            if 'model_preview' in item:
                model_prev = item['model_preview']
                print(f'\nmodel_preview type: {type(model_prev)}')
                if isinstance(model_prev, list) and len(model_prev) > 0:
                    print(f'model_preview length: {len(model_prev)}')
                    print('\nFirst element of model_preview:')
                    pprint.pprint(model_prev[0], depth=2)
elif isinstance(data, list):
    print(f'List length: {len(data)}')
    print('\nFirst item:')
    pprint.pprint(data[0], depth=2)

