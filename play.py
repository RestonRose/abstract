from collect_raw_data import load_json
import numpy as np


# Load your data from JSON files
input_grids = load_json('binary_data/training_input_grids.json')
test_inputs = load_json('binary_data/training_test_input_grids.json')
a = np.array(test_inputs)

for i,instance in enumerate(input_grids):
    try:
        a = np.array(instance)
    except:
        print(f'Instance: {i}')
        print(len(instance))
        print(instance)
        break