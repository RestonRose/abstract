from data_pipeline import load_json, save_to_json
import json

# load the raw data
training_input_grids = load_json('raw_data/training_input_grids.json')
training_output_grids = load_json('raw_data/training_output_grids.json')
training_test_input_grids = load_json('raw_data/training_test_input_grids.json')
training_test_output_grids = load_json('raw_data/training_test_output_grids.json')

def convert_to_binary(data):
    """
    Recursively converts every number in the multi-dimensional list `data`
    to 0 if it is 0, and 1 if it is not 0.
    """
    if isinstance(data, list):
        return [convert_to_binary(item) for item in data]
    else:
        return 1 if data != 0 else 0

# Example usage:
training_input_grids = convert_to_binary(training_input_grids)
training_output_grids = convert_to_binary(training_output_grids)
training_test_input_grids = convert_to_binary(training_test_input_grids)
training_test_output_grids = convert_to_binary(training_test_output_grids)

# Example usage
save_to_json(training_input_grids, 'binary_data/training_input_grids.json')
save_to_json(training_output_grids, 'binary_data/training_output_grids.json')
save_to_json(training_test_input_grids, 'binary_data/training_test_input_grids.json')
save_to_json(training_test_output_grids, 'binary_data/training_test_output_grids.json')