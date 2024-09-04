import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import copy

# Function to load JSON data
def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

# Save data to a JSON file
def save_to_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f)

# Load data
training_challenges = load_json('arc-agi_training_challenges.json')
training_solutions = load_json('arc-agi_training_solutions.json')
evaluation_challenges = load_json('arc-agi_evaluation_challenges.json')
evaluation_solutions = load_json('arc-agi_evaluation_solutions.json')
test_challenges = load_json('arc-agi_test_challenges.json')

# collect data
training_input_grids = [] # shape = [num_training_instances, num_input_grids_per_instance, grid_rows, grid_cols]
training_output_grids = [] # shape = [num_training_instances, num_output_grids_per_instance, grid_rows, grid_cols]
training_test_input_grids = [] # shape = [num_training_instances, grid_rows, grid_cols]
training_test_output_grids = [] # shape = [num_training_instances, grid_rows, grid_cols]
for key, task in training_challenges.items():
    # concatonate all of the training and testing inputs and outputs
    input_grids = []
    output_grids = []
    num_train = len(task['train'])
    for i in range(num_train):
        input_grids.append(task['train'][i]['input'])
        output_grids.append(task['train'][i]['output'])
    input_grids.append(task['test'][0]['input'])
    output_grids.append(training_solutions[key][0])
    
    # record all possible testing scenarios
    instance_inputs = [] 
    instance_outputs = []
    instance_test_inputs = []
    instance_test_outputs = []
    for i in range(len(input_grids)):
        instance_test_inputs.append(input_grids[i])
        instance_test_outputs.append(output_grids[i])
        instance_inputs.append([x for x in input_grids if x != instance_test_inputs[-1]])
        instance_outputs.append([x for x in output_grids if x != instance_test_outputs[-1]])
    training_input_grids.extend(instance_inputs)
    training_output_grids.extend(instance_outputs)
    training_test_input_grids.extend(instance_test_inputs)
    training_test_output_grids.extend(instance_test_outputs)

# Example usage
save_to_json(training_input_grids, 'raw_data/training_input_grids.json')
save_to_json(training_output_grids, 'raw_data/training_output_grids.json')
save_to_json(training_test_input_grids, 'raw_data/training_test_input_grids.json')
save_to_json(training_test_output_grids, 'raw_data/training_test_output_grids.json')