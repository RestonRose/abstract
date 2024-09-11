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


### load data ###

training_challenges = load_json('arc-agi_training_challenges.json')
training_solutions = load_json('arc-agi_training_solutions.json')
evaluation_challenges = load_json('arc-agi_evaluation_challenges.json')
evaluation_solutions = load_json('arc-agi_evaluation_solutions.json')
test_challenges = load_json('arc-agi_test_challenges.json')

datas = ['400', '1700']

for data in datas:

    ### collect raw data ###
    training_input_grids = [] # shape = [num_training_instances, num_input_grids_per_instance, grid_rows, grid_cols]
    training_output_grids = [] # shape = [num_training_instances, num_output_grids_per_instance, grid_rows, grid_cols]
    training_test_input_grids = [] # shape = [num_training_instances, grid_rows, grid_cols]
    training_test_output_grids = [] # shape = [num_training_instances, grid_rows, grid_cols]

    count = 0
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
        #for i in range(len(input_grids)):
        for i in range(1 if data == '400' else len(input_grids)):
            ig = copy.deepcopy(input_grids)
            og= copy.deepcopy(output_grids)
            ig.remove(input_grids[i])
            og.remove(output_grids[i])
            instance_test_inputs.append(input_grids[i])
            instance_test_outputs.append(output_grids[i])
            instance_inputs.append(ig)
            instance_outputs.append(og)
        training_input_grids.extend(instance_inputs)
        training_output_grids.extend(instance_outputs)
        training_test_input_grids.extend(instance_test_inputs)
        training_test_output_grids.extend(instance_test_outputs)

        count += 1

    # save raw data
    save_to_json(training_input_grids, 'raw_data' + data + '/training_input_grids.json')
    save_to_json(training_output_grids, 'raw_data' + data + '/training_output_grids.json')
    save_to_json(training_test_input_grids, 'raw_data' + data + '/training_test_input_grids.json')
    save_to_json(training_test_output_grids, 'raw_data' + data + '/training_test_output_grids.json')


    ### collect binary data ###

    training_input_grids = load_json('raw_data' + data + '/training_input_grids.json')
    training_output_grids = load_json('raw_data' + data + '/training_output_grids.json')
    training_test_input_grids = load_json('raw_data' + data + '/training_test_input_grids.json')
    training_test_output_grids = load_json('raw_data' + data + '/training_test_output_grids.json')

    def convert_to_binary(data):
        """
        Recursively converts every number in the multi-dimensional list `data`
        to 0 if it is 0, and 1 if it is not 0.
        """
        if isinstance(data, list):
            return [convert_to_binary(item) for item in data]
        else:
            return 1 if data != 0 else 0

    training_input_grids = convert_to_binary(training_input_grids)
    training_output_grids = convert_to_binary(training_output_grids)
    training_test_input_grids = convert_to_binary(training_test_input_grids)
    training_test_output_grids = convert_to_binary(training_test_output_grids)

    # Example usage
    save_to_json(training_input_grids, 'binary_data' + data + '/training_input_grids.json')
    save_to_json(training_output_grids, 'binary_data' + data + '/training_output_grids.json')
    save_to_json(training_test_input_grids, 'binary_data' + data + '/training_test_input_grids.json')
    save_to_json(training_test_output_grids, 'binary_data' + data + '/training_test_output_grids.json')




    ### collect evaluation data ###

    evaluation_input_grids = [] # shape = [num_evaluation_instances, num_input_grids_per_instance, grid_rows, grid_cols]
    evaluation_output_grids = [] # shape = [num_evaluation_instances, num_output_grids_per_instance, grid_rows, grid_cols]
    evaluation_test_input_grids = [] # shape = [num_evaluation_instances, grid_rows, grid_cols]
    evaluation_test_output_grids = [] # shape = [num_evaluation_instances, grid_rows, grid_cols]

    count = 0
    for key, task in evaluation_challenges.items():

        # concatonate all of the evaluation and testing inputs and outputs
        input_grids = []
        output_grids = []
        num_train = len(task['train'])
        for i in range(num_train):
            input_grids.append(task['train'][i]['input'])
            output_grids.append(task['train'][i]['output'])
        input_grids.append(task['test'][0]['input'])
        output_grids.append(evaluation_solutions[key][0])
        
        # record all possible testing scenarios
        instance_inputs = [] 
        instance_outputs = []
        instance_test_inputs = []
        instance_test_outputs = []
        for i in range(1 if data == '400' else len(input_grids)):
            ig = copy.deepcopy(input_grids)
            og= copy.deepcopy(output_grids)
            ig.remove(input_grids[i])
            og.remove(output_grids[i])
            instance_test_inputs.append(input_grids[i])
            instance_test_outputs.append(output_grids[i])
            instance_inputs.append(ig)
            instance_outputs.append(og)
        evaluation_input_grids.extend(instance_inputs)
        evaluation_output_grids.extend(instance_outputs)
        evaluation_test_input_grids.extend(instance_test_inputs)
        evaluation_test_output_grids.extend(instance_test_outputs)

        count += 1

    # save raw data
    save_to_json(evaluation_input_grids, 'raw_data' + data + '/evaluation_input_grids.json')
    save_to_json(evaluation_output_grids, 'raw_data' + data + '/evaluation_output_grids.json')
    save_to_json(evaluation_test_input_grids, 'raw_data' + data + '/evaluation_test_input_grids.json')
    save_to_json(evaluation_test_output_grids, 'raw_data' + data + '/evaluation_test_output_grids.json')

    evaluation_input_grids = load_json('raw_data' + data + '/evaluation_input_grids.json')
    evaluation_output_grids = load_json('raw_data' + data + '/evaluation_output_grids.json')
    evaluation_test_input_grids = load_json('raw_data' + data + '/evaluation_test_input_grids.json')
    evaluation_test_output_grids = load_json('raw_data' + data + '/evaluation_test_output_grids.json')

    evaluation_input_grids = convert_to_binary(evaluation_input_grids)
    evaluation_output_grids = convert_to_binary(evaluation_output_grids)
    evaluation_test_input_grids = convert_to_binary(evaluation_test_input_grids)
    evaluation_test_output_grids = convert_to_binary(evaluation_test_output_grids)

    # Example usage
    save_to_json(evaluation_input_grids, 'binary_data' + data + '/evaluation_input_grids.json')
    save_to_json(evaluation_output_grids, 'binary_data' + data + '/evaluation_output_grids.json')
    save_to_json(evaluation_test_input_grids, 'binary_data' + data + '/evaluation_test_input_grids.json')
    save_to_json(evaluation_test_output_grids, 'binary_data' + data + '/evaluation_test_output_grids.json')