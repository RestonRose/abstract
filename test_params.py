import torch
from train import train, process_instance_grids, pad_grid, load_json
from evaluation import evalutate
from binary_net import BinaryGridTransformer

if __name__ == '__main__':

    embed_dims = range(16,34,8)
    layer_nums = range(2,8,2)
    dropouts = range(1,5,2)

    datas = ['400', '1700']

    for data in datas:
        for i in embed_dims:
            for j in layer_nums:
                for k  in dropouts:

                    # Load your data from JSON files
                    input_grids = load_json('binary_data' + data + '/training_input_grids.json')
                    output_grids = load_json('binary_data' + data + '/training_output_grids.json')
                    test_inputs = load_json('binary_data' + data + '/training_test_input_grids.json')
                    test_outputs = load_json('binary_data' + data + '/training_test_output_grids.json')

                    # Process each instance in the dataset
                    input_grids_processed = [process_instance_grids(instance) for instance in input_grids]
                    output_grids_processed = [process_instance_grids(instance) for instance in output_grids]
                    test_inputs_processed = [pad_grid(torch.tensor(instance)) for instance in test_inputs]
                    test_outputs_processed = [pad_grid(torch.tensor(instance)) for instance in test_outputs]

                    print('Data Loaded')

                    # Initialize the model, set the device (GPU if available)
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = BinaryGridTransformer(model_dim=i, num_layers=j, dropout=k/10).to(device)

                    # Move data to the appropriate device
                    input_grids_processed = [[grid.to(device) for grid in instance] for instance in input_grids_processed]
                    output_grids_processed = [[grid.to(device) for grid in instance] for instance in output_grids_processed]
                    test_inputs_processed = [grid.to(device) for grid in test_inputs_processed]
                    test_outputs_processed = [grid.to(device) for grid in test_outputs_processed]

                    # Train the model
                    train(model, 
                          device, 
                          input_grids_processed, 
                          output_grids_processed, 
                          test_inputs_processed, 
                          test_outputs_processed,
                          data, 
                          embed_dim=i, 
                          num_layers=j, 
                          dropout=k/10, 
                          num_epochs=1, 
                          batch_size=1)
                    # evaluate the model
                    evalutate(data, f'models/ed{i}nl{j}dr{k/10}dt{data}.pth', i, j, k/10)
