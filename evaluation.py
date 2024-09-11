import torch
from binary_net import BinaryGridTransformer
from train import load_json, process_instance_grids, pad_grid, pad_batch_instances

def evalutate(data, path, embed_dim, num_layers, dropout):

    # Load the model for evaluation
    def load_model(model, path):
        model.load_state_dict(torch.load(path))
        model.eval()  # Set to evaluation mode

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BinaryGridTransformer(model_dim=embed_dim, num_layers=num_layers, dropout=dropout).to(device)
    load_model(model, path)

    # load evaluation data

    input_grids = load_json('binary_data' + data + '/evaluation_input_grids.json')
    output_grids = load_json('binary_data' + data + '/evaluation_output_grids.json')
    test_inputs = load_json('binary_data' + data + '/evaluation_test_input_grids.json')
    test_outputs = load_json('binary_data' + data + '/evaluation_test_output_grids.json')

    # Process each instance in the dataset
    input_grids_processed = [process_instance_grids(instance) for instance in input_grids]
    output_grids_processed = [process_instance_grids(instance) for instance in output_grids]
    test_inputs_processed = [pad_grid(torch.tensor(instance)) for instance in test_inputs]
    test_outputs_processed = [pad_grid(torch.tensor(instance)) for instance in test_outputs]

    # Move data to the appropriate device
    input_grids_processed = [[grid.to(device) for grid in instance] for instance in input_grids_processed]
    output_grids_processed = [[grid.to(device) for grid in instance] for instance in output_grids_processed]
    test_inputs_processed = [grid.to(device) for grid in test_inputs_processed]
    test_outputs_processed = [grid.to(device) for grid in test_outputs_processed]

    print('Evaluation data Loaded')

    score = 0
    total_score = 0
    num_samples = len(input_grids)
    batch_size = 1

    for i in range(0, num_samples, batch_size):

        inputs_batch = input_grids_processed[i:i + batch_size]
        outputs_batch = output_grids_processed[i:i + batch_size]
        test_inputs_batch = test_inputs_processed[i:i + batch_size]
        test_outputs_batch = test_outputs_processed[i:i + batch_size]

        padded_inputs_batch = pad_batch_instances(inputs_batch)
        padded_outputs_batch = pad_batch_instances(outputs_batch)

        padded_test_inputs_batch = torch.stack([pad_grid(test_grid) for test_grid in test_inputs_batch])
        padded_test_outputs_batch = torch.stack([pad_grid(test_grid) for test_grid in test_outputs_batch])

        # Forward pass (no thresholding during training)
        predicted_output = model(padded_inputs_batch, padded_outputs_batch, padded_test_inputs_batch, apply_threshold=True)
        padded_test_outputs_batch = padded_test_outputs_batch.view(batch_size, 30, 30).float()

        match = predicted_output == padded_test_outputs_batch
        total_score += match.sum().item()
        if match.sum().item() == match.numel():
            score += 1

        print(f'{path}, Score = {score} / {(i+1)} - {score/(i+1):.4f}%', end = '\r')

    print(f'{path}, Score = {score} / {(i+1)} - {score/(i+1):.4f}%')


if __name__ == '__main__':

    path = 'models/binary_net.pth'
    evalutate(path)
