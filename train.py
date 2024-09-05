from collect_raw_data import load_json
from binary_net import BinaryGridTransformer
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# Load your data from JSON files
input_grids = load_json('drive/MyDrive/binary_data/training_input_grids.json')
output_grids = load_json('drive/MyDrive/binary_data/training_output_grids.json')
test_inputs = load_json('drive/MyDrive/binary_data/training_test_input_grids.json')
test_outputs = load_json('drive/MyDrive/binary_data/training_test_output_grids.json')

# Pad each individual grid to 30x30
def pad_grid(grid, target_size=(30, 30)):
    grid_rows, grid_cols = grid.shape[-2], grid.shape[-1]
    padding = (0, target_size[1] - grid_cols, 0, target_size[0] - grid_rows)
    return F.pad(grid, padding)

# Pad each grid within the input/output/test grids per instance
def process_instance_grids(instance_grids):
    return [pad_grid(torch.tensor(grid)) for grid in instance_grids]

# Process each instance in the dataset
input_grids_processed = [process_instance_grids(instance) for instance in input_grids]
output_grids_processed = [process_instance_grids(instance) for instance in output_grids]
test_inputs_processed = [pad_grid(torch.tensor(instance)) for instance in test_inputs]
test_outputs_processed = [pad_grid(torch.tensor(instance)) for instance in test_outputs]

print('Data Loaded')

# Pad a batch of instances with varying grid counts
def pad_batch_instances(grids_batch):
    # Each instance in grids_batch is a list of padded grids
    # We need to pad across the number of examples in each instance
    padded_batch = [torch.stack(grids) for grids in grids_batch]
    return pad_sequence(padded_batch, batch_first=True)

# Define training loop with varying grid counts and grid sizes
def train(model, input_grids, output_grids, test_inputs, test_outputs, num_epochs=20, batch_size=8, lr=0.001):
    # Set the model to training mode
    model.train()
    
    # Define the loss function (binary cross-entropy)
    criterion = nn.BCELoss()

    # Optimizer (Adam optimizer)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Number of training instances
    num_samples = len(input_grids)

    for epoch in range(num_epochs):
        running_loss = 0.0
        
        # Shuffle the data
        perm = torch.randperm(num_samples)
        input_grids = [input_grids[i] for i in perm]
        output_grids = [output_grids[i] for i in perm]
        test_inputs = [test_inputs[i] for i in perm]
        test_outputs = [test_outputs[i] for i in perm]

        for i in range(0, num_samples, batch_size):
            # Get the batch
            inputs_batch = input_grids[i:i + batch_size]
            outputs_batch = output_grids[i:i + batch_size]
            test_inputs_batch = test_inputs[i:i + batch_size]
            test_outputs_batch = test_outputs[i:i + batch_size]
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Pad the batches dynamically for variable-length input
            padded_inputs_batch = pad_batch_instances(inputs_batch)
            padded_outputs_batch = pad_batch_instances(outputs_batch)

            # Convert test inputs/outputs to tensors and pad them to 30x30
            padded_test_inputs_batch = torch.stack([pad_grid(test_grid) for test_grid in test_inputs_batch])
            padded_test_outputs_batch = torch.stack([pad_grid(test_grid) for test_grid in test_outputs_batch])
            
            # Forward pass
            predicted_output = model(padded_inputs_batch, padded_outputs_batch, padded_test_inputs_batch)
            
            # Reshape the test_outputs_batch to match the predicted_output shape
            padded_test_outputs_batch = padded_test_outputs_batch.view(batch_size, 30, 30).float()
            
            # Compute the loss
            loss = criterion(predicted_output, padded_test_outputs_batch)
            
            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

            # Accumulate the running loss
            running_loss += loss.item()

        # Print the average loss for this epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / (num_samples // batch_size):.4f}')

    print("Training complete!")


# Initialize the model, set the device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BinaryGridTransformer().to(device)

# Move data to the appropriate device
input_grids_processed = [[grid.to(device) for grid in instance] for instance in input_grids_processed]
output_grids_processed = [[grid.to(device) for grid in instance] for instance in output_grids_processed]
test_inputs_processed = [grid.to(device) for grid in test_inputs_processed]
test_outputs_processed = [grid.to(device) for grid in test_outputs_processed]

# Train the model
train(model, input_grids_processed, output_grids_processed, test_inputs_processed, test_outputs_processed, num_epochs=10, batch_size=16)
