import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=900):  # max_len is 30x30
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

# Transformer Model with Dropout
class BinaryGridTransformer(nn.Module):
    def __init__(self, input_dim=1, model_dim=64, nhead=8, num_layers=4, dropout=0.1, max_size=30):
        super(BinaryGridTransformer, self).__init__()

        self.max_size = max_size
        self.embedding = nn.Linear(input_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

        self.pos_encoder = PositionalEncoding(model_dim, max_len=max_size*max_size)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.fc_out = nn.Linear(model_dim, 1)
        
    def forward(self, input_grids, output_grids, test_grid):
        # input_grids, output_grids: [batch_size, num_examples, H, W]
        # test_grid: [batch_size, 1, H, W]

        batch_size = input_grids.size(0)
        num_examples = input_grids.size(1)

        # Flatten and pad the example grids (input + output) to the max_size (30x30)
        input_grids = input_grids.view(batch_size * num_examples, -1, 1)  # Flatten to (batch_size * num_examples, H*W, 1)
        output_grids = output_grids.view(batch_size * num_examples, -1, 1)

        # Concatenate input and output grids for each example
        combined_example_grids = torch.cat([input_grids, output_grids], dim=1)  # Shape: (batch_size * num_examples, 2*H*W, 1)
        
        # Process the test grid separately
        test_grid = test_grid.view(batch_size, -1, 1)  # Flatten test grid: (batch_size, H*W, 1)

        # Embed and apply dropout to example grids
        combined_example_grids = self.embedding(combined_example_grids.float())
        combined_example_grids = self.dropout(combined_example_grids)

        # Add positional encoding to example grids
        combined_example_grids = self.pos_encoder(combined_example_grids)

        # Transformer encoding for the combined example grids
        transformer_output = self.transformer_encoder(combined_example_grids)

        # Reshape transformer output to split it back into batch size and num_examples
        transformer_output = transformer_output.view(batch_size, num_examples, -1, transformer_output.size(-1))

        # Take the average across the num_examples axis
        transformer_output = torch.mean(transformer_output, dim=1)  # Average across examples

        # Embed and apply dropout to the test grid
        test_grid = self.embedding(test_grid.float())
        test_grid = self.dropout(test_grid)

        # Add positional encoding to the test grid
        test_grid = self.pos_encoder(test_grid)

        # Concatenate transformer output from examples with the test grid
        combined_test_example = torch.cat([transformer_output, test_grid], dim=1)

        # Transformer encoding for the combined test input
        combined_output = self.transformer_encoder(combined_test_example)

        # Output prediction based on the test grid part
        predicted_output = self.fc_out(combined_output[:, -test_grid.size(1):])  # Only take the test grid part
        predicted_output = predicted_output.view(batch_size, self.max_size, self.max_size)

        # Apply sigmoid to convert to binary
        predicted_output = torch.sigmoid(predicted_output)
        predicted_output = (predicted_output > 0.5).float()  # Thresholding at 0.5

        return predicted_output


''' # Example usage with up to 5 input/output grids and 1 test grid
batch_size = 2
input_grids = torch.randint(0, 2, (batch_size, 5, 12, 12))  # 5 example input grids
output_grids = torch.randint(0, 2, (batch_size, 5, 12, 12))  # 5 example output grids
test_grid = torch.randint(0, 2, (batch_size, 1, 10, 10))     # 1 test grid

# Padding to 30x30
input_grids_padded = F.pad(input_grids, (0, 18, 0, 18))  # (batch_size, num_examples, 30, 30)
output_grids_padded = F.pad(output_grids, (0, 18, 0, 18))  # (batch_size, num_examples, 30, 30)
test_grid_padded = F.pad(test_grid, (0, 20, 0, 20))      # (batch_size, 1, 30, 30)

# Initialize the model and forward pass
model = BinaryGridTransformer()
predicted_output = model(input_grids_padded, output_grids_padded, test_grid_padded)

print(predicted_output.shape)  # Should be (batch_size, 30, 30) '''