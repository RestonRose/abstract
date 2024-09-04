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
        # Flatten and pad the grids
        batch_size = input_grids.size(0)
        input_grids = input_grids.view(batch_size, -1, 1)  # Flatten to (batch_size, H*W, 1)
        output_grids = output_grids.view(batch_size, -1, 1)
        test_grid = test_grid.view(batch_size, -1, 1)

        # Concatenate all grids together
        combined_input = torch.cat([input_grids, output_grids, test_grid], dim=1)

        # Convert input to float type for embedding
        combined_input = self.embedding(combined_input.float())

        # Apply dropout after embedding
        combined_input = self.dropout(combined_input)

        # Add positional encoding
        combined_input = self.pos_encoder(combined_input)

        # Transformer encoding
        transformer_output = self.transformer_encoder(combined_input)

        # Apply dropout before final output layer
        transformer_output = self.dropout(transformer_output)

        # Predict the output grid
        predicted_output = self.fc_out(transformer_output[:, -test_grid.size(1):])  # Only take the test grid part
        predicted_output = predicted_output.view(batch_size, self.max_size, self.max_size)

        # Apply sigmoid to convert to binary
        predicted_output = torch.sigmoid(predicted_output)
        predicted_output = (predicted_output > 0.5).float()  # Thresholding at 0.5

        return predicted_output

# Example usage:
# Assume we have batch_size=2, input/output grids with size 12x12, and test grid with size 10x10
batch_size = 2
input_grids = torch.randint(0, 2, (batch_size, 12, 12))
output_grids = torch.randint(0, 2, (batch_size, 12, 12))
test_grid = torch.randint(0, 2, (batch_size, 10, 10))

# Padding to 30x30
input_grids_padded = F.pad(input_grids, (0, 18, 0, 18))
output_grids_padded = F.pad(output_grids, (0, 18, 0, 18))
test_grid_padded = F.pad(test_grid, (0, 20, 0, 20))

# Initialize and forward pass through the model
model = BinaryGridTransformer()
predicted_output = model(input_grids_padded, output_grids_padded, test_grid_padded)

print(predicted_output.size())