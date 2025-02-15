import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers, output_size):
        super(Transformer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Input projection layer
        self.input_fc = nn.Linear(input_size, hidden_size)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_size)

        # Transformer Encoder (with batch_first=True to resolve the warning)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)

        # Output fully connected layer
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Project the input to hidden_size
        x = self.input_fc(x)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Transformer expects (batch_size, seq_len, input_dim) with batch_first=True
        #out = self.transformer_encoder(x)[:, -1, :]
        out = self.transformer_encoder(x)

        # Extract the last time step from the sequence or use all time steps
        # Final output layer
        out = self.fc_out(out)
        return out


class ProbSparseAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(ProbSparseAttention, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=False)  # Set batch_first=False

    def forward(self, query, key, value):
        # Perform sparse attention (standard multi-head attention here)
        attention_out, _ = self.attention(query, key, value)
        return attention_out

class DistillationLayer(nn.Module):
    def __init__(self, d_model):
        super(DistillationLayer, self).__init__()
        self.pooling = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        # Adjust pooling to work with (seq_len, batch_size, d_model)
        x = x.permute(1, 2, 0)  # Change to (batch_size, d_model, seq_len)
        x = self.pooling(x)      # Apply pooling along the seq_len dimension
        x = x.permute(2, 0, 1)   # Back to (new_seq_len, batch_size, d_model)
        return x

class Informer(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers, output_size, max_len=5000):
        super(Informer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Input projection layer
        self.input_fc = nn.Linear(input_size, hidden_size)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_size, max_len=max_len)

        # Informer Encoder Layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            attention = ProbSparseAttention(hidden_size, num_heads)
            distillation = DistillationLayer(hidden_size)
            self.layers.append(nn.ModuleList([attention, distillation]))

        # Output fully connected layer
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Project input to the hidden size
        x = self.input_fc(x)  # Shape: (batch_size, seq_len, hidden_size)

        # Transpose to (seq_len, batch_size, hidden_size) for compatibility with positional encoding
        x = x.transpose(0, 1)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Informer Encoder with Distillation
        for attention, distillation in self.layers:
            # Apply sparse attention
            x = attention(x, x, x)

            # Apply distillation if sequence length is even
            if x.size(0) % 2 == 0:
                x = distillation(x)

        # Final output layer (transpose back to batch-first format for output)
        out = self.fc_out(x.transpose(0, 1))
        return out