import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
        # Layer normalization
        #
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size, bias=True)
    

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate GRU
        out, _ = self.gru(x, h0)  # GRU output
        
        # Apply layer normalization to GRU outputs (applied to the last hidden layer)
        out = self.layer_norm(out)

        # Fully connected layer (taking the output of the last time step)
        #out = self.fc(out[:, -1, :])
        out = self.fc(out)
        return out

class AttentionGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_heads=4):
        super(AttentionGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # GRU layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size, bias=True)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate GRU
        out, _ = self.gru(x, h0)  # GRU output shape: (batch_size, seq_length, hidden_size)
        
        # Apply layer normalization
        out = self.layer_norm(out)
        
        # Apply multi-head attention on GRU output
        attention_out, _ = self.attention(out, out, out)  # Self-attention on GRU outputs

        # Optionally apply another normalization layer to stabilize attention output
        attention_out = self.layer_norm(attention_out)

        # Fully connected layer (projection)
        output = self.fc(attention_out)
        return output