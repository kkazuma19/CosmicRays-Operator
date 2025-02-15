import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Layer normalizationM
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size, bias=True)
    

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # LSTM output
        
        # Apply layer normalization to LSTM outputs (applied to the last hidden layer)
        out = self.layer_norm(out)

        # Fully connected layer (taking the output of the last time step)
        #out = self.fc(out[:, -1, :])
        out = self.fc(out)
        return out
    
class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_heads=4):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size, bias=True)
    
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # LSTM output shape: (batch_size, seq_length, hidden_size)
        
        # Apply layer normalization
        out = self.layer_norm(out)
        
        # Apply multi-head attention on LSTM output
        attention_out, _ = self.attention(out, out, out)  # Self-attention on LSTM outputs

        # Optionally apply another normalization layer to stabilize attention output
        attention_out = self.layer_norm(attention_out)

        # Fully connected layer (projection)
        output = self.fc(attention_out)
        return output
