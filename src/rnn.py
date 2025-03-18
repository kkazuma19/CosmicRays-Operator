import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size, bias=True)
    

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate RNN
        out, _ = self.rnn(x, h0)  # RNN output
        
        # Apply layer normalization to RNN outputs (applied to the last hidden layer)
        #out = self.layer_norm(out)

        # Fully connected layer (taking the output of the last time step)
        #out = self.fc(out[:, -1, :])
        #out = self.fc(out)
        return out
