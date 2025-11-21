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
    
class LSTM_Dropout(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTM_Dropout, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer with dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Layer normalization
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