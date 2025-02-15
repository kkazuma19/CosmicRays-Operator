import torch
import torch.nn as nn
import torch.nn.functional as F
from fourier_embedding import FourierFeatures
from fcn import FCN
from lstm import LSTM, AttentionLSTM
from gru import GRU, AttentionGRU
from rnn import RNN
from transformer import Transformer, Informer

class SequentialDeepONet(nn.Module):
    def __init__(self, branch_type, branch_input_size, branch_hidden_size, branch_num_layers, branch_output_size, 
                 trunk_architecture, num_outputs, use_transform=True, activation_fn=nn.ReLU, num_heads=4):
        super(SequentialDeepONet, self).__init__()

        self.use_transform = use_transform
        #self.num_timesteps = num_timesteps
        self.num_outputs = num_outputs

        # Initialize the branch network based on the specified type
        if branch_type == 'rnn':
            self.branch_net = RNN(branch_input_size, branch_hidden_size, branch_num_layers, branch_output_size)
        elif branch_type == 'lstm':
            self.branch_net = LSTM(branch_input_size, branch_hidden_size, branch_num_layers, branch_output_size)
        elif branch_type == 'gru':
            self.branch_net = GRU(branch_input_size, branch_hidden_size, branch_num_layers, branch_output_size)
        elif branch_type == 'attention_lstm':
            self.branch_net = AttentionLSTM(branch_input_size, branch_hidden_size, branch_num_layers, branch_output_size, num_heads=num_heads)
        elif branch_type == 'attention_gru':
            self.branch_net = AttentionGRU(branch_input_size, branch_hidden_size, branch_num_layers, branch_output_size, num_heads=num_heads)
        elif branch_type == 'transformer':
            self.branch_net = Transformer(branch_input_size, branch_hidden_size, num_heads, branch_num_layers, branch_output_size)
        elif branch_type == 'informer':
            self.branch_net = Informer(branch_input_size, branch_hidden_size, num_heads, branch_num_layers, branch_output_size)
        else:
            raise ValueError(f"Unsupported branch type: {branch_type}")

        # Trunk network (fully connected using the FCN class)
        self.trunk_net = FCN(trunk_architecture, activation_fn)

        if self.use_transform:
            self.final_layer = nn.Linear(branch_output_size, num_outputs)
        else:
            # Optional bias
            self.b = nn.Parameter(torch.zeros(1))

    def forward(self, branch_input, trunk_input):
        # Process branch input (sequential data) through the selected branch network
        branch_output = self.branch_net(branch_input)  # Assume this gives (batch_size, branch_output_size)

        # Get the last timestep output from the branch network
        branch_output = branch_output[:, -1, :]

        # Process trunk input (spatial data) through the trunk network
        trunk_output = self.trunk_net(trunk_input)

        #print(branch_output.shape, trunk_output.shape)
        combined_output = torch.einsum('bi,bpi->bpi', branch_output, trunk_output)  # Shape: (batch_size, num_trunk_points, hidden_size)
    

        if self.use_transform:
            # Final layer for prediction at each trunk point
            combined_output = self.final_layer(combined_output)

        else:
            # If self.num_outputs < hidden_size, slice the output to match self.num_outputs
            combined_output = combined_output[..., :self.num_outputs]

            # Add bias
            combined_output += self.b

        return combined_output