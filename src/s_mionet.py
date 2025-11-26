import torch
import torch.nn as nn
from fcn import FCN
from lstm import LSTM
from gru import GRU
from rnn import RNN

class SequentialMIONet(nn.Module):
    def __init__(self, branches_config, trunk_architecture, num_outputs, use_transform=True, activation_fn=nn.ReLU):
        super(SequentialMIONet, self).__init__()

        self.use_transform = use_transform
        self.num_outputs = num_outputs
        self.branches = nn.ModuleDict()

        # Initialize each branch network based on provided configuration with specified parameters
        for branch_name, config in branches_config.items():
            branch_type = config['type']
            input_size = config['input_size']
            hidden_size = config['hidden_size']
            num_layers = config['num_layers']
            output_size = config['output_size']
            num_heads = config.get('num_heads', 4)  # Default to 4 heads for attention-based models

            # Configure each branch network type
            if branch_type == 'rnn':
                self.branches[branch_name] = RNN(input_size, hidden_size, num_layers, output_size)
            elif branch_type == 'lstm':
                self.branches[branch_name] = LSTM(input_size, hidden_size, num_layers, output_size)
            elif branch_type == 'gru':
                self.branches[branch_name] = GRU(input_size, hidden_size, num_layers, output_size)
            else:
                raise ValueError(f"Unsupported branch type: {branch_type}")

        # Trunk network with similar architecture format
        self.trunk_net = FCN(trunk_architecture, activation_fn)

        if self.use_transform:
            self.final_layer = nn.Linear(output_size, num_outputs)
        else:
            self.b = nn.Parameter(torch.zeros(1))

    def forward(self, branch_inputs, trunk_input):
        # Process each branch input through its corresponding branch network
        branch_outputs = []
        for branch_name, branch_input in branch_inputs.items():
            branch_output = self.branches[branch_name](branch_input)
            # Take the output from the last time step if it's a sequential model
            branch_outputs.append(branch_output[:, -1, :])

        # Multiple branch outputs are combined through element-wise multiplication
        combined_branch_output = torch.prod(torch.stack(branch_outputs), dim=0)

        # Process trunk input through trunk network
        trunk_output = self.trunk_net(trunk_input)

        # Combine branch and trunk outputs
        combined_output = torch.einsum('bi,bpi->bpi', combined_branch_output, trunk_output)  # Shape: (batch_size, num_trunk_points, hidden_size)

        #print('debug point1', combined_output.shape)
        
        if self.use_transform:
            # Final layer for prediction at each trunk point
            combined_output = self.final_layer(combined_output)
        else:
            # If self.num_outputs < hidden_size, slice the output to match self.num_outputs
            #combined_output = combined_output[..., :self.num_outputs]
            combined_output = combined_output.sum(dim=-1, keepdim=True)  # Shape: [batch_size, num_trunk_points, 1]

            # Add bias
            combined_output += self.b

        return combined_output