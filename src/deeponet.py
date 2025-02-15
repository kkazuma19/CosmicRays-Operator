import torch
import torch.nn as nn
from fcn import FCN

class DeepONet(nn.Module):
    
    """
    DeepONet class
    branch_arch: List containing the architecture of the branch network
    trunk_arch: List containing the architecture of the trunk network
    num_outputs: Number of outputs (default: 1) if targets are vectors, set num_outputs to the dimension of the target vector
    (e.g., for two different quantities at each trunk point, set num_outputs=2)
    """

    def __init__(self, branch_arch, trunk_arch, num_outputs=1, activation_fn=nn.ReLU):
        super(DeepONet, self).__init__()

        # Branch network: Fully connected network for branch input
        self.branch_net = FCN(branch_arch, activation_fn)

        # Trunk network: Fully connected network for trunk input
        self.trunk_net = FCN(trunk_arch, activation_fn)

        # Final layer to combine outputs from branch and trunk
        self.final_layer = nn.Linear(branch_arch[-1], num_outputs)

    def forward(self, branch_input, trunk_input):
        """
        Forward pass
        branch_input: (batch_size, input_size)
        trunk_input: (batch_size, num_trunk_points, trunk_input_size)
        """

        # Forward pass through the branch network (process branch input for each sample)
        branch_output = self.branch_net(branch_input)  # Output: (batch_size, hidden_size)

        # Forward pass through the trunk network for each sample
        batch_size, num_trunk_points, trunk_input_size = trunk_input.shape
        trunk_output = []
        for i in range(batch_size):
            trunk_output.append(self.trunk_net(trunk_input[i]))  # Output for each sample: (num_trunk_points, hidden_size)
        trunk_output = torch.stack(trunk_output)  # Shape: (batch_size, num_trunk_points, hidden_size)

        # Combine branch and trunk outputs using einsum
        combined_output = torch.einsum('bi,bpi->bpi', branch_output, trunk_output)  # Shape: (batch_size, num_trunk_points, hidden_size)

        # Final layer for prediction at each trunk point
        prediction = self.final_layer(combined_output)  # Output shape: (batch_size, num_trunk_points, num_outputs)

        return prediction
