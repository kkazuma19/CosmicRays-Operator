import torch
import torch.nn as nn
from fourier_embedding import FourierFeatures
from fcn import FCN

class MIONet(nn.Module):
    def __init__(self, branch_arch_list, trunk_arch, num_outputs=1, activation_fn=nn.ReLU, 
                 use_fourier_features=False, num_frequencies=16, merge_type='mul'):
        """
        Args:
            branch_arch_list (list of lists): List of architectures for each branch input.
            trunk_arch (list): Architecture for the trunk network.
            activation_fn (torch.nn.Module): Activation function to use in the networks.
            use_fourier_features (bool): Whether to apply Fourier feature embedding to the trunk input.
            num_frequencies (int): Number of frequency bands for Fourier embedding.
            merge_type (str): Method to combine branch outputs ('sum' or 'mul').
        """
        super(MIONet, self).__init__()

        self.use_fourier_features = use_fourier_features
        self.merge_type = merge_type  # Option to define how to merge the branch outputs

        # Create a branch network for each branch input in branch_arch_list
        self.branch_nets = nn.ModuleList([FCN(arch, activation_fn) for arch in branch_arch_list])

        # If Fourier features are used, initialize the Fourier feature embedding
        if self.use_fourier_features:
            self.fourier_features = FourierFeatures(input_dim=trunk_arch[0], num_frequencies=num_frequencies)

            # Update the trunk architecture to accommodate Fourier feature expansion
            fourier_output_dim = 2 * num_frequencies * trunk_arch[0]
            trunk_arch[0] = fourier_output_dim

        # Trunk network
        self.trunk_net = FCN(trunk_arch, activation_fn)

        # opton to add a final linear layer to the output
        self.final_layer = nn.Linear(branch_arch_list[0][-1], num_outputs, bias=True) 

        # Optional bias added to the final output
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, branch_inputs, trunk_input):
        """
        Forward pass for MIONet.
        Args:
            branch_inputs: List of tensors, each corresponding to one branch input. Shape: [(batch_size, input_size), ...]
            trunk_input: Trunk input tensor. Shape: (batch_size, num_trunk_points, trunk_input_size)

        Returns:
            prediction: Tensor. Shape: (batch_size, num_trunk_points, 1)
        """
        # Process each branch input through its corresponding branch network
        branch_outputs = [branch_net(branch_input) for branch_net, branch_input in zip(self.branch_nets, branch_inputs)]

        # Merge branch outputs based on the selected merge_type
        if self.merge_type == 'sum':
            combined_branch_output = sum(branch_outputs)
        elif self.merge_type == 'mul':
            combined_branch_output = branch_outputs[0]
            for output in branch_outputs[1:]:
                combined_branch_output = combined_branch_output * output  # Element-wise multiplication

        # Apply Fourier features to the trunk input if enabled
        if self.use_fourier_features:
            trunk_input = self.fourier_features(trunk_input)

        # Forward pass through the trunk network
        trunk_output = self.trunk_net(trunk_input)  # Output shape: (batch_size, num_trunk_points, hidden_size)

        # einsum implementation
        combined_output = torch.einsum('bi,bpi->bpi', combined_branch_output, trunk_output)

        # sum 
        #prediction = combined_output.sum(dim=-1) + self.bias

        # Apply the final linear layer
        prediction = self.final_layer(combined_output)

        return prediction