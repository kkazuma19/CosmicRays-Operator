import torch
from torch.utils.data import Dataset
import numpy as np


class DeepONetDataset(Dataset):
    ''' 
    Custom dataset class for the DeepONet model.
    
    Args:
    - branch_data: Branch input data, shape (num_samples, input_size)
    - trunk_data: Trunk input data, shape (num_trunk_points, trunk_size)
    - target_data: Target output data, shape (num_samples, num_trunk_points)
    
    This dataset assumes the trunk input is shared across all samples.
    '''
    
    def __init__(self, branch_data, trunk_data, target_data):
        self.branch_data = torch.tensor(branch_data, dtype=torch.float32)
        self.trunk_data = torch.tensor(trunk_data, dtype=torch.float32)    # Shared trunk input (100, 1)
        self.target_data = torch.tensor(target_data, dtype=torch.float32)

    def __len__(self):
        return len(self.branch_data)  # Return the number of samples

    def __getitem__(self, idx):
        # Get the branch input and target output for this index
        branch_input = self.branch_data[idx]
        target_output = self.target_data[idx]
        # Return the branch input, shared trunk input (same for all samples), and target output
        return branch_input, self.trunk_data, target_output

class MIONetDataset(Dataset):
    def __init__(self, branch_data_list, trunk_data, target_data):
        """
        Args:
            branch_data_list (list of np.ndarray): List of branch data arrays.
            trunk (np.ndarray): Trunk data array.
            target (np.ndarray): Target data array.
        """
        # Convert each branch input to a PyTorch tensor
        self.branches = [torch.tensor(branch, dtype=torch.float32) for branch in branch_data_list]
        self.trunk = torch.tensor(trunk_data, dtype=torch.float32)
        self.target = torch.tensor(target_data, dtype=torch.float32)

    def __len__(self):
        return len(self.target)  # Assuming target length matches the dataset size

    def __getitem__(self, idx):
        # For each sample, return a tuple of all branch inputs, the trunk input, and the target output
        branch_inputs = [branch[idx] for branch in self.branches]
        target_output = self.target[idx]

        # Return as tuple: list of branch inputs, trunk input, target output
        return branch_inputs, self.trunk, target_output


class SequentialDeepONetDataset(Dataset):
    """
    Custom Dataset for Sequential DeepONet.
    Args:
        branch_data (numpy.ndarray or torch.Tensor): Sequential input data (branch input) of shape (num_samples, sequence_length, num_features).
        trunk_data (numpy.ndarray or torch.Tensor): Trunk input data (spatial/domain points) of shape (num_samples, num_trunk_points, num_trunk_features).
        target_data (numpy.ndarray or torch.Tensor): Target output data of shape (num_samples, num_trunk_points, num_outputs).
    """
    def __init__(self, branch_data, trunk_data, target_data):
        self.branch_data = torch.as_tensor(branch_data, dtype=torch.float32)
        self.trunk_data = torch.as_tensor(trunk_data, dtype=torch.float32)
        self.target_data = torch.as_tensor(target_data, dtype=torch.float32)

    def __len__(self):
        return len(self.branch_data)

    def __getitem__(self, idx):
        branch_input = self.branch_data[idx]  # Sequential input (e.g., from an LSTM branch)
        target_output = self.target_data[idx]  # Output value to predict
        return branch_input, self.trunk_data, target_output
    

class SequentialMIONetDataset(Dataset):
    """
    Custom Dataset for SequentialMIONet.
    Args:
        branch_data (numpy.ndarray or torch.Tensor): Sequential input data of shape 
            (num_samples, sequence_length, num_branches).
        trunk_data (numpy.ndarray or torch.Tensor): Trunk input data of shape 
            (num_samples, num_trunk_points, num_trunk_features).
        target_data (numpy.ndarray or torch.Tensor): Target output data of shape 
            (num_samples, num_trunk_points, num_outputs).
    """
    def __init__(self, branch_data, trunk_data, target_data):
        self.branch_data = torch.as_tensor(branch_data, dtype=torch.float32)
        self.trunk_data = torch.as_tensor(trunk_data, dtype=torch.float32)
        self.target_data = torch.as_tensor(target_data, dtype=torch.float32)
        self.num_branches = self.branch_data.shape[-1]

    def __len__(self):
        return len(self.branch_data)

    def __getitem__(self, idx):
        branch_input = {f"sensor{i+1}": self.branch_data[idx, :, i].unsqueeze(-1) for i in range(self.num_branches)}
        target_output = self.target_data[idx]
        return branch_input, self.trunk_data, target_output


def create_sliding_windows(input_data, target_data, window_size):
    """
    Creates sliding windows for time series data, where the target is the last element of each input sequence.
    
    Parameters:
    - input_data (numpy array): The input sequence of data.
    - target_data (numpy array): The target sequence of data (effective dose rate).
    - window_size (int): The size of the window (number of time steps for each input).
    
    Returns:
    - torch.Tensor: Sliding window sequences for inputs.
    - torch.Tensor: Corresponding target values (last element in each sequence).
    """
    input_sequences = []
    target_values = []

    for i in range(len(input_data) - window_size + 1):
        # Input sequence includes neutron counts from T1 to T29
        input_sequence = input_data[i: i + window_size]
        
        # Target is the effective dose rate at T29 (last element in the input sequence window)
        target_value = target_data[i + window_size - 1]
        
        input_sequences.append(input_sequence)
        target_values.append(target_value)

    # Convert lists to numpy arrays before converting to torch tensors
    input_sequences = np.array(input_sequences)
    target_values = np.array(target_values)

    return torch.tensor(input_sequences, dtype=torch.float32), torch.tensor(target_values, dtype=torch.float32)
