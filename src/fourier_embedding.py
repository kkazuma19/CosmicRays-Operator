import torch
import torch.nn as nn
import torch.nn.functional as F

# Frequency-based augmentation using Fourier features
class FourierFeatures(nn.Module):
    """
    FourierFeatures class: Applies Fourier feature embedding to the spatial coordinates (trunk inputs).
    Fourier features help capture high-frequency variations and improve the model's ability to handle complex 
    spatial patterns, especially in physics-informed neural networks.

    Parameters:
    - input_dim (int): Dimensionality of the input spatial coordinates (e.g., 2 for (x, y)).
    - num_frequencies (int): The number of frequency bands used for Fourier feature embedding (default: 16).

    Methods:
    - forward(x): Takes input tensor `x` of shape (batch_size, num_trunk_points, input_dim), applies Fourier feature 
      encoding, and returns the encoded tensor of shape (batch_size, num_trunk_points, expanded_dim).
    """
    def __init__(self, input_dim, num_frequencies=16):
        super(FourierFeatures, self).__init__()
        self.num_frequencies = num_frequencies
        # Create frequency bands
        self.freq_bands = 2 * torch.pi * torch.linspace(1.0, num_frequencies, num_frequencies)
        
    def forward(self, x):
        """
        Forward pass of FourierFeatures
        x: Tensor of shape (batch_size, num_trunk_points, input_dim) - spatial coordinates (x, y)
        
        Returns:
        - encoded: Tensor of shape (batch_size, num_trunk_points, expanded_dim), where expanded_dim = 2 * num_frequencies * input_dim
        """
        # Move freq_bands to the same device as input x
        self.freq_bands = self.freq_bands.to(x.device)

        x_expanded = x.unsqueeze(-1)  # Expand last dimension to apply frequencies
        encoded = torch.cat([torch.sin(self.freq_bands * x_expanded), torch.cos(self.freq_bands * x_expanded)], dim=-1)
        encoded = encoded.view(x.shape[0], x.shape[1], -1)  # Flatten the sine and cosine terms into a single dimension
        return encoded