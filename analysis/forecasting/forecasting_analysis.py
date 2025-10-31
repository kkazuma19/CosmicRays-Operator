import pandas as pd
import numpy as np
import torch


def create_windows_forecasting_with_index(input_data, target_data, window_size, horizon=1):
    """
    Generates sliding windows for time series forecasting, returning both the
    input/target tensors and the corresponding target indices for chronological splitting.

    Each input window consists of `window_size` consecutive time steps from `input_data`,
    and its corresponding target is taken from `target_data` at a future time step defined
    by `horizon` (e.g., horizon=1 means one-step-ahead forecasting).

    Example:
        If window_size=30 and horizon=1,
        - input covers t=1..30
        - target is at t=31

    Parameters
    ----------
    input_data : np.ndarray
        Array of shape (T, F), where T is the number of time steps
        and F is the number of input features (e.g., sensor channels).
    target_data : np.ndarray
        Array of shape (T, D), where D is the output dimension
        (e.g., flattened spatial field).
    window_size : int
        Number of past time steps to include in each input window.
    horizon : int, optional
        Forecasting horizon, i.e., how many steps ahead to predict.
        Default is 1 (one-step-ahead forecast).

    Returns
    -------
    X : torch.FloatTensor
        Tensor of shape (N, window_size, F), containing all input sequences.
    y : torch.FloatTensor
        Tensor of shape (N, D), containing the corresponding forecast targets.
    tgt_idx : np.ndarray
        Array of length N containing the target time indices (0-based),
        useful for assigning each window to train/validation/test sets
        based on the target timestamp.

    Notes
    -----
    - The function automatically discards incomplete windows near the end
      of the sequence where the target index would exceed available data.
    - To ensure temporal integrity, assign samples to splits using `tgt_idx`
      and chronological timestamps, not random shuffling.
    """
    X, y, tgt_idx = [], [], []
    N = len(input_data)

    for i in range(N - window_size - horizon + 1):
        X.append(input_data[i : i + window_size])
        y.append(target_data[i + window_size - 1 + horizon])
        tgt_idx.append(i + window_size - 1 + horizon)

    return (
        torch.tensor(np.array(X), dtype=torch.float32),
        torch.tensor(np.array(y), dtype=torch.float32),
        np.array(tgt_idx, dtype=np.int64)
    )
