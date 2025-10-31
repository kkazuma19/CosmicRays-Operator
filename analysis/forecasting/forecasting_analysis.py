import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


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


import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    scaler_target,
    num_epochs=100,
    lr=1e-3,
    weight_decay=0.0,
    scheduler_step=20,
    scheduler_gamma=0.5,
    early_stop_patience=15,
    save_path="dev_test.pt",
):
    """
    Generic training loop for Sequential DeepONet–style models with branch + trunk inputs.

    Parameters
    ----------
    model : torch.nn.Module
        The DeepONet model; must accept (branch_batch, trunk_batch) → output tensor.
    train_loader : DataLoader
        Training data loader providing (branch, trunk, target).
    val_loader : DataLoader
        Validation data loader providing (branch, trunk, target).
    device : torch.device
        CUDA or CPU device.
    scaler_target : sklearn-like scaler
        Used to inverse-transform predictions for evaluation.
    num_epochs : int
        Maximum number of epochs.
    lr : float
        Learning rate.
    weight_decay : float
        Weight-decay (L2 regularization).
    scheduler_step : int
        StepLR scheduler step size.
    scheduler_gamma : float
        StepLR scheduler decay factor.
    early_stop_patience : int
        Stop training if validation loss does not improve for these many epochs.
    save_path : str
        File path to store the best model checkpoint.

    Returns
    -------
    history : dict
        Training history with epoch-wise losses.
    """

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    criterion = nn.MSELoss()

    best_val_loss = np.inf
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for branch_batch, trunk_batch, target_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            branch_batch = branch_batch.to(device, non_blocking=True)
            trunk_batch  = trunk_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)

            optimizer.zero_grad()

            output = model(branch_batch, trunk_batch)        # Forward pass
            
            loss = criterion(output, target_batch)           # Compute MSE loss
            loss.backward()                                  # Backprop
            optimizer.step()

            running_loss += loss.item() * branch_batch.size(0)

        # Average training loss
        train_loss = running_loss / len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        # ---------------- Validation ----------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for branch_batch, trunk_batch, target_batch in val_loader:
                branch_batch = branch_batch.to(device, non_blocking=True)
                trunk_batch  = trunk_batch.to(device, non_blocking=True)
                target_batch = target_batch.to(device, non_blocking=True)

                output = model(branch_batch, trunk_batch)
                loss = criterion(output, target_batch)
                val_loss += loss.item() * branch_batch.size(0)

        val_loss /= len(val_loader.dataset)
        history["val_loss"].append(val_loss)

        scheduler.step()

        print(f"[Epoch {epoch+1:03d}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # ---------------- Early stopping ----------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {early_stop_patience} epochs).")
            break

    print(f"Training complete. Best validation loss: {best_val_loss:.6f}")
    model.load_state_dict(torch.load(save_path))

    return history
