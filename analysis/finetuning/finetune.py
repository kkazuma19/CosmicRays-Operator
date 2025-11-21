import torch
import torch.nn as nn
from tqdm import tqdm
from utils import SequentialDeepONetDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

def create_contiguous_adaptation_set(X_test_scaled, y_test_scaled, trunk, num_days, batch_size=32):
    """
    Create a realistic few-shot adaptation set using the first `num_days`
    of the 2023 test set (contiguous commissioning window).

    Args:
        X_test_scaled: (365, W, C) scaled branch inputs for 2023
        y_test_scaled: (365, P, 1) scaled targets for 2023
        trunk: (P, 2) trunk coordinates (shared across samples)
        num_days: int, number of first days to use for tuning (e.g., 7, 14, 30)
        batch_size: int

    Returns:
        adapt_loader: DataLoader over first `num_days`
        adapt_idx: numpy array of indices [0, 1, ..., num_days-1]
    """
    N = len(X_test_scaled)  # 365
    k = int(num_days)
    if k < 1 or k > N:
        raise ValueError(f"num_days must be in [1, {N}], got {k}")

    adapt_idx = np.arange(k)

    X_adapt = X_test_scaled[adapt_idx]
    y_adapt = y_test_scaled[adapt_idx]

    adapt_dataset = SequentialDeepONetDataset(X_adapt, trunk, y_adapt)
    adapt_loader  = DataLoader(adapt_dataset, batch_size=batch_size, shuffle=True)

    return adapt_loader, adapt_idx


def create_eval_set_after_contiguous_adaptation(X_test_scaled, y_test_scaled, trunk, adapt_idx, batch_size=32):
    """
    Create evaluation set as the remaining days after the contiguous adaptation window.

    Args:
        X_test_scaled, y_test_scaled, trunk: same as above
        adapt_idx: indices returned by create_contiguous_adaptation_set
        batch_size: int

    Returns:
        eval_loader: DataLoader over days not in adapt_idx
        eval_idx: numpy array of eval indices
    """
    N = len(X_test_scaled)
    all_idx = np.arange(N)
    eval_idx = np.setdiff1d(all_idx, adapt_idx)

    X_eval = X_test_scaled[eval_idx]
    y_eval = y_test_scaled[eval_idx]

    eval_dataset = SequentialDeepONetDataset(X_eval, trunk, y_eval)
    eval_loader  = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    return eval_loader, eval_idx


def freeze_for_new_station_adaptation(model):

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze the newly expanded LSTM input weights (layer 0)
    for name, p in model.branch_net.lstm.named_parameters():
        if name.startswith("weight_ih_l0") or name.startswith("bias_ih_l0"):
            p.requires_grad = True
    
    # Also unfreeze the branch FC layer
    for p in model.branch_net.fc.parameters():
        p.requires_grad = True

    # bias b
    model.b.requires_grad = True
    
def expand_lstm_input_dim_correct(lstm, add_inputs, init_std=1e-2):
    h = lstm.hidden_size
    old_in = lstm.input_size             # 12
    new_in = old_in + add_inputs         # 12 + k
    
    new_lstm = nn.LSTM(
        input_size=new_in,
        hidden_size=h,
        num_layers=lstm.num_layers,
        batch_first=True,
    )

    with torch.no_grad():
        # ---- Expand layer 0 ONLY ----
        # New shape: [4*h, new_in]
        Wih0_old = lstm.weight_ih_l0
        Wih0_new = new_lstm.weight_ih_l0

        # Copy first 12 columns (old stations)
        Wih0_new[:, :old_in] = Wih0_old
        # Initialize new station columns (the added inputs)
        Wih0_new[:, old_in:] = torch.randn_like(Wih0_new[:, old_in:]) * init_std
        # Copy hidden→hidden
        new_lstm.weight_hh_l0.copy_(lstm.weight_hh_l0)
        # Copy biases
        new_lstm.bias_ih_l0.copy_(lstm.bias_ih_l0)
        new_lstm.bias_hh_l0.copy_(lstm.bias_hh_l0)
        # ---- Copy all upper layers *exactly as-is* ----
        for i in range(1, lstm.num_layers):
            getattr(new_lstm, f"weight_ih_l{i}").copy_(
                getattr(lstm, f"weight_ih_l{i}")
            )
            getattr(new_lstm, f"weight_hh_l{i}").copy_(
                getattr(lstm, f"weight_hh_l{i}")
            )
        getattr(new_lstm, f"bias_ih_l{i}").copy_(
            getattr(lstm, f"bias_ih_l{i}")
        )
        getattr(new_lstm, f"bias_hh_l{i}").copy_(
            getattr(lstm, f"bias_hh_l{i}")
        )
    
    return new_lstm


def mask_new_station(X_batch):
    X_masked = X_batch.clone()
    X_masked[:, :, -1] = 0.0        # zero-out MXCO only
    return X_masked

def fine_tune_adapt(model, adapt_loader, optimizer, num_epochs=5, device="cuda"):
    """
    Fine-tune the expanded TRON model on few-shot adaptation data.
    Only the parameters set `requires_grad=True` will be updated.

    Args:
        model: expanded TRON model (with 13-channel branch)
        adapt_loader: few-shot DataLoader (1%, 5%, or 10% subset of 2023)
        optimizer: optimizer with ONLY trainable params
        num_epochs: number of fine-tuning epochs (2–10 recommended)
        device: "cuda" or "cpu"
    """
    
    model = model.to(device)
    model.train()

    loss_fn = nn.MSELoss()   # use same loss as original TRON

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for X_batch, trunk_batch, y_batch in tqdm(adapt_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):

            X_batch = X_batch.to(device).float()         # [B, 30, 13]
            trunk_batch = trunk_batch.to(device).float() # [P, 2]
            y_batch = y_batch.to(device).float()         # [B, P, 1]

            optimizer.zero_grad()

            # Forward prediction
            pred = model(X_batch, trunk_batch)           # [B, P, 1]

            loss = loss_fn(pred, y_batch)
            loss.backward()

            # Gradient clipping (safe)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(adapt_loader)
        print(f"Epoch {epoch+1}: avg adaptation loss = {avg_loss:.6f}")
