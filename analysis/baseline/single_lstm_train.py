# single branch GRU training script
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import argparse
import sys
import os
import pandas as pd
from torch.utils.data import DataLoader

# project root = two directories up from baseline
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

src_dir = os.path.join(proj_root, "src")
forecast_dir = os.path.join(proj_root, "analysis", "forecasting")

for p in (src_dir, forecast_dir):
    if p not in sys.path:
        sys.path.append(p)

print(">>> Added to PYTHONPATH:")
print(src_dir)
print(forecast_dir)
    
from utils import create_sliding_windows, SequentialDeepONetDataset
from s_deeponet import SequentialDeepONet

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

import random

seed = 123456
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

print(f"Set random seed to {seed}")

# %%
# Load neutron monitoring data
input_data = np.load('../../data/neutron_data_22yrs.npy')
trunk = np.load('../../data/grid_points.npy')
target = np.load('../../data/dose_array.npy')

# Normalize trunk input
trunk[:, 0] = (trunk[:, 0] - np.min(trunk[:, 0])) / (np.max(trunk[:, 0]) - np.min(trunk[:, 0]))
trunk[:, 1] = (trunk[:, 1] - np.min(trunk[:, 1])) / (np.max(trunk[:, 1]) - np.min(trunk[:, 1]))


# %%
parser = argparse.ArgumentParser()
parser.add_argument("--window_size", type=int, required=True, help="Size of the sliding window")
args = parser.parse_args()

# Use args.window_size in your training pipeline
window_size = args.window_size
print(f"Training with window_size={window_size}")

# --------------------------------------------------------------------------
from forecasting_analysis import create_windows_forecasting_with_index

dates = pd.date_range("2001-01-01", "2023-12-31", freq="D")

W, H = window_size, 0
X_all, y_all, tgt_idx = create_windows_forecasting_with_index(input_data, target, W, H)
tgt_dates = dates[tgt_idx]

train_mask = (tgt_dates <= pd.Timestamp("2021-12-31"))
val_mask   = (tgt_dates >= pd.Timestamp("2022-01-01")) & (tgt_dates <= pd.Timestamp("2022-12-31"))
test_mask  = (tgt_dates >= pd.Timestamp("2023-01-01")) & (tgt_dates <= pd.Timestamp("2023-12-31"))

X_train, y_train = X_all[train_mask], y_all[train_mask]
X_val,   y_val   = X_all[val_mask],   y_all[val_mask]
X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

# check shapes
print("Train set:", X_train.shape, y_train.shape)
print("Validation set:", X_val.shape, y_val.shape)
print("Test set:", X_test.shape, y_test.shape)


scaler_input = MinMaxScaler()
X_train_scaled = scaler_input.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_val_scaled   = scaler_input.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
X_test_scaled  = scaler_input.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)


scaler_target = MinMaxScaler()
y_train_scaled = scaler_target.fit_transform(y_train)[..., np.newaxis]
y_val_scaled   = scaler_target.transform(y_val)[..., np.newaxis]
y_test_scaled  = scaler_target.transform(y_test)[..., np.newaxis]

# --------------------------------------------------------------------------
train_dataset = SequentialDeepONetDataset(X_train_scaled, trunk, y_train_scaled)
val_dataset   = SequentialDeepONetDataset(X_val_scaled,   trunk, y_val_scaled)
test_dataset  = SequentialDeepONetDataset(X_test_scaled,  trunk, y_test_scaled)

# create dataloaders
print("Create DataLoaders for training and validation sets\n-----------------------------------------")
batch_size = 16
print("Batch size:", batch_size)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# %%
def init_model():
    dim = 128
    model = SequentialDeepONet(
        branch_type='lstm',
        branch_input_size=12,
        branch_hidden_size=128,
        branch_num_layers=4,
        branch_output_size=dim,
        trunk_architecture=[2, 128, 128, dim],
        num_outputs=1,
        use_transform=False,
        activation_fn=nn.ReLU,
    )
    return model

# %%
model = init_model().to(device)
print(model)

# %%
print("Set the hyperparameters\n-----------------------------------------")
# save path
save_dir = 'single_branch/'
save_path = os.path.join(save_dir, f'lstm_window_{window_size}.pth')
print(save_path)

num_epochs = 1000
learning_rate = 1e-3
patience = 10

print("Number of epochs:", num_epochs)
print("Learning rate:", learning_rate)
print("Patience:", patience)

# Loss function
criterion = nn.MSELoss()
print("Loss function:", criterion)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
print("Optimizer:", optimizer)
print("-----------------------------------------")

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience, save_path):
    best_val_loss = np.inf
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, trunk, targets in train_loader:
            inputs, trunk, targets = inputs.to(device), trunk.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, trunk)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, trunk, targets in val_loader:
                inputs, trunk, targets = inputs.to(device), trunk.to(device), targets.to(device)
                outputs = model(inputs, trunk)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Scheduler step
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {train_loss:.6f}, "
            f"Validation Loss: {val_loss:.6f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.5e}"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("Training completed!")


# train
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience, save_path)


# %% load the model from the saved path
model = init_model().to(device)
model.load_state_dict(torch.load(save_path))
model.eval()

# %%
def evaluate_model(model, test_loader, scaler, device='cuda'):
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for branch_batch, trunk_batch, target_batch in test_loader:
            branch_batch, trunk_batch, target_batch = (
                branch_batch.to(device),
                trunk_batch.to(device),
                target_batch.to(device),
            )
            output = model(branch_batch, trunk_batch)
            all_preds.append(output.cpu().numpy())
            all_targets.append(target_batch.cpu().numpy())

    # Convert lists to numpy arrays
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Reshape to 2D (n_samples, n_features) for inverse scaling
    all_preds = all_preds.reshape(all_preds.shape[0], -1)
    all_targets = all_targets.reshape(all_targets.shape[0], -1)

    print("All predictions shape after reshape:", all_preds.shape)
    print("All targets shape after reshape:", all_targets.shape)

    # Inverse scaling
    all_preds = scaler.inverse_transform(all_preds)
    all_targets = scaler.inverse_transform(all_targets)
    
    # save the predictions and targets to a file together
    save_path = os.path.join(save_dir, f'array/lstm_window_{window_size}_preds_targets.npy')
    np.save(save_path, np.stack((all_preds, all_targets), axis=1))
    print(f"Predictions and targets saved to {save_path}")
    
# %%
print(evaluate_model(model, test_loader, scaler_target, device=device))
