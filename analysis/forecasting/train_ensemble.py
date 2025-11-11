import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import sys
import os
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

# now imports that rely on those paths
from utils import SequentialDeepONetDataset
from helper import convert2dim, fit, compute_metrics_region, load_model_experiment, plot_field_region, load_model_experiment_deeponet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Load Datasets
input_data = np.load('../../data/neutron_data_22yrs.npy')
trunk = np.load('../../data/grid_points.npy')
target = np.load('../../data/dose_array.npy')

print("Input data shape:", input_data.shape)
print("Trunk shape:", trunk.shape)
print("Target shape:", target.shape)

from forecasting_analysis import create_windows_forecasting_with_index

dates = pd.date_range("2001-01-01", "2023-12-31", freq="D")

W, H = 30, 1
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

# dataset and dataloader
# create datasets
train_dataset = SequentialDeepONetDataset(X_train_scaled, trunk, y_train_scaled)
val_dataset   = SequentialDeepONetDataset(X_val_scaled,   trunk, y_val_scaled)
test_dataset  = SequentialDeepONetDataset(X_test_scaled,  trunk, y_test_scaled)

# create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Train Ensemble of TRON models
from helper import init_model
from forecasting_analysis import train_model

NUM_MODELS = 5  # typical: 3–10
models = []

for seed in range(NUM_MODELS):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"Training model {seed+1}/{NUM_MODELS} with seed {seed}")

    model_k = init_model()
    model_k = model_k.to(device)

    history = train_model(
        model=model_k,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        scaler_target=scaler_target,
        num_epochs=200,
        lr=1e-3,
        weight_decay=1e-3,
        scheduler_step=20,
        scheduler_gamma=0.7,
        early_stop_patience=20,
        save_path=f"ensemble/model_ensemble_{seed}.pt"
    )

    print(f"Model {seed+1} training complete.")

print("All models trained and saved.")
