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
import time
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

from utils import SequentialDeepONetDataset
from s_deeponet import SequentialDeepONet
from helper import convert2dim, compute_metrics_region, plot_field_region, fit, load_model_experiment


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Load neutron monitoring data
input_data = np.load('../../data/neutron_data_22yrs.npy')
trunk = np.load('../../data/grid_points.npy')
target = np.load('../../data/dose_array.npy')

# Normalize trunk input
trunk[:, 0] = (trunk[:, 0] - np.min(trunk[:, 0])) / (np.max(trunk[:, 0]) - np.min(trunk[:, 0]))
trunk[:, 1] = (trunk[:, 1] - np.min(trunk[:, 1])) / (np.max(trunk[:, 1]) - np.min(trunk[:, 1]))

from forecasting_analysis import create_windows_forecasting_with_index

dates = pd.date_range("2001-01-01", "2023-12-31", freq="D")

W, H = 30, 0
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
batch_size = 1
print("Batch size:", batch_size)


#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = load_model_experiment('../baseline/single_branch/lstm_window_30.pth')
model = model.to(device)

model.eval()
model.to(device)

# Get one batch from the data_loader
X_batch, trunk_batch, y_batch = next(iter(test_loader))

# pritn shapes
print("X_batch shape:", X_batch.shape)
print("trunk_batch shape:", trunk_batch.shape)
print("y_batch shape:", y_batch.shape)

X_batch = X_batch.to(device).float()
trunk_batch = trunk_batch.to(device).float()
y_batch = y_batch.to(device).float()

# Warmup runs (important on GPU)
with torch.no_grad():
    for _ in range(10):
        _ = model(X_batch, trunk_batch)

# Benchmark
n_runs = 100
times = []

with torch.no_grad():
    if device.type == "cuda":
        torch.cuda.synchronize()
    for _ in range(n_runs):
        t0 = time.perf_counter()
        outputs = model(X_batch, trunk_batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

times = np.array(times)
avg_time_s = times.mean()
std_time_s = times.std(ddof=1)  # sample std
avg_time_ms = avg_time_s * 1000.0
std_time_ms = std_time_s * 1000.0

batch_size = X_batch.shape[0]

print(f"Batch size: {batch_size}")
print(f"Avg inference time per batch: {avg_time_ms:.3f} ± {std_time_ms:.3f} ms")
print(f"Avg time per sample: {avg_time_ms / batch_size:.3f} ± {std_time_ms / batch_size:.3f} ms")
print(f"Min / Max per batch: {times.min()*1000:.3f} / {times.max()*1000:.3f} ms")