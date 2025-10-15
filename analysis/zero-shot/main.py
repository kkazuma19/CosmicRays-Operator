import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import sys
import os
from torch.utils.data import DataLoader


# compactly add project src and analysis/zero-shot to sys.path if not already present
for rel in ('../../src', 'analysis/zero-shot'):
    p = os.path.abspath(os.path.join(os.getcwd(), rel))
    if p not in sys.path:
        sys.path.append(p)

# now imports that rely on those paths
from utils import create_sliding_windows, SequentialDeepONetDataset
from helper import load_model_experiment, convert2dim, train_val_test_split, plot_global_field_cartopy, plot_global_field_box

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


#------------------------------
# Load model
#------------------------------
model_path = '../../single_branch/lstm_window_30.pth'

model = load_model_experiment(model_path).to(device)

print(model)

#------------------------------
# Load test input function (unseen 2023 data)
# Load test target (resolution 1 deg)
#------------------------------
input_sensor = np.load('../../data/neutron_data_22yrs.npy')

# 1 degree target (scaled)
output_1deg = np.load('../../data/dose_array.npy')

# data splitting keeping the consistency with training phase
train_input, train_target, val_input, val_target, test_input, test_target = train_val_test_split(input_sensor, output_1deg)


# input scaler
scaler = MinMaxScaler()

dummy = scaler.fit_transform(train_input)
test_input = scaler.transform(test_input)

# target data normalization (min-max scaling)
scaler_target = MinMaxScaler()
train_target = scaler_target.fit_transform(train_target)[..., np.newaxis]
test_target = scaler_target.transform(test_target)[..., np.newaxis]


# location points for 1 degree target
trunk_1deg = np.load('../../data/grid_points.npy')

print('location range (1 deg):', np.min(trunk_1deg[:,0]), np.max(trunk_1deg[:,0]), np.min(trunk_1deg[:,1]), np.max(trunk_1deg[:,1]))

# Normalize trunk input
trunk_1deg[:, 0] = (trunk_1deg[:, 0] - np.min(trunk_1deg[:, 0])) / (np.max(trunk_1deg[:, 0]) - np.min(trunk_1deg[:, 0]))
trunk_1deg[:, 1] = (trunk_1deg[:, 1] - np.min(trunk_1deg[:, 1])) / (np.max(trunk_1deg[:, 1]) - np.min(trunk_1deg[:, 1]))

# Generate sequences for the testing set
test_input_seq, test_target_seq = create_sliding_windows(test_input, test_target, window_size=30)
print('test_input_seq shape:', test_input_seq.shape)
print('test_target_seq shape:', test_target_seq.shape)

# Create DataLoader for test set
test_dataset = SequentialDeepONetDataset(test_input_seq, trunk_1deg, test_target_seq)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def fit(model, data_loader, device):
    all_outputs = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for branch_batch, trunk_batch, target_batch in data_loader:
            branch_batch, trunk_batch, target_batch = (
                branch_batch.to(device),
                trunk_batch.to(device),
                target_batch.to(device),
            )
            output = model(branch_batch, trunk_batch)
            
            all_outputs.append(output.cpu())
            all_targets.append(target_batch.cpu())

    # ...existing code...
    # After loop:
    outputs = torch.cat(all_outputs, dim=0)  # [N_test, ...]
    targets = torch.cat(all_targets, dim=0)  # [N_test, ...]

    # move to numpy
    outputs = outputs.cpu().numpy()
    targets = targets.cpu().numpy()

    # flatten to 2D (n_samples, n_features) for scaler
    out_shape = outputs.shape
    tgt_shape = targets.shape
    outputs_flat = outputs.reshape(outputs.shape[0], -1)
    targets_flat = targets.reshape(targets.shape[0], -1)

    # inverse scale
    outputs_flat = scaler_target.inverse_transform(outputs_flat)
    targets_flat = scaler_target.inverse_transform(targets_flat)

    # restore original shapes
    outputs = outputs_flat.reshape(out_shape)
    targets = targets_flat.reshape(tgt_shape)
    # ...existing code...

    
    return outputs, targets

predictions_1deg, targets_1deg = fit(model, test_loader, device)
print('predictions shape:', predictions_1deg.shape)
print('targets shape:', targets_1deg.shape)

# convert to 2dim data and alighment for 1 degree grid
#_, _, pred_1deg = convert2dim(predictions_1deg)
#_, _, targ_1deg = convert2dim(targets_1deg)


lon_grid, lat_grid, pred_img = convert2dim(predictions_1deg)  # (N,H,W)
_,        _,        targ_img = convert2dim(targets_1deg)      # (N,H,W)


import numpy as np
from skimage.metrics import structural_similarity as ssim

pred_img = pred_img.astype(np.float64)
targ_img = targ_img.astype(np.float64)

N = pred_img.shape[0]
eps = 1e-12

# ---- Relative L2 (%) per-sample + mean ----
pf = pred_img.reshape(N, -1)
tf = targ_img.reshape(N, -1)
rel_l2 = np.linalg.norm(pf - tf, axis=1) / np.maximum(np.linalg.norm(tf, axis=1), eps)
rel_l2_pct = 100.0 * rel_l2
print("Mean rel L2 (%):", rel_l2_pct.mean())

# (optional) global relative L2 (%)
global_rel_l2_pct = 100.0 * np.linalg.norm(pf - tf) / (np.linalg.norm(tf) + eps)
print("Global rel L2 (%):", global_rel_l2_pct)

# ---- SSIM per-sample + mean (uses target’s per-sample dynamic range) ----
ssim_scores = np.empty(N, dtype=float)
for i in range(N):
    dr = float(targ_img[i].max() - targ_img[i].min()) or 1.0
    ssim_scores[i] = ssim(targ_img[i], pred_img[i], data_range=dr)
print("Mean SSIM:", ssim_scores.mean())

import cmocean
cmap_seq = cmocean.cm.thermal


# Example usage
# Atlantic-centered box (0°)
plot_global_field_box(
    lon_grid, lat_grid, pred_img, i=10,
    title="Prediction (orig units)",
    units_label="Effective Dose Rate (μSv/h)",
    cmap=cmap_seq, central_longitude=0,
    savepath="pred_box_0.png"
)

