import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import sys
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

# Add src path for module imports
src_path = os.path.abspath(os.path.join(os.getcwd(), 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

from utils import create_sliding_windows, SequentialDeepONetDataset
from s_deeponet import SequentialDeepONet

# ------------------------------
# 🖥️ HPC Argument Parsing
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--window_size", type=int, required=True, help="Size of the sliding window")
args = parser.parse_args()

window_size = args.window_size
print(f"Training with window_size={window_size}")

# ------------------------------
# 🌎 Device Configuration
# ------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


# ------------------------------
# 📂 Load and Normalize Data
# ------------------------------
input_data = np.load('data/neutron_data_22yrs.npy')
trunk = np.load('data/grid_points.npy')
target = np.load('data/dose_array.npy')

# Normalize trunk input
trunk[:, 0] = (trunk[:, 0] - np.min(trunk[:, 0])) / (np.max(trunk[:, 0]) - np.min(trunk[:, 0]))
trunk[:, 1] = (trunk[:, 1] - np.min(trunk[:, 1])) / (np.max(trunk[:, 1]) - np.min(trunk[:, 1]))

# Hold out the last 365 samples for testing
test_size = 365
train_val_input = input_data[:-test_size]
train_val_target = target[:-test_size]
test_input = input_data[-test_size:]
test_target = target[-test_size:]

# Normalize input and target data
scaler = MinMaxScaler()
train_val_input = scaler.fit_transform(train_val_input)
test_input = scaler.transform(test_input)

scaler_target = MinMaxScaler()
train_val_target = scaler_target.fit_transform(train_val_target)[..., np.newaxis]
test_target = scaler_target.transform(test_target)[..., np.newaxis]

# ------------------------------
# 🔄 K-Fold Cross-Validation
# ------------------------------
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True)

# Store all fold metrics
all_rmse, all_mae, all_r2, all_l2_error = [], [], [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_input)):
    print(f"\n--- Fold {fold+1}/{k_folds} ---")

    # Split data into train and validation sets
    train_input, val_input = train_val_input[train_idx], train_val_input[val_idx]
    train_target, val_target = train_val_target[train_idx], train_val_target[val_idx]

    # Create sliding window sequences
    train_input_seq, train_target_seq = create_sliding_windows(train_input, train_target, window_size)
    val_input_seq, val_target_seq = create_sliding_windows(val_input, val_target, window_size)

    # Create DataLoaders
    batch_size = 16
    train_dataset = SequentialDeepONetDataset(train_input_seq, trunk, train_target_seq)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    val_dataset = SequentialDeepONetDataset(val_input_seq, trunk, val_target_seq)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ------------------------------
    # 🏗 Initialize Model
    # ------------------------------
    model = SequentialDeepONet(
        branch_type='gru',
        branch_input_size=12,
        branch_hidden_size=128,
        branch_num_layers=4,
        branch_output_size=128,
        trunk_architecture=[2, 128, 128, 128],
        num_outputs=1,
        use_transform=False,
        activation_fn=nn.ReLU,
    ).to(device)

    # ------------------------------
    # 🔧 Training Configuration
    # ------------------------------
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 1000
    patience = 5
    best_val_loss = np.inf
    counter = 0

    # ------------------------------
    # 🚀 Training Loop
    # ------------------------------
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

        # Validation step
        model.eval()
        val_loss = 0.0
        for inputs, trunk, targets in val_loader:
            inputs, trunk, targets = inputs.to(device), trunk.to(device), targets.to(device)
            outputs = model(inputs, trunk)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Fold {fold+1}, Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Save best model for this fold
    save_dir = 'single_branch/'
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, f'gru_model_fold{fold+1}.pth')
    torch.save(best_model_state, model_save_path)
    print(f"Best model for fold {fold+1} saved at {model_save_path}")