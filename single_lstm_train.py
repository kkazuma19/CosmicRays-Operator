# single branch GRU training script
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import argparse
import sys
import os

src_path = os.path.abspath(os.path.join(os.getcwd(), 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)
    
from utils import create_sliding_windows, SequentialDeepONetDataset
from s_deeponet import SequentialDeepONet

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


import random

#seed = 1
#random.seed(seed)
#np.random.seed(seed)
#torch.manual_seed(seed)
#torch.cuda.manual_seed_all(seed)

# %%
# Load neutron monitoring data
input_data = np.load('data/neutron_data_22yrs.npy')
trunk = np.load('data/grid_points.npy')
target = np.load('data/dose_array.npy')

# Normalize trunk input
trunk[:, 0] = (trunk[:, 0] - np.min(trunk[:, 0])) / (np.max(trunk[:, 0]) - np.min(trunk[:, 0]))
trunk[:, 1] = (trunk[:, 1] - np.min(trunk[:, 1])) / (np.max(trunk[:, 1]) - np.min(trunk[:, 1]))

# %%
def train_val_test_split(input_data, target):
    # Define the number of test samples (last 365 days)
    test_size = 365

    # Split data into training+validation and test
    train_val_input = input_data[:-test_size]
    train_val_target = target[:-test_size]
    test_input = input_data[-test_size:]
    test_target = target[-test_size:]

    # Calculate split index for training and validation
    train_size = int(len(train_val_input) * 0.5)  
    val_size = len(train_val_input) - train_size  

    # Training set
    train_input = train_val_input[:train_size]
    train_target = train_val_target[:train_size]

    # Validation set
    val_input = train_val_input[train_size:]
    val_target = train_val_target[train_size:]

    # Final shapes check
    print("Train input shape:", train_input.shape)
    print("Validation input shape:", val_input.shape)
    print("Test input shape:", test_input.shape)

    return train_input, train_target, val_input, val_target, test_input, test_target

# Assuming input_data and target are defined elsewhere in the notebook
train_input, train_target, val_input, val_target, test_input, test_target = train_val_test_split(input_data, target)

# %%
# input data normalization (min-max scaling)
scaler = MinMaxScaler()

# Add Gaussian noise directly to the test input
noise_level = 0.05  # Set noise level (e.g., 10% noise)
noisy_test_input = test_input * (1 +  np.random.normal(0, noise_level, test_input.shape) )

train_input = scaler.fit_transform(train_input)
val_input = scaler.transform(val_input)
test_input = scaler.transform(test_input)
noisy_test_input = scaler.transform(noisy_test_input)  # Normalizing noisy input using the same scaler

# target data normalization (min-max scaling)
scaler_target = MinMaxScaler()
train_target = scaler_target.fit_transform(train_target)[..., np.newaxis]
val_target = scaler_target.transform(val_target)[..., np.newaxis]
test_target = scaler_target.transform(test_target)[..., np.newaxis]


# %%
parser = argparse.ArgumentParser()
parser.add_argument("--window_size", type=int, required=True, help="Size of the sliding window")
args = parser.parse_args()

# Use args.window_size in your training pipeline
window_size = args.window_size
print(f"Training with window_size={window_size}")

#window_size = 7  # 7 days window size
#print("Window size:", window_size)

# Generate sequences for the training set
train_input_seq, train_target_seq = create_sliding_windows(train_input, train_target, window_size)

# Generate sequences for the testing set
test_input_seq, test_target_seq = create_sliding_windows(test_input, test_target, window_size)

# Generate sequences for the noisy testing set
noisy_test_input_seq, test_target_seq = create_sliding_windows(noisy_test_input, test_target, window_size)

# generate sequences for the validation set
val_input_seq, val_target_seq = create_sliding_windows(val_input, val_target, window_size)


# print the shapes of the generated sequences
print("Check the shapes of the generated sequences\n-----------------------------------------")
print("Train input shape:", train_input_seq.shape)
print("Train target shape:", train_target_seq.shape)
print("Validation input shape:", val_input_seq.shape)
print("Validation target shape:", val_target_seq.shape)
print("Test input shape:", test_input_seq.shape)
print("Test target shape:", test_target_seq.shape)
print("-----------------------------------------")


# %%
# Create DataLoaders for training and validation sets
print("Create DataLoaders for training and validation sets\n-----------------------------------------")
batch_size = 16
print("Batch size:", batch_size)

train_dataset = SequentialDeepONetDataset(train_input_seq, trunk, train_target_seq)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

val_dataset = SequentialDeepONetDataset(val_input_seq, trunk, val_target_seq)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = SequentialDeepONetDataset(test_input_seq, trunk, test_target_seq)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# noizy
noisy_dataset = SequentialDeepONetDataset(noisy_test_input_seq, trunk, test_target_seq)
noisy_loader = torch.utils.data.DataLoader(noisy_dataset, batch_size=batch_size, shuffle=False)


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
save_path = os.path.join(save_dir, f'lstm_window_{window_size}_1.pth')
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
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
#print("Optimizer:", optimizer)
print("-----------------------------------------")


# %%
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, save_path):
    best_val_loss = np.inf
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for i, (inputs, trunk, targets) in enumerate(train_loader):
            inputs, trunk, targets = inputs.to(device), trunk.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, trunk)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        for i, (inputs, trunk, targets) in enumerate(val_loader):
            inputs, trunk, targets = inputs.to(device), trunk.to(device), targets.to(device)
            outputs = model(inputs, trunk)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")

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
#train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, save_path)
# %% load the model from the saved path
#model = init_model().to(device)
#model.load_state_dict(torch.load(save_path))
#model.eval()

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
    
    print("All predictions shape before reshape:", all_preds.shape)
    print("All targets shape before reshape:", all_targets.shape)

    # Reshape to 2D (n_samples, n_features) for inverse scaling
    all_preds = all_preds.reshape(all_preds.shape[0], -1)
    all_targets = all_targets.reshape(all_targets.shape[0], -1)

    print("All predictions shape after reshape:", all_preds.shape)
    print("All targets shape after reshape:", all_targets.shape)
    
    # Inverse scaling
    all_preds = scaler.inverse_transform(all_preds)
    all_targets = scaler.inverse_transform(all_targets)
    
    # save the predictions and targets to a file together
    save_path = os.path.join(save_dir, f'array/lstm_window_{window_size}_preds_targets_1.npy')
    np.save(save_path, np.stack((all_preds, all_targets), axis=1))
    print(f"Predictions and targets saved to {save_path}")
    
    # Compute metrics for each sample
    rmse, mae, r2, l2_error = [], [], [], []
    for i in range(all_preds.shape[0]):
        rmse.append(np.sqrt(np.mean((all_preds[i] - all_targets[i]) ** 2)))
        mae.append(np.mean(np.abs(all_preds[i] - all_targets[i])))
        r2.append(1 - np.sum((all_preds[i] - all_targets[i]) ** 2) / np.sum((all_targets[i] - np.mean(all_targets[i])) ** 2))
        l2_error.append(np.linalg.norm(all_preds[i] - all_targets[i], 2))

    # Convert lists to numpy arrays
    rmse = np.array(rmse)
    mae = np.array(mae)
    r2 = np.array(r2)
    l2_error = np.array(l2_error)
    
    # save the results to a file
    results = np.stack((rmse, mae, r2, l2_error), axis=1)
    save_path = os.path.join(save_dir, f'array/lstm_window_{window_size}_results_1.npy')
    np.save(save_path, results)
    print(f"Results saved to {save_path}")
    
    # Compute average metrics
    rmse = np.mean(rmse)
    mae = np.mean(mae)
    r2 = np.mean(r2)
    l2_error = np.mean(l2_error)

    print(f"Final Model Evaluation on Test Set:")
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, L2 Error: {l2_error:.4f}")

    return rmse, mae, r2, l2_error

# %%
#print(evaluate_model(model, test_loader, scaler_target, device=device))



def benchmark_training(model_fn, train_loader, val_loader, criterion, optimizer_fn, num_epochs=1000, patience=10, device='cuda', repeats=5):
    import time

    all_times = []

    for repeat in range(repeats):
        print(f"\n--- Repetition {repeat+1}/{repeats} ---")
        model = model_fn().to(device)
        optimizer = optimizer_fn(model)

        best_val_loss = np.inf
        counter = 0
        total_samples = len(train_loader.dataset)

        start_time = time.time()

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

            model.eval()
            val_loss = 0.0
            for inputs, trunk, targets in val_loader:
                inputs, trunk, targets = inputs.to(device), trunk.to(device), targets.to(device)
                outputs = model(inputs, trunk)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
            val_loss /= len(val_loader)

            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        end_time = time.time()
        total_time = end_time - start_time
        time_per_sample = total_time / (len(train_loader.dataset) * (epoch + 1))

        print(f"Repetition {repeat+1} completed: Total time = {total_time:.2f}s, Time/sample = {time_per_sample:.6f}s")
        all_times.append((total_time, time_per_sample))

    print("\n====== Benchmark Summary ======")
    for i, (t, s) in enumerate(all_times):
        print(f"Run {i+1}: Total = {t:.2f}s, Per Sample = {s:.6f}s")

    avg_total = np.mean([t[0] for t in all_times])
    avg_sample = np.mean([t[1] for t in all_times])
    print(f"\nAverage Total Time: {avg_total:.2f}s")
    print(f"Average Time per Sample: {avg_sample:.6f}s")

benchmark_training(
    model_fn=init_model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=nn.MSELoss(),
    optimizer_fn=lambda model: optim.Adam(model.parameters(), lr=1e-3),
    num_epochs=1000,
    patience=10,
    device=device,
    repeats=5
)
