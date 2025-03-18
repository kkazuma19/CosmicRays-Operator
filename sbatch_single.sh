#!/bin/bash
#SBATCH --job-name=single_lstm_gru
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --gpus-per-node=4
#SBATCH --account=bcnx-delta-gpu
#SBATCH -t 6:00:00

# Load necessary modules
conda deactivate
module purge
module load anaconda3_gpu

# Activate the Conda environment
source activate /u/kazumak2/.conda/envs/pytorch-env

# Print active Conda environment
conda info --envs


# Define different window sizes
WINDOW_SIZES=(7 30 60 90)

# Launch jobs in parallel for different window sizes
for i in {0..3}; do
    GPU_ID=$i
    WINDOW_SIZE=${WINDOW_SIZES[$i]}

    echo "Starting job for window_size=${WINDOW_SIZE} on GPU ${GPU_ID}"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python single_gru_train.py --window_size $WINDOW_SIZE > logs/single_gru_window_${WINDOW_SIZE}.log 2>&1 &
    #CUDA_VISIBLE_DEVICES=$GPU_ID python single_fnn_train.py --window_size $WINDOW_SIZE > logs/single_fnn_window_${WINDOW_SIZE}.log 2>&1 &
done

# Wait for all background processes to finish
#wait

for i in {0..3}; do
    GPU_ID=$i
    WINDOW_SIZE=${WINDOW_SIZES[$i]}

    echo "Starting job for window_size=${WINDOW_SIZE} on GPU ${GPU_ID}"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python single_lstm_train.py --window_size $WINDOW_SIZE > logs/single_lstm_window_${WINDOW_SIZE}.log 2>&1 &
done


for i in {0..3}; do
    GPU_ID=$i
    WINDOW_SIZE=${WINDOW_SIZES[$i]}

    echo "Starting job for window_size=${WINDOW_SIZE} on GPU ${GPU_ID}"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python single_fnn_train.py --window_size $WINDOW_SIZE > logs/single_fnn_window_${WINDOW_SIZE}.log 2>&1 &
done


# Wait for all background processes to finish
wait

echo "All training jobs completed."