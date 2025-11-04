#!/bin/bash
#SBATCH --job-name=single_tron
#SBATCH --partition=ghx4
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --account=begc-dtai-gh
#SBATCH -t 12:00:00
#SBATCH --output=multi_debug.out
#SBATCH --error=multi_debug.err

set -e   # stop immediately if any model crashes

module purge
cd $SLURM_SUBMIT_DIR

# Load correct CUDA & PyTorch
module load cuda/12.2.0
module load python/miniforge3_pytorch/2.5.0

eval "$(conda shell.bash hook)"
conda activate /u/kazumak2/.conda/envs/pytorch

python3 - <<EOF
import torch
print("CUDA:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
EOF

WINDOW_SIZES=(7 30 60 90)

echo "===== Starting GRU sweep ====="
for i in {0..3}; do
    GPU_ID=$i
    WINDOW_SIZE=${WINDOW_SIZES[$i]}
    echo "[GRU] window=$WINDOW_SIZE → GPU $GPU_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID python -u multi_gru_train.py \
        --window_size $WINDOW_SIZE &> multi_branch/logs/gru_w${WINDOW_SIZE}.log &
done
wait
echo "===== GRU sweep finished ====="


echo "===== Starting LSTM sweep ====="
for i in {0..3}; do
    GPU_ID=$i
    WINDOW_SIZE=${WINDOW_SIZES[$i]}
    echo "[LSTM] window=$WINDOW_SIZE → GPU $GPU_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID python -u multi_lstm_train.py \
        --window_size $WINDOW_SIZE &> multi_branch/logs/lstm_w${WINDOW_SIZE}.log &
done
wait
echo "===== LSTM sweep finished ====="

echo "===== ALL TRAINING COMPLETED SUCCESSFULLY ====="
