#!/bin/bash
#SBATCH --job-name=single_tron
#SBATCH --partition=ghx4
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --account=begc-dtai-gh
#SBATCH -t 12:00:00

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

python -u train_ensemble.py &> ensemble_debug.log