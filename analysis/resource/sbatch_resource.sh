#!/bin/bash
#SBATCH --job-name=neural_operator
#SBATCH --partition=gpuA100x4            
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --account=bcnx-delta-cpu
#SBATCH --time=12:00:00

# Load modules
module purge
module load openmpi/4.1.6

# Activate Conda environment
eval "$(conda shell.bash hook)"
conda activate pytorch-env

# Change to project directory
python -u run_eval.py &> logs/eval_cpu.log

