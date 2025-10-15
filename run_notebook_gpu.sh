#!/bin/bash
# ==========================================================
# JUPYTER GPU JOB FOR DELTA (SYSTEM JUPYTER, no conda activation)
# ==========================================================

# Generate a random port number between 49152 and 59151
MYPORT=$(($(($RANDOM % 10000))+49152))
echo "Using port: $MYPORT"

# --- Load the system module for Jupyter ---
module load python/miniforge3_pytorch

srun --account=begc-dtai-gh --partition=ghx4 --gpus=1 --time=10:00:00 --mem=32g jupyter-notebook --no-browser --port=$MYPORT --ip=0.0.0.0
