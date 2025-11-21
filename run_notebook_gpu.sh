#!/bin/bash
# ==========================================================
# UNIVERSAL JUPYTER GPU JOB FOR DELTA / DELTA-AI
# ==========================================================

# Generate a random port number between 49152 and 59151
MYPORT=$(($(($RANDOM % 10000)) + 49152))
echo "Using port: $MYPORT"

# --- Ask user to select cluster ---
echo "Select cluster to launch Jupyter Notebook:"
echo "  1) DeltaAI (GH partition)"
echo "  2) Delta (A40/A100/H100 partition)"
read -p "Enter choice [1 or 2]: " CLUSTER_CHOICE

# --- Launch job based on selection ---
case $CLUSTER_CHOICE in
  1)
    echo "Launching on DeltaAI cluster..."
    module load python/miniforge3_pytorch
    srun --account=begc-dtai-gh \
         --partition=ghx4 \
         --gpus=1 \
         --time=08:00:00 \
         --mem=128g \
         jupyter-notebook --no-browser --port=$MYPORT --ip=0.0.0.0
    ;;
  2)
    echo "Launching on Delta cluster..."
    srun --account=bcnx-delta-gpu \
         --partition=gpuA40x4,gpuA100x4 \
         --nodes=1 \
         --ntasks-per-node=1 \
         --gpus=1 \
         --time=08:00:00 \
         --mem=128g \
         jupyter-notebook --no-browser --port=$MYPORT --ip=0.0.0.0
    ;;
  *)
    echo "Invalid choice. Exiting."
    exit 1
    ;;
esac
