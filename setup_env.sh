#!/bin/bash
#SBATCH --job-name=setup_gpu_env
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=ENSTA-h100

mkdir -p logs

echo "Creating virtual environment with PyTorch GPU support…"
python -m venv ~/envs/deeplearning
source ~/envs/deeplearning/bin/activate

pip install --upgrade pip

# ── Install PyTorch with CUDA support ────────────────────────────────────────
# Choose the right version for your cluster's CUDA version.
# Check with: nvcc --version  or  nvidia-smi

# CUDA 12.1  (most common on modern clusters)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8  (older clusters) — uncomment if needed:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU-only fallback — uncomment if no GPU:
# pip install torch torchvision

# ── other dependencies ────────────────────────────────────────────────────────
pip install numpy scipy matplotlib

echo ""
echo "Checking installation…"
python -c "
import torch
print(f'PyTorch  : {torch.__version__}')
print(f'CUDA ok  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU name : {torch.cuda.get_device_name(0)}')
    print(f'VRAM     : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
import numpy, scipy, matplotlib
print('numpy / scipy / matplotlib : OK')
"
echo "Setup complete."