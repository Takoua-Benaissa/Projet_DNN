#!/bin/bash
#SBATCH --job-name=dnn_mnist_gpu
#SBATCH --output=logs/dnn_mnist_%j.out
#SBATCH --error=logs/dnn_mnist_%j.err
#SBATCH --time=02:00:00          # ~10× faster than CPU version
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1             # ← request 1 GPU
#SBATCH --partition=ENSTA-h100

# ── load modules (adjust to your cluster) ────────────────────────────────────
# module load python/3.10
# module load cuda/11.8
# module load anaconda3

# ── activate environment ──────────────────────────────────────────────────────
# source ~/envs/deeplearning/bin/activate
# conda activate deeplearning

cd $SLURM_SUBMIT_DIR
mkdir -p logs

echo "=============================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "GPU        : $CUDA_VISIBLE_DEVICES"
echo "Start time : $(date)"
echo "=============================="

# verify GPU is visible
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"

python principal_DNN_MNIST.py

echo "=============================="
echo "End time : $(date)"
echo "=============================="