#!/bin/bash
#SBATCH --job-name=bonus_gen
#SBATCH --output=logs/bonus_%j.out
#SBATCH --error=logs/bonus_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=ENSTA-h100

cd $SLURM_SUBMIT_DIR
mkdir -p logs

echo "=============================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "GPU        : $CUDA_VISIBLE_DEVICES"
echo "Start time : $(date)"
echo "=============================="

# vérifier que le GPU est bien visible
python -c "
import torch
print(f'PyTorch  : {torch.__version__}')
print(f'CUDA ok  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU name : {torch.cuda.get_device_name(0)}')
    print(f'VRAM     : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
"

python bonus_generative_models.py

echo "=============================="
echo "End time : $(date)"
echo "=============================="