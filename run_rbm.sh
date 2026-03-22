#!/bin/bash
#SBATCH --job-name=rbm_alpha_gpu
#SBATCH --output=logs/rbm_%j.out
#SBATCH --error=logs/rbm_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=ENSTA-h100

# module load python/3.10
# module load cuda/11.8
# source ~/envs/deeplearning/bin/activate

cd $SLURM_SUBMIT_DIR
mkdir -p logs

echo "Job $SLURM_JOB_ID  |  GPU: $CUDA_VISIBLE_DEVICES  |  $(date)"

python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python principal_RBM_alpha.py

echo "Done: $(date)"