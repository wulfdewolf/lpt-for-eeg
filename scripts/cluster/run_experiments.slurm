#!/bin/bash
#SBATCH --job-name=fpt-for-eeg
#SBATCH --time=120:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus=2

module load PyTorch/1.7.1-fosscuda-2020b
module load wandb/0.12.5-GCCcore-10.2.0


# Cwd
cd ~/fpt-for-eeg

# Env
virtualenv --system-site-packages /tmp/env
source /tmp/env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r scripts/cluster/requirements.txt

# WandB key
export WANDB_API_KEY=32c332a8b457162261635d4a8a940da285a355f0

# Optimisation run for raw
#python run.py --cluster --wandb --optimise 40 \
#              --pretrained-transformer --freeze-ff --freeze-attn

# Optimisation run for features
python run.py --cluster --wandb --optimise 40 --features \
              --pretrained-transformer --freeze-ff --freeze-attn