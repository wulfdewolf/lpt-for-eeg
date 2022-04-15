#!/bin/bash
#SBATCH --job-name=fpt-for-eeg
#SBATCH --time=96:00:00
#SBTACH --gres=gpus:1
#SBATCH --mem=32G
#STBACH --cpus-per-task=20

module load PyTorch/1.7.1-fosscuda-2020b
module load wandb/0.12.5-GCCcore-10.2.0


# Cwd
cd ~/fpt-for-eeg

# Env
virtualenv --system-site-packages $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r scripts/cluster/AI/requirements.txt

# WandB key
export WANDB_API_KEY=32c332a8b457162261635d4a8a940da285a355f0

# Optimisation run for raw
#python run.py --cluster --wandb --optimise 100 \
#              --pretrained-transformer --freeze-ff --freeze-attn

# Optimisation run for features
python run.py --cluster --wandb --optimise 100 --features \
              --pretrained-transformer --freeze-ff --freeze-attn