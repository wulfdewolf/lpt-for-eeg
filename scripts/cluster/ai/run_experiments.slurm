#!/bin/bash
#SBATCH --job-name=fpt-for-eeg
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus=16

module load PyTorch/1.4.0-fosscuda-2019b-Python-3.7.4

# Cwd
cd ~/fpt-for-eeg

# Env
virtualenv --system-site-packages env
source env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r scripts/cluster/ai/requirements.txt

# WandB key
export WANDB_API_KEY=32c332a8b457162261635d4a8a940da285a355f0

# Optimisation run for features
python run.py --cluster /home/wdewolf/fpt-for-eeg --wandb --optimise 50 --features \
              --pretrained-transformer --freeze-ff --freeze-attn --freeze-until 11