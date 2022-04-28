#!/bin/bash
#SBATCH --job-name=lpt-for-eeg
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus=8

module load PyTorch/1.4.0-fosscuda-2019b-Python-3.7.4

# Cwd
cd ~/lpt-for-eeg

# Env
virtualenv --system-site-packages env
source env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r scripts/cluster/ai/requirements.txt

# WandB key
export WANDB_API_KEY=32c332a8b457162261635d4a8a940da285a355f0

python run.py --cluster /home/wdewolf/lpt-for-eeg \
              --wandb --name features-overfitting \
              --pretrained-transformer --freeze-ff --freeze-attn --freeze-until 11
              --batch-size X --epochs Y --learning-rate Z --decay 0.9