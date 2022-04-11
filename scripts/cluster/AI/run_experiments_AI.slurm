#!/bin/bash
#SBATCH --job-name=fpt-for-eeg
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=7G

module load PyTorch/1.4.0-fosscuda-2019b-Python-3.7.4

cd ~/fpt-for-eeg
export WANDB_API_KEY=32c332a8b457162261635d4a8a940da285a355f0

python run.py --pretrained-transformer