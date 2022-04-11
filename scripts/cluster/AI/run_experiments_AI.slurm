#!/bin/bash
#SBATCH --job-name=fpt-for-eeg
#SBATCH --time=74:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mem=32G

module load PyTorch/1.4.0-fosscuda-2019b-Python-3.7.4

cd ~/fpt-for-eeg
source env/bin/activate
export WANDB_API_KEY=32c332a8b457162261635d4a8a940da285a355f0

python run.py --pretrained-transformer --freeze-until 5 --freeze-pos --freeze-ff
