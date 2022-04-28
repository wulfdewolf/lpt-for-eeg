#!/bin/bash
#SBATCH --job-name=lpt-for-eeg
#SBATCH --time=2-00:00:00
#SBATCH --gpus=1
#SBATCH --partition=ampere_gpu,pascal_gpu
#SBATCH --mem=16G
#SBATCH --cpus-per-gpu=8

module load PyTorch/1.7.1-fosscuda-2020b
module load wandb/0.12.5-GCCcore-10.2.0


# Cwd
cd ~/lpt-for-eeg

# Env
virtualenv --system-site-packages /tmp/env
source /tmp/env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r scripts/cluster/hydra/requirements.txt

# WandB key
export WANDB_API_KEY=32c332a8b457162261635d4a8a940da285a355f0

python run.py --cluster /data/brussel/102/vsc10248 \
              --wandb --name signal-random \
              --batch-size X --epochs Y --learning-rate Z 