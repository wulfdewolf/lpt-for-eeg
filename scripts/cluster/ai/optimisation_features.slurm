#!/bin/bash
#SBATCH --job-name=lpt-for-eeg
#SBATCH --time=5-00:00:00
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

# Underneath experiments all use the time-frequency features

# Optimisation run
python run.py --cluster /home/wdewolf/lpt-for-eeg \
              --wandb --name features-optimisation \
              --features \
              --optimise 50 \
              --pretrained-transformer --freeze-ff --freeze-attn --freeze-until 11

# No pretraining
python run.py --cluster /data/brussel/102/vsc10248 \
              --wandb --name features-random \
              --features \
              --batch-size X --epochs Y --learning-rate Z 

# Unfreezing layers
for layer in {-1..11}; do
    python run.py --cluster /home/wdewolf/lpt-for-eeg \
                  --wandb --name features-layers-${layer} \
                  --features \
                  --pretrained-transformer \
                  --batch-size X --epochs Y --learning-rate Z \
                  --freeze-ff --freeze-attn --freeze-until ${layer}
done