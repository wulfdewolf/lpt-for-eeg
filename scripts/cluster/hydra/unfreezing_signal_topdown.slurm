#!/bin/bash
#SBATCH --job-name=lpt-for-eeg
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --partition=ampere_gpu,pascal_gpu
#SBATCH --exclusive

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

parallel --delay 0.2 -j 2 --joblog runtask.log --resume srun -N 1 -n 1 -c 8  --gpus=1 --mem=16G python run.py --cluster /data/brussel/102/vsc10248 \
                                                                                                              --wandb --name signal-unfreezing-from-{} \
                                                                                                              --pretrained-transformer \
                                                                                                              --batch-size 64 --epochs 32 --learning-rate 0.0004 \
                                                                                                              --freeze-ff --freeze-attn --freeze-between {} 12 ::: {2..12}
wait                                                                                        

# Final run where nothing is frozen
python run.py --cluster /data/brussel/102/vsc10248 \
              --wandb --name signal-unfreezing-0 \
              --pretrained-transformer \
              --batch-size 64 --epochs 32 --learning-rate 0.0004 \
              --freeze-ff --freeze-attn --freeze-between 0 0