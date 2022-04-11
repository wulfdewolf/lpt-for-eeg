#!/bin/bash

module load PyTorch/1.4.0-fosscuda-2019b-Python-3.7.4

# Install reqs
pip install --user -r scripts/cluster/AI/requirements.txt
pip install --user --ignore-installed six

# Cleanup
module purge