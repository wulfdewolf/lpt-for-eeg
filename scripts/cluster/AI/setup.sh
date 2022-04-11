#!/bin/bash

module load Python/3.7.4-GCCcore-8.3.0

# Make venv
python -m venv env

# Activate venv
source env/bin/activate

# Install reqs
pip install -r scripts/cluster/AI/requirements.txt

# Cleanup
module purge