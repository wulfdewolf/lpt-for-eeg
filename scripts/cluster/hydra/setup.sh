#!/bin/bash

module load Python/3.7.4-GCCcore-8.3.0

# Install reqs
pip install --user -r requirements/cluster/hydra/requirements.txt

# Cleanup
module purge