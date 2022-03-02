#!/bin/bash

module load Python/3.8.6-GCCcore-10.2.0

# Install reqs
pip install --user -r ~/fpt-for-eeg/cluster/cluster_requirements.txt
git clone git@github.com:SPOClab-ca/dn3.git
cd dn3
python3 setup.py sdist
pip install --user .
rm -rf dn3

