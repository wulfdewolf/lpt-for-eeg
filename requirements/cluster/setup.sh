#!/bin/bash

module load Python/3.8.6-GCCcore-10.2.0

# Install reqs
pip install --user -r requirements/cluster/requirements.txt

# Install dn3
git clone git@github.com:SPOClab-ca/dn3.git
cd dn3
sed -i '106d' dn3/utils.py
python3 setup.py sdist
pip3 install --user .

# Cleanup
cd ..
rm -rf dn3