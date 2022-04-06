#!/bin/bash

# Make venv
python3 -m venv env

# Activate venv
source env/bin/activate

# Install reqs
pip3 install -r scripts/local/requirements.txt

# Install dn3
git clone git@github.com:wulfdewolf/dn3.git
cd dn3
python3 setup.py sdist
pip3 install .

# Cleanup
cd ..
rm -rf dn3