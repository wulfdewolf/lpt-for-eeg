#!/bin/bash

# Make venv
python3 -m venv env

# Activate venv
source env/bin/activate

# Install reqs
pip3 install -r requirements.txt
git clone git@github.com:SPOClab-ca/dn3.git
cd dn3
python3 setup.py sdist
pip3 install .

# Cleanup
cd ..
rm -rf dn3

# Add to pythonpath
export $PYTHONPATH=/home/wolf/fpt-for-eeg:$PYTHONPATH