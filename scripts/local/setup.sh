#!/bin/bash

# Make venv
python3 -m venv env

# Activate venv
source env/bin/activate

# Install reqs
pip3 install -r scripts/local/requirements.txt