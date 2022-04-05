#!/bin/bash

# Make directory for raws
if [ ! -d data/motor_imagery ]; then
    mkdir data/motor_imagery
fi
if [ ! -d data/motor_imagery/raw ]; then
    mkdir data/motor_imagery/raw
fi

# Download training data
if [ ! -d data/motor_imagery/raw/training ]; then
    mkdir data/motor_imagery/raw/training
fi
if ! test -f data/motor_imagery/raw/training/A01T.mat; then
    wget -P data/motor_imagery/raw/training "http://bnci-horizon-2020.eu/database/data-sets/001-2014/A01T.mat"
fi
if ! test -f data/motor_imagery/raw/training/A02T.mat; then
    wget -P data/motor_imagery/raw/training "http://bnci-horizon-2020.eu/database/data-sets/001-2014/A02T.mat"
fi
if ! test -f data/motor_imagery/raw/training/A03T.mat; then
    wget -P data/motor_imagery/raw/training "http://bnci-horizon-2020.eu/database/data-sets/001-2014/A03T.mat"
fi
if ! test -f data/motor_imagery/raw/training/A04T.mat; then
    wget -P data/motor_imagery/raw/training "http://bnci-horizon-2020.eu/database/data-sets/001-2014/A04T.mat"
fi
if ! test -f data/motor_imagery/raw/training/A05T.mat; then
    wget -P data/motor_imagery/raw/training "http://bnci-horizon-2020.eu/database/data-sets/001-2014/A05T.mat"
fi
if ! test -f data/motor_imagery/raw/training/A06T.mat; then
    wget -P data/motor_imagery/raw/training "http://bnci-horizon-2020.eu/database/data-sets/001-2014/A06T.mat"
fi
if ! test -f data/motor_imagery/raw/training/A07T.mat; then
    wget -P data/motor_imagery/raw/training "http://bnci-horizon-2020.eu/database/data-sets/001-2014/A07T.mat"
fi
if ! test -f data/motor_imagery/raw/training/A08T.mat; then
    wget -P data/motor_imagery/raw/training "http://bnci-horizon-2020.eu/database/data-sets/001-2014/A08T.mat"
fi
if ! test -f data/motor_imagery/raw/training/A09T.mat; then
    wget -P data/motor_imagery/raw/training "http://bnci-horizon-2020.eu/database/data-sets/001-2014/A09T.mat"
fi

# Download evaluation data
if [ ! -d data/motor_imagery/raw/evaluation ]; then
    mkdir data/motor_imagery/raw/evaluation
fi
if ! test -f data/motor_imagery/raw/evaluation/A01E.mat; then
    wget -P data/motor_imagery/raw/evaluation "http://bnci-horizon-2020.eu/database/data-sets/001-2014/A01E.mat"
fi
if ! test -f data/motor_imagery/raw/evaluation/A02E.mat; then
    wget -P data/motor_imagery/raw/evaluation "http://bnci-horizon-2020.eu/database/data-sets/001-2014/A02E.mat"
fi
if ! test -f data/motor_imagery/raw/evaluation/A03E.mat; then
    wget -P data/motor_imagery/raw/evaluation "http://bnci-horizon-2020.eu/database/data-sets/001-2014/A03E.mat"
fi
if ! test -f data/motor_imagery/raw/evaluation/A04E.mat; then
    wget -P data/motor_imagery/raw/evaluation "http://bnci-horizon-2020.eu/database/data-sets/001-2014/A04E.mat"
fi
if ! test -f data/motor_imagery/raw/evaluation/A05E.mat; then
    wget -P data/motor_imagery/raw/evaluation "http://bnci-horizon-2020.eu/database/data-sets/001-2014/A05E.mat"
fi
if ! test -f data/motor_imagery/raw/evaluation/A06E.mat; then
    wget -P data/motor_imagery/raw/evaluation "http://bnci-horizon-2020.eu/database/data-sets/001-2014/A06E.mat"
fi
if ! test -f data/motor_imagery/raw/evaluation/A07E.mat; then
    wget -P data/motor_imagery/raw/evaluation "http://bnci-horizon-2020.eu/database/data-sets/001-2014/A07E.mat"
fi
if ! test -f data/motor_imagery/raw/evaluation/A08E.mat; then
    wget -P data/motor_imagery/raw/evaluation "http://bnci-horizon-2020.eu/database/data-sets/001-2014/A08E.mat"
fi
if ! test -f data/motor_imagery/raw/evaluation/A09E.mat; then
    wget -P data/motor_imagery/raw/evaluation "http://bnci-horizon-2020.eu/database/data-sets/001-2014/A09E.mat"
fi

# Make directory for weights
if [ ! -d data/weights ]; then
    mkdir data/weights
fi

# Download
if ! test -f data/weights/contextualizer.pt; then
    wget -P data/weights "https://github.com/SPOClab-ca/BENDR/releases/download/v0.1-alpha/contextualizer.pt"
fi
if ! test -f data/weights/encoder.pt; then
    wget -P data/weights "https://github.com/SPOClab-ca/BENDR/releases/download/v0.1-alpha/encoder.pt"
fi