#!/bin/bash

# Make directory for raw data
if [ ! -d data/raw ]; then
    mkdir data/raw
fi

# Download data per subject
for subject in {1..9}; do 
    if [ ! -d data/raw/subject${subject} ]; then
        mkdir data/raw/subject${subject}
    fi
    if ! test -f data/raw/subject${subject}/A0${subject}T.mat; then
        wget -P data/raw/subject${subject} "http://bnci-horizon-2020.eu/database/data-sets/001-2014/A0${subject}T.mat"
    fi
    if ! test -f data/raw/subject${subject}/A0${subject}E.mat; then
        wget -P data/raw/subject${subject} "http://bnci-horizon-2020.eu/database/data-sets/001-2014/A0${subject}E.mat"
    fi
done