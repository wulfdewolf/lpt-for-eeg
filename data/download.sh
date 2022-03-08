#!/bin/bash

# Make directory for raws
#mkdir data/competition
#mkdir data/competition/raw
#
## Download
#wget -P data/competition/raw/ "http://bnci-horizon-2020.eu/database/data-sets/001-2014/A01T.mat"
#wget -P data/competition/raw/ "http://bnci-horizon-2020.eu/database/data-sets/001-2014/A02T.mat"
#wget -P data/competition/raw/ "http://bnci-horizon-2020.eu/database/data-sets/001-2014/A03T.mat"
#wget -P data/competition/raw/ "http://bnci-horizon-2020.eu/database/data-sets/001-2014/A04T.mat"
#wget -P data/competition/raw/ "http://bnci-horizon-2020.eu/database/data-sets/001-2014/A05T.mat"
#wget -P data/competition/raw/ "http://bnci-horizon-2020.eu/database/data-sets/001-2014/A06T.mat"
#wget -P data/competition/raw/ "http://bnci-horizon-2020.eu/database/data-sets/001-2014/A07T.mat"
#wget -P data/competition/raw/ "http://bnci-horizon-2020.eu/database/data-sets/001-2014/A08T.mat"
#wget -P data/competition/raw/ "http://bnci-horizon-2020.eu/database/data-sets/001-2014/A09T.mat"

# Make directory for weights
mkdir data/weights

# Download
wget -P data/weights "https://github.com/SPOClab-ca/BENDR/releases/download/v0.1-alpha/contextualizer.pt"
wget -P data/weights "https://github.com/SPOClab-ca/BENDR/releases/download/v0.1-alpha/encoder.pt"