# Pretrained transformers as EEG decoders

This repository contains the source code for my master's thesis at the [Vrije Universiteit Brussel](https://www.vub.be).

Our contact information:
| Name | Email address | Linkedin | GitHub |
| :--- | :--- | :--- | :--- |
| Wolf De Wulf | [wolf.de.wulf@vub.be](mailto:wolf.de.wulf@vub.be) | https://www.linkedin.com/in/wolf-de-wulf/ | https://github.com/wulfdewolf |

## Installation

### 1. Installing requirements

Run the [`setup.sh`](requirements/local/setup.sh) script to create a virtual environment that has all
the required python packages installed:

```console
./requirements/local/setup.sh
```

Then, activate that environment:

```console
source env/bin/activate
```

To deactivate the virtual environment, use:

```console
deactivate
```

### 2. Data and pretraining weights

Run the [`download.sh`](data/download.sh) script to download the data and the pretraining weights:

```console
./data/download.sh
```

Run the [`process.py`](data/process.py) script to process the data:

```console
python data/process.py
```

## Usage

Run the [`experiments.sh`](experiment.sh) script to run the experiments:

```console
./experiments.sh
```
