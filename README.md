# Pretrained transformers as EEG decoders

This repository contains a framework that allows for running a number of EEG classification tasks, each time using two types of models.
The first type of models consists of traditional EEG decoding models, while the second type of models are huge NLP models.
Research by [Lu et al.](https://arxiv.org/abs/2103.05247) has shown that these huge NLP models, or _Pretrained Transformers (PT)_, that have been trained on natural language can be generalized to other modalities with minimal finetuning.

The framework in this repository, which is the code companion to a master's thesis at the [VUB](https://www.vub.be/), was created to verify if said PT have any adaptability in the EEG decoding field.

## Installation

### 1. Virtual environment

Firstly, create a new virtual python environment:

```console
python -m venv env
```

Then, activate that environment:

```console
source env/bin/activate
```

Lastly, install the required libraries in the activated environment:

```console
pip install -r requirements.txt
```

To deactivate the virtual environment, use:

```console
deactivate
```

### 2. Environment variables

Add the `src` folder to the `PYTHONPATH` environment variable:

```console
export PYTHONPATH=/path/to/fpt-for-eeg:$PYTHONPATH
```

## Usage

To run the pipeline, use:

```console
python scripts/run.py -h
```

The code in the `run.py` file allows for setting up different types of experiments.
