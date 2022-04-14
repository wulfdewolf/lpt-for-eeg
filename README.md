# Fine-tuning a language-pretrained transformer to the classification of electroencephalography

This repository contains the source code accompanying my master's thesis at the [Vrije Universiteit Brussel](https://www.vub.be).

Contact information:
| Name | Email address | Linkedin | GitHub |
| :--- | :--- | :--- | :--- |
| Wolf De Wulf | [wolf.de.wulf@vub.be](mailto:wolf.de.wulf@vub.be) | https://www.linkedin.com/in/wolf-de-wulf/ | https://github.com/wulfdewolf |

## Usage

### 1. Installing requirements

Run the [`setup.sh`](requirements/local/setup.sh) script to create a virtual environment that has all
the required python packages installed:

```console
./scripts/local/setup.sh
```

Then, activate that environment:

```console
source env/bin/activate
```

To deactivate the virtual environment, use:

```console
deactivate
```

### 2. Data

Run the [`download.sh`](data/download.sh) script to download the data:

```console
./data/download.sh
```

The raw data is downloaded to `data/raw`.

Run the [`process.py`](data/process.py) script to process the data:

```console
python data/process.py
```

The processed data is saved in `data/processed`.

Run the [`feature_extract.py`](data/feature_extract.py) script to process the data and extract features from it:

```console
python data/process.py
```

The features are saved in `data/feature_extracted`

### 3. Running

Run the [`run.py`](run.py) script to see what it can do:

```console
python run.py
```

## Reproducing the experiments

Run the [`run_experiments.sh`](scripts/local/run_experiments.sh) script to do all of the above and to run
the experiments:

```console
./scripts/local/run_experiments.sh
```

**Warning:** The experiments consist of hyperparameter optimisation runs, each of which run subject-wise cross-validation of a large deep learning model. Running the experiments on a device without a GPU is highly discouraged. Even for devices with a high-end GPU, running the experiments can take a long time and a lot of memory.
During research all experiments were ran on the [VUB Hydra cluster](https://hpc.vub.be/) using the [`run_experiments.slurm`](scripts/cluster/run_experiments.slurm) slurm script.
