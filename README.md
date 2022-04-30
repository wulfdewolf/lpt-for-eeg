# Transfer Learning in BCI's: Language-Pretrained Transformers for EEG Classification

This repository contains the source code accompanying my master's thesis at the [Vrije Universiteit Brussel](https://www.vub.be).

Contact information:
| Name | Email address | Linkedin | GitHub |
| :--- | :--- | :--- | :--- |
| Wolf De Wulf | [wolf.de.wulf@vub.be](mailto:wolf.de.wulf@vub.be) / [wolf.de.wulf@ed.ac.uk](mailto:wolf.de.wulf@ed.ac.uk) | https://www.linkedin.com/in/wolf-de-wulf/ | https://github.com/wulfdewolf |

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

The processed data is saved in `data/processed`, the features are saved in `data/feature_extracted`

### 3. Running

Run the [`run.py`](run.py) script to see what it can do:

```console
python run.py --help
```

## Reproducing the empirical evaluations

The scripts that were used to produce the results presented in the thesis can be found in the [`scripts/cluster`](scripts/cluster) folder.  
A summary of the results can be found on [Weights & Biases](https://wandb.ai/wulfdewolf/lpt-for-eeg/reports/Transfer-learning-in-BCI-s-language-pretrained-transformers-for-EEG-classification--VmlldzoxOTIxNDU2?accessToken=r4hzxv3i86ovxcf01fdzcebnnpy79nc57stoew4gasvoboual6f2c93131ra4u1z).

**Warning:**  
The evaluations consist of hyperparameter optimisation runs, each of which run subject-wise cross-validation of a large deep learning model.   
Running the evaluations on a device without a GPU is highly discouraged.  
Even for devices with a high-end GPU, running them can take a long time and a lot of memory.  
During research all evaluations were ran on the [VUB Hydra HPC](https://hpc.vub.be/) and the [VUB AI lab HPC](https://comopc3.vub.ac.be/).

If for some reason you want the results in `.csv` format or if you have questions, feel free to contact me via mail.