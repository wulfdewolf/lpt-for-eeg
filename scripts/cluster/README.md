# Cluster specific scripts

The scripts in this folder are specific to the [cluster](https://hpc.vub.be/) that was used for this project.

## Installation

### 1. Virtual environment

The `make_env.sh` script should be ran from the VSC user's root folder. This script installs all the required libraries that are not available in pre-installed packages on Hydra.

## Usage

To run an experiment on the cluster, a [slurm](https://slurm.schedmd.com/documentation.html) task needs to be queued:

```console
sbatch cluster/run.pbs
```

The `run.pbs` file contains a slurm script that contains the following things:

- The allowed resources
- The setting up of environment variables
- The command that starts the experiment

Just as when running locally, the code in the `run.py` file allows for setting up different types of experiments.
Note that the command in the `run.pbs` file should contain the `-c` or `--cluster` option, to indicate that the experiment will run on the cluster.
