from torch.utils.data import DataLoader
from braindecode.datautil.serialization import load_concat_dataset
import numpy as np
import os
from braindecode.datautil.preprocess import (
    preprocess,
    Preprocessor,
)


class Dataset:
    def __init__(
        self,
        device,
        task,
        batch_size,
        seed,
        data_dir,
        model_type,
        window_size=None,
    ):

        self.device = device
        self._ind = 0
        self.batch_size = batch_size
        self.window_size = window_size
        self.seed = seed
        self.data_dir = data_dir
        self.task = task

        # Map task to MOABB dataset name
        if task == "BCI_Competition_IV_2a":
            self.dataset_name = "BNCI2014001"
            self.classes = 4
        else:
            raise NotImplementedError(
                "The dataset (identifier) for this task has not been implemented!"
            )
        self.data_dir = os.path.join(data_dir, self.dataset_name, model_type)

        # Load data if it exists, otherwise download it
        try:
            self.dataset = load_concat_dataset(
                path=self.data_dir,
                preload=True,
            )
        except:
            self.download()

        # Cut MEG data
        preprocess(
            self.dataset,
            [
                Preprocessor("pick_types", eeg=True, meg=False, stim=False),
                Preprocessor("resample", sfreq=100),
            ],
        )

        # Preprocess
        self.process()

        # Cut windows
        self.cut_windows()

        # Cut into train and validation split
        self.split()

    # Split into train and validation set
    def split(self):
        splitted = self.windows.split("session")

        """
        Set worker seeds for reproducibility
        """
        import torch
        import random

        def seed_worker(worker_id):
            torch.manual_seed(self.seed)
            random.seed(self.seed)
            np.random.seed(self.seed)

        g = torch.Generator()
        g.manual_seed(self.seed)

        """
        Data loader
        """
        self.d_train = DataLoader(
            splitted["session_T"],
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        self.d_test = DataLoader(
            splitted["session_E"],
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

        # Store channels and classes
        self.n_channels = splitted["session_T"][0][0].shape[0]
        self.input_window_samples = splitted["session_T"][0][0].shape[1]

        self.train_enum = enumerate(self.d_train)
        self.test_enum = enumerate(self.d_test)

    # Download the dataset and store it
    def download(self):

        """
        Downloading dataset

        !! downloads to ~/mne_data, this folder must exist
        """
        from braindecode.datasets.moabb import MOABBDataset

        subject_id = 3
        self.dataset = MOABBDataset(
            dataset_name=self.dataset_name,
            subject_ids=[subject_id],
        )

        self.dataset.save(path=self.data_dir, overwrite=True)

    # Set _ind to 0
    def start_epoch(self):
        self._ind = 0

    """
    Visualising methods
    """

    def plot_raw(self, trial):
        self.dataset.datasets[trial].raw.plot(show_scrollbars=False).savefig(
            "plots/raw.pdf"
        )

    def plot_raw_interactive(self, trial):
        self.dataset.datasets[trial].raw.plot(block=True)

    def plot_windows_interactive(self, trial):
        print(len(self.windows.datasets[trial].windows))
        self.windows.datasets[trial].windows.plot(block=True)
