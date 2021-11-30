from torch.utils.data import DataLoader
from braindecode.datautil.serialization import load_concat_dataset
import numpy as np
import os


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
        self.data_dir = data_dir
        self.task = task

        # Map task to MOABB dataset name
        if task == "BCI_Competition_IV_2a":
            self.dataset_name = "BNCI2014001"
            self.data_dir = os.path.join(data_dir, self.dataset_name, model_type)
            self.classes = 4
        else:
            raise NotImplementedError(
                "The dataset (identifier) for this task has not been implemented!"
            )

        # Load processed data if it exists
        try:
            windows_dataset = load_concat_dataset(
                path=self.data_dir,
                preload=True,
            )
        except:
            self.download()
            self.process()
            windows_dataset = load_concat_dataset(
                path=self.data_dir,
                preload=True,
            )

        """
        Split
        """
        splitted = windows_dataset.split("session")

        """
        Set worker seeds for reproducibility
        """
        import torch
        import random

        def seed_worker(worker_id):
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        g = torch.Generator()
        g.manual_seed(seed)

        """
        Data loader
        """
        self.d_train = DataLoader(
            splitted["session_T"],
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        self.d_test = DataLoader(
            splitted["session_E"],
            batch_size=batch_size,
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

    def download(self):

        """
        Downloading dataset

        !! downloads to ~/mne_data, after calling the process function once, this folder can be removed
        """
        from braindecode.datasets.moabb import MOABBDataset

        subject_id = 3
        self.dataset = MOABBDataset(
            dataset_name=self.dataset_name,
            subject_ids=[subject_id],
        )

    def start_epoch(self):
        self._ind = 0
