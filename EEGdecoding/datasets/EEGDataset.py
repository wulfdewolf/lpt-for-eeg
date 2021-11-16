from einops import rearrange
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

"""
Create Dataset object
"""
from EEGdecoding.datasets.dataset import Dataset


class EEGDataset(Dataset):
    def __init__(self, batch_size, seed, window_size=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.batch_size = batch_size
        self.window_size = window_size

        """
        Loading dataset
        """
        from braindecode.datasets.moabb import MOABBDataset

        subject_id = 3
        dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[subject_id])

        """
        Preprocessing
        """
        from braindecode.datautil.preprocess import (
            exponential_moving_standardize,
            preprocess,
            Preprocessor,
        )

        low_cut_hz = 4.0  # low cut frequency for filtering
        high_cut_hz = 38.0  # high cut frequency for filtering
        # Parameters for exponential moving standardization
        factor_new = 1e-3
        init_block_size = 1000

        preprocessors = [
            Preprocessor(
                "pick_types", eeg=True, meg=False, stim=False
            ),  # Keep EEG sensors
            Preprocessor(lambda x: x * 1e6),  # Convert from V to uV
            Preprocessor(
                "filter", l_freq=low_cut_hz, h_freq=high_cut_hz
            ),  # Bandpass filter
            Preprocessor(
                exponential_moving_standardize,  # Exponential moving standardization
                factor_new=factor_new,
                init_block_size=init_block_size,
            ),
        ]

        # Transform the data
        preprocess(dataset, preprocessors)

        """
        Cutting compute windows
        """
        import numpy as np
        from braindecode.datautil.windowers import create_windows_from_events

        trial_start_offset_seconds = -0.5

        # Extract sampling frequency, check that they are same in all datasets
        sfreq = dataset.datasets[0].raw.info["sfreq"]
        assert all([ds.raw.info["sfreq"] == sfreq for ds in dataset.datasets])

        # Calculate the trial start offset in samples.
        trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

        # Create windows using braindecode function for this.
        if window_size is not None:
            # Cropped
            windows_dataset = create_windows_from_events(
                dataset,
                trial_start_offset_samples=trial_start_offset_samples,
                trial_stop_offset_samples=0,
                window_size_samples=window_size,
                window_stride_samples=10,
                drop_last_window=False,
                preload=True,
            )
        else:
            # Trail-wise
            windows_dataset = create_windows_from_events(
                dataset,
                trial_start_offset_samples=trial_start_offset_samples,
                trial_stop_offset_samples=0,
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

    def get_batch(self, train=True):
        _, (x, y, _) = next(
            self.train_enum if train else self.test_enum, (None, (None, None, None))
        )

        if x is None:
            if train:
                self.train_enum = enumerate(self.d_train)
            else:
                self.test_enum = enumerate(self.d_test)
            _, (x, y, _) = next(self.train_enum if train else self.test_enum)

        x = x.to(device=self.device)
        y = y.to(device=self.device)

        self._ind += 1

        return x, y
