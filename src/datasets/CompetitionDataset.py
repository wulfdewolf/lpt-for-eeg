import torch
from src.datasets.dataset import Dataset
import GPUtil


class CompetitionDataset(Dataset):
    def __init__(self, *args, **kwargs):
        from braindecode.datasets.moabb import MOABBDataset

        # Load data
        # !! downloads to ~/mne_data, this folder must exist
        self.dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[3])
        super().__init__(n_channels=22, classes=4, *args, **kwargs)

    # Preprocessing
    def process(self):
        from braindecode.datautil.preprocess import (
            preprocess,
            Preprocessor,
        )

        low_cut_hz = 4.0  # low cut frequency for filtering
        high_cut_hz = 38.0  # high cut frequency for filtering

        preprocessors = [
            # Preprocessor(lambda x: x * 1e6),  # Convert from V to uV
            Preprocessor(
                "filter", l_freq=low_cut_hz, h_freq=high_cut_hz
            ),  # Bandpass filter
        ]

        # Transform the data
        # preprocess(self.dataset, preprocessors)

    # Cutting compute windows
    def cut_windows(self):
        from braindecode.datautil.windowers import create_windows_from_events

        trial_start_offset_seconds = -0.5

        # Extract sampling frequency, check that they are same in all datasets
        sfreq = self.dataset.datasets[0].raw.info["sfreq"]
        assert all([ds.raw.info["sfreq"] == sfreq for ds in self.dataset.datasets])

        # Calculate the trial start offset in samples.
        trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

        # Create windows using braindecode function for this.
        self.windows = create_windows_from_events(
            self.dataset,
            trial_start_offset_samples=trial_start_offset_samples,
            trial_stop_offset_samples=0,
            window_size_samples=self.window_size,
            window_stride_samples=self.window_size,
            preload=True,
        )

        # Delete the raw dataset
        del self.dataset

    # Getting a single batch
    def get_batch(self, batch_size=None, train=True):
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

        return x, y
