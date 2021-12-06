import torch

"""
Create Dataset object
"""
from src.datasets.dataset import Dataset


class FPTDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(model_type="FPT", *args, **kwargs)

    # Preprocessing
    def process(self):
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
        preprocess(self.dataset, preprocessors)

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

        # Rearrange
        x = torch.transpose(x, 1, 2)

        x = x.to(device=self.device)
        y = y.to(device=self.device)

        self._ind += 1

        return x, y
