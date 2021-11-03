from einops import rearrange
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

"""
Create Dataset object
"""
from EEG_decoding.datasets.dataset import Dataset


class EEGDataset(Dataset):

    def __init__(self, batch_size, patch_size=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.batch_size = batch_size  # we fix it so we can use dataloader
        self.patch_size = patch_size  # grid of (patch_size x patch_size)

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
            exponential_moving_standardize, preprocess, Preprocessor)
        
        low_cut_hz = 4.  # low cut frequency for filtering
        high_cut_hz = 38.  # high cut frequency for filtering
        # Parameters for exponential moving standardization
        factor_new = 1e-3
        init_block_size = 1000
        
        preprocessors = [
            Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
            Preprocessor(lambda x: x * 1e6),  # Convert from V to uV
            Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
            Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                         factor_new=factor_new, init_block_size=init_block_size)
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
        sfreq = dataset.datasets[0].raw.info['sfreq']
        assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
        # Calculate the trial start offset in samples.
        trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

        # Create windows using braindecode function for this. It needs parameters to define how
        # trials should be used.
        windows_dataset = create_windows_from_events(
            dataset,
            trial_start_offset_samples=trial_start_offset_samples,
            trial_stop_offset_samples=0,
            preload=True,
        )

        """
        Split
        """
        splitted = windows_dataset.split('session')
        self.d_train = DataLoader(splitted['session_T'])
        self.d_test = DataLoader(splitted['session_E'])
        n_chans = self.d_train.dataset[0][0].shape[0]
        input_window_samples = self.d_test.dataset[0][0].shape[1]
        print(n_chans)
        print(input_window_samples)

        self.train_enum = enumerate(self.d_train)
        self.test_enum = enumerate(self.d_test)

    def get_batch(self, batch_size=None, train=True):
        if train:
            _, x = next(self.train_enum, (None, (None, None)))
            y = x[1]
            x = x[0]
            if x is None:
                self.train_enum = enumerate(self.d_train)
                _, (x, y) = next(self.train_enum)
        else:
            _, (x, y) = next(self.test_enum, (None, (None, None)))
            if x is None:
                self.test_enum = enumerate(self.d_test)
                _, (x, y) = next(self.test_enum)

        if self.patch_size is not None:
            x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)

        x = x.to(device=self.device)
        y = y.to(device=self.device)

        self._ind += 1

        return x, y
