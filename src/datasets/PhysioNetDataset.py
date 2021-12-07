from src.datasets.dataset import Dataset


class PhysioNetDataset(Dataset):
    def __init__(self, *args, **kwargs):
        from braindecode.datasets.sleep_physionet import SleepPhysionet

        # Load data
        # !! downloads to ~/mne_data, this folder must exist
        self.dataset = SleepPhysionet(
            subject_ids=[0, 1, 2], recording_ids=[1], crop_wake_mins=30
        )
        super().__init__(classes=4, *args, **kwargs)

    # Preprocessing
    def process(self):
        from braindecode.datautil.preprocess import (
            preprocess,
            Preprocessor,
            scale,
        )

        high_cut_hz = 30

        preprocessors = [
            Preprocessor(scale, factor=1e6, apply_on_array=True),
            Preprocessor("filter", l_freq=None, h_freq=high_cut_hz, n_jobs=1),
        ]
        preprocess(self.dataset, preprocessors)

    # Cutting compute windows
    def cut_windows(self):
        from braindecode.datautil.windowers import create_windows_from_events
        from sklearn.preprocessing import scale as standard_scale
        from sklearn.preprocessing import (
            preprocess,
            Preprocessor,
        )

        # Extract sampling frequency, check that they are same in all datasets
        sfreq = self.dataset.datasets[0].raw.info["sfreq"]
        assert all([ds.raw.info["sfreq"] == sfreq for ds in self.dataset.datasets])

        window_size_s = 30
        window_size_samples = window_size_s * sfreq

        mapping = {  # We merge stages 3 and 4 following AASM standards.
            "Sleep stage W": 0,
            "Sleep stage 1": 1,
            "Sleep stage 2": 2,
            "Sleep stage 3": 3,
            "Sleep stage 4": 3,
            "Sleep stage R": 4,
        }

        self.windows = create_windows_from_events(
            self.dataset,
            trial_start_offset_samples=0,
            trial_stop_offset_samples=0,
            window_size_samples=window_size_samples,
            window_stride_samples=window_size_samples,
            preload=True,
            mapping=mapping,
        )
        preprocess(self.windows, [Preprocessor(standard_scale, channel_wise=True)])

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

        self._ind += 1

        return x, y
