from torch.utils.data import DataLoader
from braindecode.datautil.serialization import load_concat_dataset
import numpy as np
import matplotlib.pyplot as plt
import mne
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
        classes,
        window_size=None,
        process=True,
        window=True,
    ):

        self.device = device
        self._ind = 0
        self.batch_size = batch_size
        self.window_size = window_size
        self.seed = seed
        self.data_dir = data_dir
        self.task = task
        self.classes = classes
        self.model_type = model_type
        self.data_dir = os.path.join(data_dir, self.task)

        # !! Subclasses should implement __init__ to download and assign self.dataset

        # Keep only EEG data and downsample
        preprocess(
            self.dataset,
            [
                Preprocessor("pick_types", eeg=True, meg=False),
                Preprocessor("resample", sfreq=100),
            ],
        )

        # Preprocess
        if process:
            self.process()

        # Cut windows
        if window:
            self.cut_windows()
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

    # Set _ind to 0
    def start_epoch(self):
        self._ind = 0

    """
    Visualising methods
    """

    def plot_raw(self, trial, name):
        self.dataset.datasets[trial].raw.plot(show_scrollbars=False).savefig(
            "plots/" + name + ".pdf", bbox_inches="tight"
        )

    def plot_raw_interactive(self, trial):
        self.dataset.datasets[trial].raw.plot(block=True)

    def plot_windows(self, trial, name):
        events = mne.pick_events(
            self.windows.datasets[trial].windows.events, include=[1, 2, 3]
        )
        self.windows.datasets[trial].windows["tongue"].plot(
            events=events,
            show_scrollbars=False,
            event_id={"left_hand": 1, "right_hand": 2, "tongue": 3},
            event_color=dict(tongue="r", left_hand="b", right_hand="k"),
        ).savefig("plots/" + name + ".pdf", bbox_inches="tight")

    def plot_windows_interactive(self, trial):
        events = mne.pick_events(
            self.windows.datasets[trial].windows.events, include=[1, 2, 3]
        )
        self.windows.datasets[trial].windows["tongue"].plot(
            events=events,
            block=True,
            show_scrollbars=True,
            # event_id={"left_hand": 1, "right_hand": 2, "tongue": 3},
            event_color="blue",  # dict(tongue="r", left_hand="b", right_hand="k"),
        )

    def plot_psd(self, trial, name):
        self.dataset.datasets[trial].raw.plot_psd(picks="eeg").savefig(
            "plots/" + name + ".pdf", bbox_inches="tight"
        )
