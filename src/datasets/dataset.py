from torch.utils.data import DataLoader
import numpy as np
from braindecode.datautil.preprocess import (
    preprocess,
    Preprocessor,
)
from braindecode.datasets import BaseConcatDataset


class Dataset:
    def __init__(
        self,
        task,
        seed,
        n_classes,
        n_channels,
        n_subjects,
        device="cpu",
        batch_size=1,
        window_size=None,
        window=False,
    ):

        self.device = device
        self.batch_size = batch_size
        self.window_size = window_size
        self.seed = seed
        self.task = task
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_subjects = n_subjects

        # !! Subclasses should implement __init__ to download and assign self.dataset

        # Select EEG
        preprocess(
            self.dataset,
            [
                Preprocessor("pick_types", eeg=True, meg=False),
            ],
        )

        # Cut windows
        if window:
            self.cut_windows()

    # Split into train and test set via fold ids
    def set_loaders(self, validation_subject):

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
        Fold random samplers 
        """
        #train_subsampler = torch.utils.data.RandomSampler()
        #test_subsampler = torch.utils.data.RandomSampler()

        """
        Data loaders
        """
        self.d_train = DataLoader(
            BaseConcatDataset(self.data_per_subject[:validation_subject] + self.data_per_subject[validation_subject+1:]),
            batch_size=self.batch_size,
            #sampler=train_subsampler,
            worker_init_fn=seed_worker,
            generator=g,
        )
        self.d_test = DataLoader(
            self.data_per_subject[validation_subject],
            batch_size=self.batch_size,
            #sampler=test_subsampler,
            worker_init_fn=seed_worker,
            generator=g,
        )

        self.train_enum = enumerate(self.d_train)
        self.test_enum = enumerate(self.d_test)

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
