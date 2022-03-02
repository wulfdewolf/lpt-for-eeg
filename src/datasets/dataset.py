from torch.utils.data import DataLoader
import numpy as np
import mne
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
        classes,
        n_channels,
        window_size=None,
        window=True,
        downsample=True,
    ):

        self.device = device
        self.batch_size = batch_size
        self.window_size = window_size
        self.seed = seed
        self.task = task
        self.classes = classes
        self.n_channels = n_channels

        # !! Subclasses should implement __init__ to download and assign self.dataset

        # Downsample
        if downsample:
            preprocess(
                self.dataset,
                [
                    Preprocessor("pick_types", eeg=True, meg=False),
                ],
            )
        self.process()

        # Cut windows
        if window:
            self.cut_windows()

    # Split into train and test set via fold ids
    def set_loaders(self, train_ids, test_ids):

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
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        """
        Data loaders
        """
        self.d_train = DataLoader(
            self.windows,
            batch_size=self.batch_size,
            sampler=train_subsampler,
            worker_init_fn=seed_worker,
            generator=g,
        )
        self.d_test = DataLoader(
            self.windows,
            batch_size=self.batch_size,
            sampler=test_subsampler,
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
