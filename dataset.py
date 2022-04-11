import torch
import os
import mne
import numpy
import random


class EpochsDataset(torch.utils.data.Dataset):
    """Class to expose an MNE Epochs object as PyTorch dataset
    Parameters
    ----------
    epochs_data : 3d array, shape (n_epochs, n_channels, n_times)
        The epochs data.
    epochs_labels : array of int, shape (n_epochs,)
        The epochs labels.
    transform : callable | None
        The function to eventually apply to each epoch
        for preprocessing (e.g. scaling). Defaults to None.
    """

    def __init__(self, epochs_data, epochs_labels):
        assert len(epochs_data) == len(epochs_labels)
        self.epochs_data = epochs_data
        self.epochs_labels = epochs_labels

    def to(self, device):
        self.device = device

        # Send complete dataset to device
        self.epochs_data = torch.as_tensor(
            self.epochs_data, device=device, dtype=torch.float32
        )
        self.epochs_labels = torch.as_tensor(
            self.epochs_labels, device=device, dtype=torch.long
        )

    def get_batch(self, indices):
        return self.epochs_data.index_select(
            0, torch.as_tensor(indices, device=self.device)
        ), self.epochs_labels.index_select(0, torch.as_tensor(indices, device=self.device))

    def __len__(self):
        return len(self.epochs_labels)


class RandomSampler:
    def __init__(self, n):
        self.n = n
        self.indices = [i for i in range(n)]

    def next(self, n):
        if n <= len(self.indices):
            return random.sample(self.indices, n)
        else:
            return self.indices

    def reset(self):
        self.__init__(self.n)


def get_training_batch(subjects, indices):

    subject_index_separations = numpy.cumsum([len(subject) - 1 for subject in subjects])
    Xs = []
    ys = []
    for idx in indices:
        subject_start = 0
        for subject, subject_end in enumerate(subject_index_separations):
            if idx <= subject_end:
                X, y = subjects[subject].get_batch([idx - subject_start])
                Xs.append(X)
                ys.append(y)
                break
            else:
                subject_start = subject_end + 1

    return torch.cat(Xs, dim=0), torch.cat(ys, dim=0)


def dataset_per_subject(directory):
    """Function to read .fif files per subject as EpochsDataset
    ----------
    directory : path to directory that contains the -epo.fif files per subject
    ----------
    returns : list of EpochsDataset, number of subjects, number of channels, number of classes
    """

    epochs_list = [
        mne.read_epochs(directory + "/" + file) for file in os.listdir(directory)
    ]

    n_subjects = len(epochs_list)
    n_channels = epochs_list[0].get_data().shape[1]
    n_classes = len(epochs_list[0].event_id.keys())

    return (
        [
            EpochsDataset(
                # Go from (channels x samples) to (samples x channels)
                numpy.swapaxes(epochs.get_data(), 1, 2),
                epochs.events[:, 2] - 1,
            )
            for epochs in epochs_list
        ],
        n_subjects,
        n_channels,
        n_classes,
    )
