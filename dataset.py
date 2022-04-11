import torch
import os
import mne
import numpy

# Class taken from mne-torch tools:
# https://github.com/mne-tools/mne-torch/blob/master/common.py
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

    def __init__(self, epochs_data, epochs_labels, device, transform=None):
        assert len(epochs_data) == len(epochs_labels)
        self.epochs_data = epochs_data
        self.epochs_labels = epochs_labels
        self.device = device
        self.transform = transform

    def __len__(self):
        return len(self.epochs_labels)

    def __getitem__(self, idx):
        X, y = self.epochs_data[idx], self.epochs_labels[idx]
        if self.transform is not None:
            X = self.transform(X)
        X = torch.as_tensor(X, device=self.device, dtype=torch.float32)
        return X, y


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
