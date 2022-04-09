import torch

# Class taken from mne-torch tools:
# https://github.com/mne-tools/mne-torch/blob/master/common.py
class EpochsDataset(torch.Dataset):
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

    def __init__(self, epochs_data, epochs_labels, transform=None):
        assert len(epochs_data) == len(epochs_labels)
        self.epochs_data = epochs_data
        self.epochs_labels = epochs_labels
        self.transform = transform

    def __len__(self):
        return len(self.epochs_labels)

    def __getitem__(self, idx):
        X, y = self.epochs_data[idx], self.epochs_labels[idx]
        if self.transform is not None:
            X = self.transform(X)
        X = torch.as_tensor(X[None, ...])
        return X, y


def dataset_per_subject(directory):
    """Function to read .fif files per subject as EpochsDataset
    ----------
    directory : path to directory that contains the .fif files per subject
    ----------
    returns : list of EpochsDataset, one per subject
    """
