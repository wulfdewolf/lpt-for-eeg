import torch
import os
import mne
import numpy
import random


class EpochsDataset(torch.utils.data.Dataset):
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
        ), self.epochs_labels.index_select(
            0, torch.as_tensor(indices, device=self.device)
        )

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

    subject_last_indices = [
        subject_last_idx - 1
        for subject_last_idx in numpy.cumsum([len(subject) for subject in subjects])
    ]
    Xs = []
    ys = []
    for idx in indices:
        subject_start_idx = 0
        for subject, subject_last_idx in enumerate(subject_last_indices):
            if idx <= subject_last_idx:
                X, y = subjects[subject].get_batch([idx - subject_start_idx])
                Xs.append(X)
                ys.append(y)
                break
            else:
                subject_start_idx = subject_last_idx + 1

    return torch.cat(Xs, dim=0), torch.cat(ys, dim=0)


def dataset_per_subject(directory):

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
