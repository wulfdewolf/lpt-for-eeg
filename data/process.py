import os
from venv import create
import mne
import numpy

import util


def process(create_labels):

    # Per subject
    for subject_id in range(1, len(os.listdir("data/raw")) + 1):

        subject_epochs = util.subject_epochs(subject_id)
        epochs_data = subject_epochs.get_data(units="uV")

        # Standardize
        epochs_data = mne.decoding.Scaler(scalings="mean").fit_transform(epochs_data)

        # Save data
        with open(
            "data/processed/subject" + str(subject_id) + "_timepoints.npy", "wb"
        ) as f:
            numpy.save(f, epochs_data)

        if create_labels:
            epochs_labels = subject_epochs.events[:, 2] - 1
            with open(
                "data/labels/subject" + str(subject_id) + "_labels.npy", "wb"
            ) as f:
                numpy.save(f, epochs_labels)

            # Safety check
            assert len(epochs_data) == len(epochs_labels)


if __name__ == "__main__":

    # Create processed folder
    if os.path.isdir("data/processed"):
        print("processed folder exists already, clear first!")
        quit()
    else:
        os.mkdir("data/processed")

    # Create labels folder
    create_labels = not os.path.isdir("data/labels")
    if create_labels:
        os.mkdir("data/labels")

    process(create_labels)
