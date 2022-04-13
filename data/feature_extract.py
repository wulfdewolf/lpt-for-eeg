import mne
import os
import numpy

import process


def feature_extract():

    # Loop over processed data
    for subject_id in range(len(os.listdir("data/processed"))):

        epochs = mne.read_epochs(
            "data/processed/subject" + str(subject_id + 1) + "-epo.fif", preload=True
        )

        # PSD -> (n_epochs, n_channels, n_freq, n_windows)
        psd = mne.time_frequency.psd_welch(
            epochs,
            n_overlap=0,
            fmin=process.l_freq,
            fmax=process.h_freq,
            average=None,
            n_fft=int(
                (process.window_size * process.final_sfreq) / 10
            ),  # ( epoch_length * sfreq) / n_windows_in_epoch) we want 10 non-overlapping windows
        )[0]

        # Transform to have correct labels + flatten channels
        psd = numpy.swapaxes(psd, 2, 3)  # (n_epochs, n_channels, n_windows, n_freq)
        psd = psd.reshape(
            -1, *psd.shape[-2:]
        )  # (n_epochs * n_channels, n_windows, n_freq)
        psd_labels = epochs.events[:, 2].repeat(22)

        # Save data
        with open(
            "data/feature_extracted/subject" + str(subject_id + 1) + "_data.npy", "wb"
        ) as f:
            numpy.save(f, psd)

        # Save labels
        with open(
            "data/feature_extracted/subject" + str(subject_id + 1) + "_labels.npy", "wb"
        ) as f:
            numpy.save(f, psd_labels)


if __name__ == "__main__":

    # Verify if processed data exists
    if not os.path.isdir("data/processed"):
        print("processed data doesn't exist!")
        quit()

    # Create feature_extracted folder
    if os.path.isdir("data/feature_extracted"):
        print("feature_extracted folder exists already, delete first!")
        quit()
    else:
        os.mkdir("data/feature_extracted")

    feature_extract()
