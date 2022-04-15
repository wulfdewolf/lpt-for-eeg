import mne
import os
import numpy

import util


def feature_extract(create_labels):

    # Per subject
    for subject_id in range(1, len(os.listdir("data/processed")) + 1):

        subject_epochs = util.subject_epochs(subject_id)
        epochs_data = subject_epochs.get_data(units="uV")

        # Drop last of epochs data samples (always one extra)
        epochs_data = epochs_data[:, :, : epochs_data[0][0].shape[0] - 1]

        # Create windows in epochs (windows, epochs, channels, samples)
        windows_in_epochs_data = numpy.split(epochs_data, util.f_windows, 2)

        # Calculate psds for windows
        psds = []
        psd_estimator = mne.decoding.PSDEstimator(
            util.sfreq, fmin=util.l_freq, fmax=util.h_freq
        )
        for window in windows_in_epochs_data:
            psd = psd_estimator.transform(window)
            psds.append(psd)

        # Stack (epochs, windows, channels, freqs)
        concatenated_psds = numpy.stack(psds, axis=1)

        # Vectorize
        vectorized = []
        vectorizer = mne.decoding.Vectorizer()
        for epoch in concatenated_psds:
            vectorized.append(vectorizer.fit_transform(epoch))

        # Stack (epochs, windows, channels * freqs)
        vectorized = numpy.stack(vectorized, axis=0)

        # Save features
        with open(
            "data/feature_extracted/subject" + str(subject_id) + "_features.npy", "wb"
        ) as f:
            numpy.save(f, vectorized)

        if create_labels:
            epochs_labels = subject_epochs.events[:, 2] - 1
            with open(
                "data/labels/subject" + str(subject_id) + "_labels.npy", "wb"
            ) as f:
                numpy.save(f, epochs_labels)

            # Safety check
            assert len(epochs_data) == len(epochs_labels)


if __name__ == "__main__":

    # Create feature_extracted folder
    if os.path.isdir("data/feature_extracted"):
        print("feature_extracted folder exists already, delete first!")
        quit()
    else:
        os.mkdir("data/feature_extracted")

    # Create labels folder
    create_labels = not os.path.isdir("data/labels")
    if create_labels:
        os.mkdir("data/labels")

    feature_extract(create_labels)
