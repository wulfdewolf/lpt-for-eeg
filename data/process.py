from turtle import down
import mne
import numpy
import os
import shutil
from scipy.io import loadmat

# Predefined channel names
ch_names = [
    "Fz",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "P1",
    "Pz",
    "P2",
    "POz",
    "EOG1",
    "EOG2",
    "EOG3",
    "stim",
]

# Predefined channel types
ch_types = ["eeg"] * 22 + ["eog"] * 3 + ["stim"]

# Predefined class labels
event_dict = {"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4}

# Frequency bounds
l_freq = 1
h_freq = 45

# Frequency to downsample to (Hz)
sfreq = 125

# Epoch bounds and size
tmin = -2
tmax = 4
window_size = abs(tmin) + tmax

# Windows for feature extraction
f_windows = 10


def to_mne_raw(run):
    """Converts one run to mne.Raw"""

    montage = mne.channels.make_standard_montage("standard_1005")
    eeg_data = 1e-6 * run.X  # Convert to V
    original_sfreq = run.fs

    # Stim channel
    trigger = numpy.zeros((len(eeg_data), 1))
    trigger[run.trial - 1, 0] = run.y
    eeg_data = numpy.c_[eeg_data, trigger]

    # Events
    events = numpy.column_stack((run.trial, numpy.zeros(len(run.y), dtype=int), run.y))

    # Annotations
    annotations = mne.annotations_from_events(
        events, event_desc={v: k for k, v in event_dict.items()}, sfreq=original_sfreq
    )

    # Create MNE raw structure
    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=original_sfreq)
    raw = mne.io.RawArray(data=eeg_data.T, info=info)
    raw.set_montage(montage)
    raw.add_events(events, stim_channel="stim")
    raw.set_annotations(annotations)

    return raw


def to_mne_epochs(raw, events):
    """Convert raw to mne.Epochs"""

    return mne.Epochs(
        raw,
        events,
        picks="eeg",
        tmin=tmin,
        tmax=tmax,
        preload=True,
    )


def get_subject_epochs(subject_id, downsample=True):
    """
    Collect a subject's runs in one large mne.Epochs

    Does the following:
    - frequency filtering
    - downsampling
    - windowing
    - baseline correction
    """

    subject_epochs = []

    # Per data type (training & evaluation)
    for run_file in os.listdir("thesis_data/raw/subject" + str(subject_id)):

        # Load raw file
        data = loadmat(
            "thesis_data/raw/subject" + str(subject_id) + "/" + str(run_file),
            struct_as_record=False,
            squeeze_me=True,
        )

        # Collect Epochs per session
        for run in data["data"][3:]:

            raw = to_mne_raw(run)

            # Frequency filter
            raw.filter(l_freq=l_freq, h_freq=h_freq)

            # Downsample to 125Hz (> 2*45Hz)
            if downsample:
                raw.resample(sfreq)

            # Get events now that resampling has been done
            events = mne.find_events(raw)

            # Drop stim and EOG
            raw.pick(picks="eeg")

            # Window
            epochs = to_mne_epochs(raw, events)
            print("Number of epochs in run: " + str(len(epochs)))
            subject_epochs.append(epochs)

    # Combine subject's Epochs into one large Epochs
    concatenated_epochs = mne.concatenate_epochs(subject_epochs)
    print("Number of concatenated epochs for subject: " + str(len(concatenated_epochs)))

    return concatenated_epochs


if __name__ == "__main__":

    # Verify if raw folder exists
    if not os.path.isdir("thesis_data/raw"):
        print(
            "thesis_data/raw folder does not exist, run thesis_data/download.sh first!"
        )
        quit()

    """
        PREPROCESSING
    """
    processed_exists = os.path.isdir("thesis_data/processed")

    if processed_exists:
        redo = input("thesis_data/processed folder exists already, redo? (y/n)")
        if redo == "y":
            shutil.rmtree("thesis_data/processed")

    if not processed_exists or redo == "y":
        os.mkdir("thesis_data/processed")

        # Per subject
        for subject_id in range(1, len(os.listdir("thesis_data/raw")) + 1):

            subject_epochs = get_subject_epochs(subject_id, downsample=True)
            epochs_data = subject_epochs.get_data(units="uV")
            epochs_labels = subject_epochs.events[:, 2] - 1

            # Safety check
            assert len(epochs_data) == len(epochs_labels)

            # Save labels
            with open(
                "thesis_data/processed/subject" + str(subject_id) + "_labels.npy", "wb"
            ) as f:
                numpy.save(f, epochs_labels)

            # Standardize
            processed_data = mne.decoding.Scaler(scalings="mean").fit_transform(
                epochs_data
            )

            # Change dimensions: (epochs, channels, samples) -> (epochs, samples, channels)
            processed_data = numpy.swapaxes(processed_data, 1, 2)

            # Save processed data
            with open(
                "thesis_data/processed/subject" + str(subject_id) + "_timepoints.npy",
                "wb",
            ) as f:
                numpy.save(f, processed_data)

    """
        FEATURE EXTRACTION
    """
    feature_extracted_exists = os.path.isdir("thesis_data/feature_extracted")

    if feature_extracted_exists:
        redo = input("thesis_data/feature_extracted folder exists already, redo? (y/n)")
        if redo == "y":
            shutil.rmtree("thesis_data/feature_extracted")

    if not feature_extracted_exists or redo == "y":
        os.mkdir("thesis_data/feature_extracted")

        # Per subject
        for subject_id in range(1, len(os.listdir("thesis_data/raw")) + 1):

            subject_epochs = get_subject_epochs(subject_id, downsample=False)
            epochs_data = subject_epochs.get_data(units="uV")
            epochs_labels = subject_epochs.events[:, 2] - 1

            # Safety check
            assert len(epochs_data) == len(epochs_labels)

            # Save labels
            with open(
                "thesis_data/feature_extracted/subject"
                + str(subject_id)
                + "_labels.npy",
                "wb",
            ) as f:
                numpy.save(f, epochs_labels)

            # Drop last of epochs data samples (always one extra)
            features_data = epochs_data[:, :, : epochs_data[0][0].shape[0] - 1]

            # Create windows in epochs (windows, epochs, channels, samples)
            windows_in_features = numpy.split(features_data, f_windows, 2)

            # Calculate psds for windows
            psds = []
            psd_estimator = mne.decoding.PSDEstimator(sfreq, fmin=l_freq, fmax=h_freq)
            for window in windows_in_features:
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
                "thesis_data/feature_extracted/subject"
                + str(subject_id)
                + "_features.npy",
                "wb",
            ) as f:
                numpy.save(f, vectorized)
