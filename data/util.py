import mne
import numpy
import os
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
    """Convert one run to raw."""

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
    return mne.Epochs(
        raw,
        events,
        picks="eeg",
        tmin=tmin,
        tmax=tmax,
        preload=True,
    )


def subject_epochs(subject_id):
    subject_epochs = []

    # Per data type (training & evaluation)
    for run_file in os.listdir("data/raw/subject" + str(subject_id)):

        # Load raw file
        data = loadmat(
            "data/raw/subject" + str(subject_id) + "/" + str(run_file),
            struct_as_record=False,
            squeeze_me=True,
        )

        # Collect Epochs per session
        for run in data["data"][3:]:

            raw = to_mne_raw(run)

            # Frequency filter
            raw.filter(l_freq=l_freq, h_freq=h_freq)

            # Downsample to 125Hz (> 2*45Hz)
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
