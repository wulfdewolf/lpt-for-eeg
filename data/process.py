import mne
import os
import numpy as np
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

# Create processed folder
if os.path.isdir("data/processed"):
    print("Processed data exists already, clear first!")
    quit()
else:
    os.mkdir("data/processed")

# Per subject
for subject_id, subject_file in enumerate(sorted(os.listdir("data/raw"))):

    # Collect Epochs per subject
    subject_epochs = []

    # Per data type (training & evaluation)
    for run_file in os.listdir("data/raw/subject" + str(subject_id + 1)):

        # Load raw file
        data = loadmat(
            "data/raw/subject" + str(subject_id + 1) + "/" + str(run_file),
            struct_as_record=False,
            squeeze_me=True,
        )

        # Collect Epochs per session
        for run_id, run in enumerate(data["data"][3:]):

            """Convert one run to raw."""
            event_id = {}
            n_chan = run.X.shape[1]
            montage = mne.channels.make_standard_montage("standard_1005")
            eeg_data = 1e-6 * run.X
            sfreq = run.fs

            # Stim channel
            trigger = np.zeros((len(eeg_data), 1))
            trigger[run.trial - 1, 0] = run.y
            eeg_data = np.c_[eeg_data, trigger]

            # Events
            event_id = {ev: (ii + 1) for ii, ev in enumerate(run.classes)}
            event_ids = {
                class_value: class_label
                for class_value, class_label in zip(np.unique(run.y), run.classes)
            }
            events = np.column_stack(
                (run.trial, np.zeros(len(run.y), dtype=int), run.y)
            )

            # Annotations
            annotations = mne.annotations_from_events(
                events, event_desc=event_ids, sfreq=sfreq
            )

            # Create MNE raw structure
            info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
            raw = mne.io.RawArray(data=eeg_data.T, info=info)
            raw.set_montage(montage)
            raw.add_events(events, stim_channel="stim")

            # Frequency filter
            raw.filter(l_freq=1, h_freq=80)
            raw.notch_filter(50)

            # Downsample to 160Hz
            raw.resample(160)

            # ICA using EoG
            ica = mne.preprocessing.ICA(
                n_components=20, method="fastica", random_state=23
            )
            ica.fit(raw, reject=dict(mag=5e-12, grad=4000e-13))
            eog_indices, eog_scores = ica.find_bads_eog(raw)
            ica.exclude = eog_indices
            ica.apply(raw)

            # Drop EOG
            raw.pick(picks="eeg")

            # Window
            epochs = mne.Epochs(raw, events, picks="eeg", tmin=-2, tmax=4)
            subject_epochs.append(epochs)

    # Save subject's Epochs as one large Epochs
    mne.concatenate_epochs(subject_epochs).save(
        "data/processed/subject" + str(subject_id + 1) + "-epo.fif"
    )
