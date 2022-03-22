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
os.mkdir("data/competition/processed")

# Per type
for type, folder in zip(
    ["T", "E"], ["data/competition/raw/training", "data/competition/raw/evaluation"]
):
    # Per subject
    for subject_id, subject_file in enumerate(os.listdir(folder)):

        # Create subject folder
        if not os.path.exists("data/competition/processed/" + str(subject_id)):
            os.mkdir("data/competition/processed/" + str(subject_id))

        # Load raw file
        data = loadmat(
            folder + "/" + str(subject_file),
            struct_as_record=False,
            squeeze_me=True,
        )

        # Per session
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
            print(event_ids)

            # Annotations
            annotations = mne.annotations_from_events(
                events, event_desc=event_ids, sfreq=sfreq
            )

            # Create MNE structures
            info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
            raw = mne.io.RawArray(data=eeg_data.T, info=info)
            raw.set_montage(montage)
            # raw.set_annotations(annotations)
            raw.add_events(events, stim_channel="stim")

            # Save
            raw.save(
                "data/competition/processed/"
                + str(subject_id)
                + "/"
                + str(run_id)
                + type
                + "-raw.fif"
            )
