import mne
import os
import numpy
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

# Frequency boudns
l_freq = 1
h_freq = 45

# Frequency to downsample to (Hz)
final_sfreq = 125

# Window size (s)
window_size = 6


def to_mne_raw(run):
    """Convert one run to raw."""

    montage = mne.channels.make_standard_montage("standard_1005")
    eeg_data = 1e-6 * run.X
    sfreq = run.fs

    # Stim channel
    trigger = numpy.zeros((len(eeg_data), 1))
    trigger[run.trial - 1, 0] = run.y
    eeg_data = numpy.c_[eeg_data, trigger]

    # Events
    events = numpy.column_stack((run.trial, numpy.zeros(len(run.y), dtype=int), run.y))

    # Annotations
    annotations = mne.annotations_from_events(
        events, event_desc={v: k for k, v in event_dict.items()}, sfreq=sfreq
    )

    # Create MNE raw structure
    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = mne.io.RawArray(data=eeg_data.T, info=info)
    raw.set_montage(montage)
    raw.add_events(events, stim_channel="stim")
    raw.set_annotations(annotations)

    return raw


def process():

    # Per subject
    for subject_id in range(len(os.listdir("data/raw"))):

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
            for run in data["data"][3:]:

                raw = to_mne_raw(run)

                # Frequency filter
                raw.filter(l_freq=l_freq, h_freq=h_freq)

                # Downsample to 125Hz (> 2*45Hz)
                raw.resample(final_sfreq)

                # ICA using EoG --> we don't do this here because we are investigating mobile BCI's
                # ica = mne.preprocessing.ICA(
                #    n_components=20, method="fastica", random_state=23
                # )
                # ica.fit(raw, reject=dict(mag=5e-12, grad=4000e-13))
                # eog_indices, eog_scores = ica.find_bads_eog(raw)
                # ica.exclude = eog_indices
                # ica.apply(raw)

                # Drop EOG and get events now that resampling has been done
                events = mne.find_events(raw)
                raw.pick(picks="eeg")

                # Window
                epochs = mne.Epochs(
                    raw,
                    events,
                    picks="eeg",
                    tmin=-2,
                    tmax=4,
                    preload=True,
                )
                print("Number of epochs in run: " + str(len(epochs)))
                subject_epochs.append(epochs)

        # Save subject's Epochs as one large Epochs
        concatenated_epochs = mne.concatenate_epochs(subject_epochs)
        print(
            "Number of concatted epochs for subject: " + str(len(concatenated_epochs))
        )

        # Save concatenated epochs
        concatenated_epochs.save(
            "data/processed/subject" + str(subject_id + 1) + "-epo.fif"
        )


if __name__ == "__main__":

    # Create processed folder
    if os.path.isdir("data/processed"):
        print("Processed data exists already, clear first!")
        quit()
    else:
        os.mkdir("data/processed")

    process()
