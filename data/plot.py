import mne
import os
from scipy.io import loadmat

import process


def plot_raw():

    # Read raw
    data = loadmat(
        "data/raw/subject1/A01T.mat",
        struct_as_record=False,
        squeeze_me=True,
    )
    raw = process.to_mne_raw(data["data"][3:][0])
    raw.add_events(mne.find_events(raw))

    # Plot raw
    raw_plot = raw.plot(duration=10, n_channels=22, show_scrollbars=False)
    raw_plot.savefig("data/plots/raw.pdf")

    # Plot psd
    psd_plot = raw.plot_psd()
    psd_plot.savefig("data/plots/raw_psd.pdf")

    # ICA
    ica = mne.preprocessing.ICA(n_components=10, method="fastica", random_state=23)
    ica.fit(raw, reject=dict(mag=5e-12, grad=4000e-13))
    raw.load_data()
    ica_plot = ica.plot_sources(raw, start=0, stop=10, show_scrollbars=False)
    ica_plot.savefig("data/plots/ica.pdf")

    # Downsample
    downsampled = raw.copy()
    downsampled = downsampled.resample(process.final_sfreq)
    downsampled_plot = downsampled.plot(
        duration=10, n_channels=22, show_scrollbars=False
    )
    downsampled_plot.savefig("data/plots/raw_downsampled.pdf")

    # Bandpass filter
    filtered = raw.copy()
    filtered = filtered.filter(l_freq=process.l_freq, h_freq=process.h_freq)
    filtered_plot = filtered.plot(duration=10, n_channels=22, show_scrollbars=False)
    filtered_plot.savefig("data/plots/raw_filtered.pdf")


def plot_epochs():

    # Read Epochs
    epochs = mne.read_epochs("data/processed/subject1-epo.fif", preload=True)

    # Plot Epochs for single channel
    epochs_plot = epochs.plot_image(picks=["Fz"])[0]
    epochs_plot.savefig("data/plots/epochs.pdf")

    # Plot PSD
    psd_plot = epochs.plot_psd()
    psd_plot.savefig("data/plots/epochs_psd.pdf")


if __name__ == "__main__":

    # Create plots folder
    if not os.path.exists("data/plots"):
        os.mkdir("data/plots")

    plot_raw()
    plot_epochs()
