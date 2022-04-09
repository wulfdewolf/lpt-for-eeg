import mne
import os

# Create plots folder
if not os.path.exists("data/plots"):
    os.mkdir("data/plots")

# Read Epochs
epochs = mne.read_epochs("data/processed/subject1-epo.fif", preload=True)
event_dict = {"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4}
events = epochs.events

# Plot Epochs for single channel
epochs_plot = epochs.plot_image(picks=["Fz"])[0]
epochs_plot.savefig("data/plots/windows.pdf")

# Plot PSD
psd_plot = epochs.plot_psd()
psd_plot.savefig("data/plots/psd.pdf")

# Plot downsampled
downsampled_plot = epochs.resample(100).plot(
    n_epochs=5, events=events, event_id=event_dict, show_scrollbars=False
)
downsampled_plot.savefig("data/plots/downsampled.pdf")

# Plot filtered
filtered_plot = epochs.filter(l_freq=4, h_freq=38).plot(
    n_epochs=5, events=events, event_id=event_dict, show_scrollbars=False
)
filtered_plot.savefig("data/plots/filtered.pdf")
