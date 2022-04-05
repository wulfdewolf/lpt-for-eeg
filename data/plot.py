import mne
import os

# Create plots folder
if not os.path.exists("data/plots"):
    os.mkdir("data/plots")

# Read raw
raw = mne.io.read_raw_fif("data/motor_imagery/processed/0/0T-raw.fif", preload=True)

# Get events
events = mne.find_events(raw)
raw.add_events(events)

# Get annotations
annotations = mne.annotations_from_events(
    events,
    sfreq=250,
    event_desc={1: "left hand", 2: "right hand", 3: "feet", 4: "tongue"},
)
raw.set_annotations(annotations)

# Plot raw
plot = raw.plot(duration=10, n_channels=22, show_scrollbars=False)
plot.savefig("data/plots/raw.pdf")

# Plot psd
plot = raw.plot_psd()
plot.savefig("data/plots/psd.pdf")

# Downsample
downsampled = raw.copy()
downsampled = downsampled.resample(100)

# Plot resampled
plot = downsampled.plot(duration=10, n_channels=22, show_scrollbars=False)
plot.savefig("data/plots/resampled.pdf")

# Bandpass filter
filtered = raw.copy()
filtered = filtered.filter(l_freq=4, h_freq=38)

# Plot filtered
plot = filtered.plot(duration=10, n_channels=22, show_scrollbars=False)
plot.savefig("data/plots/filtered.pdf")

# Epochs
epochs = mne.Epochs(raw, events, tmin=-2, tmax=4)
plot = epochs.plot_image(picks=["Fz"])[0]
plot.savefig("data/plots/windows.pdf")
