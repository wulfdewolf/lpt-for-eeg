from braindecode.datautil.preprocess import preprocess
from torch.utils import data
from src.datasets.CNNDataset import CNNDataset
from src.datasets.FPTDataset import FPTDataset
import mne

"""
Plotting preprocessing
"""
dataset = FPTDataset(
    seed=20200220,
    task="BCI_Competition_IV_2a",
    batch_size=16,
    window_size=200,
    device="gpu",
    data_dir="./data",
    process=False,
    window=True,
)

dataset.plot_windows_interactive(1)
