from torch.utils import data
from src.datasets.CNNDataset import CNNDataset
from src.datasets.FPTDataset import FPTDataset

dataset = FPTDataset(
    seed=20200220,
    task="BCI_Competition_IV_2a",
    batch_size=16,
    window_size=1000,
    device="gpu",
    data_dir="./data",
)
dataset.plot_windows_interactive(5)
