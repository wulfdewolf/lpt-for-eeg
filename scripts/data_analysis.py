from torch.utils import data
from src.datasets.CNNDataset import CNNDataset

dataset = CNNDataset(
    seed=20200220,
    task="BCI_Competition_IV_2a",
    batch_size=16,
    window_size=1000,
    device="gpu",
    data_dir="./data",
)
dataset.plot_raw_interactive(0)
