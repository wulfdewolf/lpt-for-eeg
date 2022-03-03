from src.datasets.CompetitionDataset import CompetitionDataset

"""
Plotting preprocessing
"""
dataset = CompetitionDataset(
    seed=20200220,
    task="BCI_Competition_IV_2a",
    batch_size=16,
    window_size=6,
    device="cpu",
    data_dir="./data",
    model_type="FPT",
    process=False,
    window=True,
)

dataset.plot_windows_interactive(1)
