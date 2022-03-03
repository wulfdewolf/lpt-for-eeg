from src.datasets.CompetitionDataset import CompetitionDataset

"""
Plotting preprocessing
"""
dataset = CompetitionDataset(
    seed=20200220,
    task="BCI_Competition_IV_2a",
)

dataset.plot_windows_interactive(1)
