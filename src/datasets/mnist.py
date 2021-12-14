from einops import rearrange
import torchvision
import torchvision.transforms as transforms

from src.datasets.dataset import Dataset
from torch.utils.data import ConcatDataset


class MNISTDataset(Dataset):
    def __init__(self, *args, **kwargs):

        # Load data
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=0.0, std=1.0),
            ]
        )
        train = torchvision.datasets.MNIST(
            "data/mnist", download=True, train=True, transform=transform
        )
        test = torchvision.datasets.MNIST(
            "data/mnist", download=True, train=False, transform=transform
        )
        self.windows = ConcatDataset([train, test])
        self.patch_size = 4
        super().__init__(
            n_channels=self.patch_size ** 2,
            classes=10,
            process=False,
            window=False,
            downsample=False,
            *args,
            **kwargs
        )

    # Preprocessing
    def process(self):
        return 0

    # Cutting compute windows
    def cut_windows(self):
        return 0

    # Getting a single batch
    def get_batch(self, batch_size=None, train=True):
        if train:
            _, (x, y) = next(self.train_enum, (None, (None, None)))
            if x is None:
                self.train_enum = enumerate(self.d_train)
                _, (x, y) = next(self.train_enum)
        else:
            _, (x, y) = next(self.test_enum, (None, (None, None)))
            if x is None:
                self.test_enum = enumerate(self.d_test)
                _, (x, y) = next(self.test_enum)

        if self.patch_size is not None:
            x = rearrange(
                x,
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=self.patch_size,
                p2=self.patch_size,
            )

        x = x.to(device=self.device)
        y = y.to(device=self.device)

        return x, y
