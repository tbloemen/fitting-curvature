from enum import Enum

import torch
from torch import Tensor
from torchvision import datasets
from torchvision.transforms import ToTensor

from src.synthetic_data import SYNTHETIC_DATASETS, load_synthetic

VALID_DATASETS = ["mnist"] + list(SYNTHETIC_DATASETS.keys())


class DatasetKind(Enum):
    ALL = 1
    TRAIN = 2
    TEST = 3


def load_mnist(kind: DatasetKind) -> tuple[torch.Tensor, torch.Tensor]:
    train_dataset = datasets.MNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )
    train_x = torch.stack([img for img, _ in train_dataset]).reshape(-1, 784)
    train_y = torch.tensor([label for _, label in train_dataset])

    if kind == DatasetKind.TRAIN:
        return train_x, train_y

    test_dataset = datasets.MNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )
    test_x = torch.stack([img for img, _ in test_dataset]).reshape(-1, 784)
    test_y = torch.tensor([label for _, label in test_dataset])

    if kind == DatasetKind.TEST:
        return test_x, test_y
    return torch.concat((train_x, test_x)), torch.concat((train_y, test_y))


def load_raw_data(
    dataset: str, kind=DatasetKind.ALL, n_samples: int = 500
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Load a dataset by name.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor | None]
        (X, y, D) where D is a precomputed geodesic distance matrix or None.
    """
    stripped_dataset = dataset.lower().strip()

    if stripped_dataset == "mnist":
        X, y = load_mnist(kind)
        return X, y, None

    if stripped_dataset in SYNTHETIC_DATASETS:
        return load_synthetic(stripped_dataset, n_samples)

    raise ValueError("Dataset not recognised: ", stripped_dataset)
