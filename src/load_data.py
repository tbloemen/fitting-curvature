from enum import Enum

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor


class DatasetKind(Enum):
    ALL = 1
    TRAIN = 2
    TEST = 3


def load_mnist(kind: DatasetKind) -> tuple[torch.Tensor, torch.Tensor]:
    train_dataset = datasets.MNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )
    train_x = torch.stack([img for img, _ in train_dataset])
    train_y = torch.tensor([label for _, label in train_dataset])

    if kind == DatasetKind.TRAIN:
        return train_x, train_y

    test_dataset = datasets.MNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )
    test_x = torch.stack([img for img, _ in test_dataset])
    test_y = torch.tensor([label for _, label in test_dataset])

    if kind == DatasetKind.TEST:
        return test_x, test_y
    return torch.concat((train_x, test_x)), torch.concat((train_y, test_y))


def load_data(dataset: str):
    stripped_dataset = dataset.lower().strip()
    # MNIST only for now.
    if stripped_dataset == "mnist":
        return load_mnist(DatasetKind.ALL)

    raise ValueError("Dataset not recognised: ", stripped_dataset)
