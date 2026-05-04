from typing import List, Tuple
from torch import Tensor
from torch.utils.data import DataLoader
from datasets.taasrad19 import TAASRAD19
from datasets.moving_mnist import MovingMNIST


def get_datasets_parameters(name: str) -> dict:
    if name == "taasrad19":
        return TAASRAD19.metadata
    elif name == "moving_mnist":
        return MovingMNIST.metadata
    raise Exception(f"Dataset '{name}' not implemented")


def load_dataset(
    name: str,
    dataset_mode: str,
    dataset_size: float,
    normalize_values: bool,
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
    split_validation: bool = False,
) -> Tuple[DataLoader, float]:
    if name == "taasrad19":
        return TAASRAD19.load(
            dataset_mode,
            dataset_size,
            normalize_values,
            batch_size,
            num_workers,
            shuffle,
        )
    elif name == "moving_mnist":
        return MovingMNIST.load(
            dataset_mode,
            dataset_size,
            normalize_values,
            batch_size,
            num_workers,
            shuffle,
            split_validation,
        )
    raise Exception(f"Dataset '{name}' not implemented")


def denormalize_values(values: Tensor, max_value: float):
    """Denormalize a Tensor of values between [0, 1]"""
    return values * max_value
