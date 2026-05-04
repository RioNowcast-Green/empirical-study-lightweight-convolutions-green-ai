from pathlib import Path
from typing import Callable, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from datasets.validate_data import validate_loaded_sequence_length


class TAASRAD19(Dataset):
    metadata = {"input_seq_len": 5, "output_seq_len": 5, "input_size": 64}

    def __init__(
        self,
        dataset_mode="train",
        normalize: bool = False,
        transform: Optional[Callable] = None,
    ):
        data_file_name = ""
        if dataset_mode == "train":
            data_file_name = "taasrad19_2016_2017_2018_64x64.npy"
        elif dataset_mode == "test":
            data_file_name = "taasrad19_2019_64x64.npy"
        elif dataset_mode == "val":
            data_file_name = "taasrad19_2018_64x64.npy"

        folder_data = Path(__file__).resolve().parent.parent
        data_tensor = torch.from_numpy(
            np.load(f"{folder_data}/data/taasrad19/{data_file_name}")
        )
        self.data = data_tensor.data.contiguous()
        self.max_value = 52.5

        validate_loaded_sequence_length(
            seq_len=self.data.shape[1],
            input_seq_len=self.metadata["input_seq_len"],
            output_seq_len=self.metadata["output_seq_len"],
        )

        if normalize:
            self.data = self.normalize_values()

        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.transform is not None:
            data = self.transform(data)

        return data

    def normalize_values(self):
        return self.data / self.max_value

    @classmethod
    def load(
        cls,
        dataset_mode,
        dataset_size,
        normalize_values,
        batch_size,
        num_workers,
        shuffle=False,
    ):
        dataset = cls(
            dataset_mode=dataset_mode,
            normalize=normalize_values,
        )
        max_value = dataset.max_value

        # Tests purposes
        if dataset_size < 1.0:
            num_samples = round(dataset_size * len(dataset))
            dataset = Subset(dataset, range(num_samples))

        dataloader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
        )

        return dataloader, max_value
