from pathlib import Path
import os
import errno
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import Lambda
from datasets.validate_data import validate_loaded_sequence_length


# Ref: https://github.com/tychovdo/MovingMNIST/blob/master/MovingMNIST.py
# Note: This code contains some modifications compared to the original implementation.
class MovingMNIST(Dataset):
    """`MovingMNIST <http://www.cs.toronto.edu/~nitish/unsupervised_video/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        split (int, optional): Train/test split size. Number defines how many samples
            belong to test set.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        split_rate (int, optional): Number of frames to keep from the original 20 frames.
    """

    urls = ["https://github.com/tychovdo/MovingMNIST/raw/master/mnist_test_seq.npy.gz"]
    raw_folder = "raw"
    processed_folder = "processed"
    training_file = "moving_mnist_train.pt"
    test_file = "moving_mnist_test.pt"
    metadata = {"input_seq_len": 5, "output_seq_len": 5, "input_size": 64}

    def __init__(
        self,
        root,
        dataset_mode="train",
        split=1000,
        split_rate=10,
        transform=None,
        download=False,
        split_validation=False,
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split
        self.dataset_mode = dataset_mode  # 'train', 'val' or 'test'
        self.val_ratio = 0.1

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found." + " You can use download=True to download it"
            )

        if self.dataset_mode == "train":
            self.train_data = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file)
            ).unsqueeze(-3)[:, :split_rate]
            if split_validation:
                qtd_val_elems = round(self.train_data.shape[0] * self.val_ratio)
                self.train_data = self.train_data[:-qtd_val_elems].clone()
        elif self.dataset_mode == "val":
            train_data = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file)
            ).unsqueeze(-3)[:, :split_rate]
            qtd_val_elems = round(train_data.shape[0] * self.val_ratio)
            self.val_data = train_data[-qtd_val_elems:].clone()
        elif self.dataset_mode == "test":
            self.test_data = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file)
            ).unsqueeze(-3)[:, :split_rate]

        validate_loaded_sequence_length(
            seq_len=split_rate,
            input_seq_len=self.metadata["input_seq_len"],
            output_seq_len=self.metadata["output_seq_len"],
        )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            seq (Tensor): Resulted sequence.
        """
        seq = None

        if self.dataset_mode == "train":
            seq = self.train_data[index]
        elif self.dataset_mode == "val":
            seq = self.val_data[index]
        elif self.dataset_mode == "test":
            seq = self.test_data[index]

        if self.transform is not None:
            seq = self.transform(seq)

        return seq

    def __len__(self):
        if self.dataset_mode == "train":
            return len(self.train_data)
        elif self.dataset_mode == "val":
            return len(self.val_data)
        elif self.dataset_mode == "test":
            return len(self.test_data)
        raise Exception("Dataset not found.")

    def _check_exists(self):
        return os.path.exists(
            os.path.join(self.root, self.processed_folder, self.training_file)
        ) and os.path.exists(
            os.path.join(self.root, self.processed_folder, self.test_file)
        )

    def download(self):
        """Download the Moving MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print("Downloading " + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition("/")[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, "wb") as f:
                f.write(data.read())
            with open(file_path.replace(".gz", ""), "wb") as out_f, gzip.GzipFile(
                file_path
            ) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print("Processing...")

        training_set = torch.from_numpy(
            np.load(
                os.path.join(self.root, self.raw_folder, "mnist_test_seq.npy")
            ).swapaxes(0, 1)[: -self.split]
        )
        test_set = torch.from_numpy(
            np.load(
                os.path.join(self.root, self.raw_folder, "mnist_test_seq.npy")
            ).swapaxes(0, 1)[-self.split :]
        )

        with open(
            os.path.join(self.root, self.processed_folder, self.training_file), "wb"
        ) as f:
            torch.save(training_set, f)
        with open(
            os.path.join(self.root, self.processed_folder, self.test_file), "wb"
        ) as f:
            torch.save(test_set, f)

        print("Done!")

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        tmp = self.dataset_mode
        fmt_str += "    Train/test: {}\n".format(tmp)
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str

    @classmethod
    def load(
        cls,
        dataset_mode,
        dataset_size,
        normalize_values,
        batch_size,
        num_workers,
        shuffle=False,
        split_validation=False,
    ):
        transformation = None
        if normalize_values:
            transformation = Lambda(
                lambda x: x.float() / 255.0 if x.max() > 1 else x.float()
            )

        folder_data = Path(__file__).resolve().parent.parent
        dataset = MovingMNIST(
            root=f"{folder_data}/data/moving_mnist",
            split_rate=10,
            split=1000,
            dataset_mode=dataset_mode,
            download=True,
            transform=transformation,
            split_validation=split_validation,
        )
        max_value = 255.0

        # Tests purposes
        if dataset_size < 1.0:
            num_samples = round(dataset_size * len(dataset))
            dataset = Subset(dataset, range(num_samples))

        dataloader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
        )

        return dataloader, max_value
