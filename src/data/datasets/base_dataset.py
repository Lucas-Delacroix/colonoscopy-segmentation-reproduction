from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):

    def __init__(self, root: str, split: str, transform=None):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split '{split}'. Expected one of: train, val, test.")
        self.root = root
        self.split = split
        self.transform = transform
        self.samples = self._load_samples()

    @abstractmethod
    def _load_samples(self) -> list[Any]:
        ...

    def __len__(self) -> int:
        return len(self.samples)

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]:
        ...
