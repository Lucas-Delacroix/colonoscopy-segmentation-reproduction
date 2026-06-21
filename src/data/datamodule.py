from torch.utils.data import DataLoader, Dataset

from data.datasets.kvasir import KvasirDataset
from data.transforms.augmentation import get_train_transforms, get_val_transforms


class PolypDataModule:
    DATASETS: dict[str, type[Dataset]] = {
        "kvasir": KvasirDataset,
    }

    def __init__(
        self,
        dataset_name: str,
        data_root: str,
        image_size: int = 352,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        if dataset_name not in self.DATASETS:
            options = ", ".join(sorted(self.DATASETS))
            raise ValueError(f"Unknown dataset '{dataset_name}'. Available datasets: {options}")

        self.dataset_cls = self.DATASETS[dataset_name]
        self.data_root = data_root
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def setup(self) -> None:
        self._train_dataset = self.dataset_cls(
            root=self.data_root,
            split="train",
            transform=get_train_transforms(self.image_size),
            image_size=self.image_size,
        )
        self._val_dataset = self.dataset_cls(
            root=self.data_root,
            split="val",
            transform=get_val_transforms(self.image_size),
            image_size=self.image_size,
        )
        self._test_dataset = self.dataset_cls(
            root=self.data_root,
            split="test",
            transform=get_val_transforms(self.image_size),
            image_size=self.image_size,
        )

        self._log_split_info()

    def _log_split_info(self) -> None:
        print("Dataset loaded:")
        print(f"  Train:    {len(self._train_dataset)} images")
        print(f"  Validation: {len(self._val_dataset)} images")
        print(f"  Test:     {len(self._test_dataset)} images")

    @staticmethod
    def _require_dataset(dataset: Dataset | None, split: str) -> Dataset:
        if dataset is None:
            raise RuntimeError(f"DataModule.setup() must be called before requesting the {split} loader.")
        return dataset

    def train_loader(self) -> DataLoader:
        return DataLoader(
            self._require_dataset(self._train_dataset, "train"),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_loader(self) -> DataLoader:
        return DataLoader(
            self._require_dataset(self._val_dataset, "validation"),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_loader(self) -> DataLoader:
        return DataLoader(
            self._require_dataset(self._test_dataset, "test"),
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
