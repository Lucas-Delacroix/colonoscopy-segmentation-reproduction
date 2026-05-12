import json
import os
import random
from pathlib import Path
import cv2
import numpy as np
import torch

from data.datasets.base_dataset import BaseDataset


class KvasirDataset(BaseDataset):
    """
        data/raw/kvasir-seg/
            images/
                *.jpg
            masks/
                *.jpg
    """

    SPLIT_FILE = "data/splits/kvasir_split.json"

    def __init__(self, root: str, split: str, transform=None, image_size: int = 352):
        self.image_size = image_size
        super().__init__(root, split, transform)

    def _load_samples(self) -> list:
        images_dir = Path(self.root) / "images"
        masks_dir = Path(self.root) / "masks"

        all_images = sorted(images_dir.glob("*.jpg"))
        all_masks = [masks_dir / img.name for img in all_images]

        for mask_path in all_masks:
            assert mask_path.exists(), f"Máscara não encontrada: {mask_path}"

        split_indices = self._get_or_create_split(len(all_images))
        indices = split_indices[self.split]

        return [(str(all_images[i]), str(all_masks[i])) for i in indices]

    def _get_or_create_split(self, total: int) -> dict:
        split_path = Path(self.SPLIT_FILE)

        if split_path.exists():
            with open(split_path) as f:
                return json.load(f)

        split_path.parent.mkdir(parents=True, exist_ok=True)

        indices = list(range(total))
        random.seed(42)
        random.shuffle(indices)

        n_train = int(total * 0.8)
        n_val = int(total * 0.1)

        splits = {
            "train": indices[:n_train],
            "val": indices[n_train : n_train + n_val],
            "test": indices[n_train + n_val :],
        }

        with open(split_path, "w") as f:
            json.dump(splits, f, indent=2)

        return splits

    def __getitem__(self, idx: int) -> dict:
        image_path, mask_path = self.samples[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        elif isinstance(mask, torch.Tensor) and mask.ndim == 2:
            mask = mask.unsqueeze(0).float()

        return {
            "image": image,
            "mask": mask,
            "image_path": image_path,
        }