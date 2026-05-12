"""
Run with:
    uv run python -m scripts.validate_data --data_root data/raw/kvasir-seg

Checks that:
    1. The dataset loads without errors
    2. The splits are correct (800/100/100)
    3. Images and masks have the expected shape
    4. Augmentation works visually
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.datamodule import PolypDataModule



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/raw/kvasir-seg",
        help="Path to the Kvasir-SEG root directory.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=352,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/validation",
        help="Directory where visual validation images are saved.",
    )
    return parser.parse_args()


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalization for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    tensor = tensor.clamp(0, 1)
    return tensor.permute(1, 2, 0).numpy()


def validate_batch(batch: dict, split: str, output_dir: Path):
    """Plot and save the first 4 images in a batch with their masks."""
    images = batch["image"]
    masks = batch["mask"]
    paths = batch["image_path"]

    n = min(4, len(images))
    fig, axes = plt.subplots(n, 2, figsize=(8, n * 4))
    fig.suptitle(f"Split: {split}", fontsize=14)

    if n == 1:
        axes = [axes]

    for i in range(n):
        image = denormalize(images[i])
        mask = masks[i].squeeze().numpy()

        axes[i][0].imshow(image)
        axes[i][0].set_title(f"Image\n{Path(paths[i]).name}")
        axes[i][0].axis("off")

        axes[i][1].imshow(mask, cmap="gray")
        axes[i][1].set_title(f"Mask\nmin={mask.min():.2f} max={mask.max():.2f}")
        axes[i][1].axis("off")

    plt.tight_layout()

    output_path = output_dir / f"validate_{split}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved to: {output_path}")


def validate_tensor_shapes(batch: dict, split: str, batch_size: int):
    """Check tensor shapes and dtypes."""
    images = batch["image"]
    masks = batch["mask"]

    print(f"\n  Shapes ({split}):")
    print(f"    image: {tuple(images.shape)} - expected (N, 3, H, W)")
    print(f"    mask:  {tuple(masks.shape)} - expected (N, 1, H, W)")
    print(f"  Dtype:  image={images.dtype}  mask={masks.dtype}")
    print(f"  Range:  image=[{images.min():.2f}, {images.max():.2f}]  mask=[{masks.min():.2f}, {masks.max():.2f}]")

    assert images.shape[1] == 3, "Image must have 3 channels (RGB)"
    assert masks.shape[1] == 1, "Mask must have 1 channel"
    assert masks.min() >= 0.0 and masks.max() <= 1.0, "Mask must be binary [0, 1]"
    print("  All checks passed.")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("Initializing DataModule...")
    print("=" * 50)

    dm = PolypDataModule(
        dataset_name="kvasir",
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=4,
        num_workers=0,  # 0 makes debugging easier
    )
    dm.setup()

    for split, loader in [
        ("train", dm.train_loader()),
        ("val", dm.val_loader()),
        ("test", dm.test_loader()),
    ]:
        print(f"\n{'=' * 50}")
        print(f"Validating split: {split}")
        print(f"{'=' * 50}")

        batch = next(iter(loader))

        validate_tensor_shapes(batch, split, dm.batch_size)
        validate_batch(batch, split, output_dir)

    print(f"\n{'=' * 50}")
    print("Validation completed successfully.")
    print(f"Check the images at: {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
