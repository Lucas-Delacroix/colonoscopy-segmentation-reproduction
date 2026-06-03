import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int = 352) -> A.Compose:
    """
    Augmentation pipeline from the paper:
    - Horizontal and vertical flipping
    - Axis shifts (up to 10%)
    - Perspective variation (up to 10%)
    - Scale (80% to 120% of the original size)
    - Gaussian noise
    - Equalization
    - Brightness, contrast, and saturation variation (up to 20%)
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=0,
            p=0.5,
        ),
        A.Perspective(scale=(0.05, 0.1), p=0.5),
        A.GaussNoise(p=0.3),
        A.Equalize(p=0.3),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.0,
            p=0.5,
        ),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 352) -> A.Compose:
    """
    Validation and test do not receive augmentation,
    only resizing and normalization.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])
