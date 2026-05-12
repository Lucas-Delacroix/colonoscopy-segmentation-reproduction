import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int = 352) -> A.Compose:
    """
    Pipeline de augmentation do paper:
    - Espelhamento horizontal e vertical 
    - Movimentação pelos eixos (até 10%)
    - Variação de perspectiva (até 10%)
    - Escala (80 a 120% do tamanho original)
    - Ruído gaussiano
    - Equalização
    - Variações de brilho, contraste e saturação (até 20%)
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
    Validação e teste não recebem augmentation,
    apenas redimensionamento e normalização.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])