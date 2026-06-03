import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.contiguous().view(probs.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)

        intersection = (probs * target).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + target.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, target) + self.dice(logits, target)


class StructureLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = 1 + 5 * torch.abs(
            F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target
        )

        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        weighted_bce = (weights * bce).sum(dim=(2, 3)) / weights.sum(dim=(2, 3))

        probs = torch.sigmoid(logits)
        intersection = ((probs * target) * weights).sum(dim=(2, 3))
        union = ((probs + target) * weights).sum(dim=(2, 3))
        weighted_iou = 1 - (intersection + 1) / (union - intersection + 1)

        return (weighted_bce + weighted_iou).mean()


def get_loss(name: str) -> nn.Module:
    losses = {
        "bce": nn.BCEWithLogitsLoss,
        "dice": DiceLoss,
        "bce_dice": BCEDiceLoss,
        "structure": StructureLoss,
    }

    key = name.lower()
    if key not in losses:
        raise ValueError(f"Unrecognized loss '{name}'. Options: {list(losses)}")
    return losses[key]()
