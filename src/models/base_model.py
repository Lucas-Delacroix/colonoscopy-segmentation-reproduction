from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the predictions segmentation map"""
        ...

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """It can be overwritten by models with customized losses."""
        raise NotImplementedError

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Returns binary mask after threshold"""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            return (probs > threshold).float()