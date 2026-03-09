"""Combined BCE + Dice loss for binary segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        return 1 - (2.0 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )


class CombinedLoss(nn.Module):
    """BCE + Dice combined loss with optional per-task pos_weight."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5,
                 pos_weight: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                task_types: list[str] | None = None) -> torch.Tensor:
        """
        Args:
            logits: raw model output (B, H, W), NOT sigmoid-ed
            target: binary mask (B, H, W), values in [0, 1]
            task_types: list of task names per sample (for per-task weighting)
        """
        # Move pos_weight to same device as logits
        self.bce.pos_weight = self.bce.pos_weight.to(logits.device)

        bce_loss = self.bce(logits, target)
        dice_loss = self.dice(torch.sigmoid(logits), target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
