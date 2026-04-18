"""
losses — Custom loss functions for Stage 3.

Exports:
    FocalLoss  : Handles severe class imbalance in micro-expression datasets.
    SupConLoss : Supervised Contrastive Loss for emotion-aware feature clustering.
"""

from .focal_loss import FocalLoss
from .supcon_loss import SupConLoss

__all__ = ["FocalLoss", "SupConLoss"]
