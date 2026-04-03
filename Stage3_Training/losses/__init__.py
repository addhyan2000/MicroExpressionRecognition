"""
losses — Custom loss functions for Stage 3.

Exports:
    FocalLoss : Handles severe class imbalance in micro-expression datasets.
"""

from .focal_loss import FocalLoss

__all__ = ["FocalLoss"]
