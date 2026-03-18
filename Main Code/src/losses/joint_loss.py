"""Joint Loss Function for Micro-Expression Recognition.

Combines Focal Loss (to handle extreme class imbalance) and Center Loss
(to minimize intra-class variance and pull deep features toward class centers).

Author  : Addhyan Pant
Project : Master's Thesis — Micro-Expression Recognition
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class Focal Loss with correct alpha decoupling.

    Computes raw (unweighted) cross-entropy first to extract the true
    predicted probability p_t, then applies the focal modulating factor
    (1 - p_t)^gamma, and finally multiplies by the per-class alpha weight.

    This avoids the fatal bug of passing alpha into F.cross_entropy, which
    corrupts p_t by raising it to the power of the alpha weight.

    Parameters
    ----------
    alpha : torch.Tensor, optional
        A 1D tensor of per-class weights (length = num_classes).
    gamma : float, optional
        Focusing parameter (default: 2.0).
    reduction : str, optional
        Specifies the reduction: 'none' | 'mean' | 'sum' (default: 'mean').
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the Focal Loss.

        Parameters
        ----------
        logits : torch.Tensor
            Raw unactivated predictions. Shape: ``(Batch, num_classes)``.
        targets : torch.Tensor
            Ground truth class indices. Shape: ``(Batch,)``.

        Returns
        -------
        torch.Tensor
            Computed Focal Loss.
        """
        # Step 1: Compute raw CE *without* alpha to get true log(p_t)
        log_pt = -F.cross_entropy(logits, targets, reduction="none")

        # Step 2: Extract p_t and defensively clamp to prevent NaN gradients
        # when the network becomes overconfident (p_t ≈ 1.0)
        pt = torch.exp(log_pt)
        pt = pt.clamp(min=1e-5, max=1.0 - 1e-5)

        # Step 3: Apply the focal modulating factor
        focal_loss = ((1.0 - pt) ** self.gamma) * (-log_pt)

        # Step 4: Apply per-class alpha *after* focal modulation.
        # Force device sync to handle GPU targets with CPU-resident module.
        if self.alpha is not None:
            # Defensive bounds check — catch data-loader bugs early
            assert (targets < self.alpha.size(0)).all(), (
                f"Target label exceeds alpha weight dimensions! "
                f"Got max target {targets.max().item()} but alpha has "
                f"only {self.alpha.size(0)} classes."
            )
            alpha_t = self.alpha.to(targets.device).gather(0, targets)
            focal_loss = focal_loss * alpha_t

        # Step 5: Reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class CenterLoss(nn.Module):
    """Center Loss for minimizing intra-class feature variance.

    Penalizes the distance between deep features and their corresponding
    class centers.  Per-sample distances are normalized by the class
    frequency within the batch so that minority classes push their centers
    just as strongly as majority classes.

    Centers are initialized with a small scale (0.01) to keep the initial
    loss magnitude comparable to Focal Loss and prevent loss explosion in
    128-dimensional embedding spaces.

    Parameters
    ----------
    num_classes : int
        Number of classes.
    feat_dim : int, optional
        Dimensionality of the deep features (default: 128).
    """

    def __init__(self, num_classes: int, feat_dim: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        # Learnable class centers — scaled down to prevent loss explosion
        # in high-dimensional spaces during the first epoch
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim) * 0.01)

    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the Center Loss.

        Parameters
        ----------
        features : torch.Tensor
            Deep features before the classification layer. Shape: ``(Batch, feat_dim)``.
        targets : torch.Tensor
            Ground truth class indices. Shape: ``(Batch,)``.

        Returns
        -------
        torch.Tensor
            Computed Center Loss.
        """
        targets = targets.long()

        # Count per-class frequency in this batch for fair normalization
        counts = torch.bincount(targets, minlength=self.num_classes)
        centers_count = counts.gather(0, targets).float().clamp(min=1.0)

        # Select the centers corresponding to the target labels
        centers_batch = self.centers.index_select(0, targets)

        # Calculate squared Euclidean distance: ||x_i - c_{y_i}||^2
        diff = features - centers_batch

        # Divide per-sample squared distance by its class frequency
        # in the batch, then take the batch mean / 2.
        # This ensures minority classes push their centers just as
        # strongly as majority classes without breaking the external
        # SGD optimizer's step-size semantics.
        loss = (diff.pow(2).sum(dim=1) / centers_count).mean() / 2.0

        return loss


class JointLoss(nn.Module):
    """Composite loss combining Focal Loss and Center Loss.

    Calculates: ``L_total = L_focal + lambda * L_center``

    Parameters
    ----------
    num_classes : int
        Number of classes.
    feat_dim : int, optional
        Dimensionality of the deep features (default: 128).
    alpha : torch.Tensor, optional
        Class weights for Focal Loss.
    gamma : float, optional
        Gamma parameter for Focal Loss (default: 2.0).
    lambda_c : float, optional
        Weight for Center Loss in the total loss sum (default: 0.1).
    """

    def __init__(
        self,
        num_classes: int,
        feat_dim: int = 128,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        lambda_c: float = 0.1,
    ):
        super().__init__()
        self.lambda_c = lambda_c
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.center_loss = CenterLoss(num_classes=num_classes, feat_dim=feat_dim)

    def forward(
        self,
        logits: torch.Tensor,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the joint loss.

        Parameters
        ----------
        logits : torch.Tensor
            Raw class predictions. Shape: ``(Batch, num_classes)``.
        features : torch.Tensor
            Deep feature embeddings. Shape: ``(Batch, feat_dim)``.
        targets : torch.Tensor
            Ground truth class indices. Shape: ``(Batch,)``.

        Returns
        -------
        torch.Tensor
            Total combined loss.
        """
        l_focal = self.focal_loss(logits, targets)
        l_center = self.center_loss(features, targets)
        return l_focal + self.lambda_c * l_center
