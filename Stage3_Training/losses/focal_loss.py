"""
focal_loss.py — Focal Loss for Severe Class Imbalance
======================================================

Implements **Focal Loss** from:

    Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    "Focal Loss for Dense Object Detection."
    IEEE International Conference on Computer Vision (ICCV).

Theory:
    Standard Cross-Entropy treats all samples equally, causing the model
    to be dominated by the majority class.  Focal Loss adds a modulating
    factor ``(1 - p_t)^γ`` that:

    • **Down-weights easy, well-classified examples** (high p_t → factor ≈ 0)
    • **Up-weights hard, misclassified examples** (low p_t → factor ≈ 1)

    Formula::

        FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    where:
        - ``p_t`` = model's predicted probability for the true class
        - ``γ`` (gamma)  = focusing parameter (default: 2.0)
        - ``α_t`` (alpha) = per-class weight to handle imbalance

    When γ=0, Focal Loss degenerates to weighted Cross-Entropy.

Micro-Expression Dataset Imbalance (from our data):
    - Negative : 128 samples (41%)
    - Others   : 102 samples (33%)
    - Positive :  47 samples (15%)
    - Surprise :  35 samples (11%)

    Without Focal Loss, the model trivially predicts "Negative" for everything.

Author  : Addhyan
Stage   : 3 — Adversarial Domain Adaptation
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    Parameters
    ----------
    alpha : torch.Tensor or None
        Per-class weights of shape ``[num_classes]``.  If ``None``,
        all classes are weighted equally.
    gamma : float
        Focusing parameter.  Higher values increase the focus on hard
        examples.  Typical range: [0.5, 5.0].  Default: 2.0.
    reduction : str
        Specifies the reduction: ``'none'`` | ``'mean'`` | ``'sum'``.
        Default: ``'mean'``.
    label_smoothing : float
        Optional label smoothing factor ∈ [0, 1).  Helps prevent
        overconfident predictions.  Default: 0.0 (no smoothing).

    Shape:
        - Input  (logits):  ``[B, C]`` where C = number of classes
        - Target (labels):  ``[B]`` integer class indices
        - Output:           scalar (if reduction='mean' or 'sum'),
                            ``[B]`` (if reduction='none')

    Example:
        >>> criterion = FocalLoss(gamma=2.0)
        >>> logits = torch.randn(8, 4)
        >>> labels = torch.randint(0, 4, (8,))
        >>> loss = criterion(logits, labels)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()

        if alpha is not None:
            # Register as buffer (not a parameter — no gradients)
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Focal Loss.

        Parameters
        ----------
        logits : torch.Tensor
            Raw (unnormalized) predictions of shape ``[B, C]``.
        targets : torch.Tensor
            Ground-truth class indices of shape ``[B]``.

        Returns
        -------
        torch.Tensor
            Focal loss value.

        Implementation Notes
        --------------------
        1. We compute log-softmax for numerical stability (avoids
           explicit softmax + log which can cause overflow).
        2. We gather the log-probability of the true class.
        3. The modulating factor ``(1 - p_t)^γ`` is computed from ``p_t``.
        4. Per-class alpha weighting is applied if provided.
        """
        # ── Validate inputs ────────────────────────────────────────
        assert logits.dim() == 2, (
            f"Expected 2D logits [B, C], got {logits.dim()}D: {logits.shape}"
        )
        assert targets.dim() == 1, (
            f"Expected 1D targets [B], got {targets.dim()}D: {targets.shape}"
        )

        num_classes = logits.size(1)
        batch_size = logits.size(0)

        # ── Step 1: Compute log-probabilities ──────────────────────
        log_probs = F.log_softmax(logits, dim=1)  # [B, C]

        # ── Step 2: Gather log-probability of the true class ───────
        # targets: [B] → [B, 1] for gather, then squeeze back to [B]
        targets_onehot_idx = targets.unsqueeze(1)  # [B, 1]
        log_pt = log_probs.gather(1, targets_onehot_idx).squeeze(1)  # [B]

        # ── Step 3: Compute p_t and modulating factor ──────────────
        pt = log_pt.exp()  # [B]
        focal_weight = (1.0 - pt) ** self.gamma  # [B]

        # ── Step 4: Apply label smoothing if requested ─────────────
        if self.label_smoothing > 0 and num_classes > 1:
            # Smooth: true class gets (1 - ε) + ε/C, others get ε/C
            # This is equivalent to mixing the true loss with uniform
            smooth_loss = -log_probs.mean(dim=1)  # [B]
            nll_loss = -log_pt  # [B]
            loss = (1.0 - self.label_smoothing) * nll_loss + \
                   self.label_smoothing * smooth_loss  # [B]
        else:
            # Standard negative log-likelihood
            loss = -log_pt  # [B]

        # ── Step 5: Apply focal modulation ─────────────────────────
        loss = focal_weight * loss  # [B]

        # ── Step 6: Apply alpha weighting ──────────────────────────
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)  # [B]
            loss = alpha_t * loss  # [B]

        # ── Step 7: Reduction ──────────────────────────────────────
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def extra_repr(self) -> str:
        """String representation for print(model)."""
        alpha_str = (
            f"alpha=[{', '.join(f'{a:.4f}' for a in self.alpha)}]"
            if self.alpha is not None
            else "alpha=None"
        )
        return (
            f"gamma={self.gamma}, "
            f"{alpha_str}, "
            f"reduction='{self.reduction}', "
            f"label_smoothing={self.label_smoothing}"
        )


# ─────────────────────────────────────────────────────────────────────
# Standalone Verification
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  Focal Loss — Standalone Verification")
    print("=" * 70)

    torch.manual_seed(42)

    # ── Test 1: Basic forward pass ──────────────────────────────────
    criterion = FocalLoss(gamma=2.0)
    logits = torch.randn(8, 4)
    targets = torch.randint(0, 4, (8,))
    loss = criterion(logits, targets)
    assert loss.dim() == 0, "Loss should be scalar with mean reduction"
    assert loss.item() > 0, "Loss should be positive"
    print(f"✓ Basic forward: loss = {loss.item():.4f}")

    # ── Test 2: Gamma=0 should approximate CE ──────────────────────
    criterion_focal = FocalLoss(gamma=0.0)
    criterion_ce = nn.CrossEntropyLoss()

    logits2 = torch.randn(16, 4)
    targets2 = torch.randint(0, 4, (16,))

    loss_focal = criterion_focal(logits2, targets2)
    loss_ce = criterion_ce(logits2, targets2)

    assert torch.allclose(loss_focal, loss_ce, atol=1e-5), (
        f"Gamma=0 Focal Loss should equal CE: focal={loss_focal:.6f} vs ce={loss_ce:.6f}"
    )
    print(f"✓ Gamma=0 matches CrossEntropy: {loss_focal:.6f} ≈ {loss_ce:.6f}")

    # ── Test 3: With alpha weights ──────────────────────────────────
    alpha = torch.tensor([1.0, 2.0, 3.0, 1.5])
    criterion_alpha = FocalLoss(gamma=2.0, alpha=alpha)
    loss_alpha = criterion_alpha(logits, targets)
    assert loss_alpha.item() > 0
    print(f"✓ Alpha-weighted loss: {loss_alpha.item():.4f}")

    # ── Test 4: Gradient flow ───────────────────────────────────────
    logits3 = torch.randn(4, 4, requires_grad=True)
    targets3 = torch.randint(0, 4, (4,))
    loss3 = FocalLoss(gamma=2.0)(logits3, targets3)
    loss3.backward()
    assert logits3.grad is not None, "Gradients should flow through Focal Loss"
    print(f"✓ Gradient flow verified")

    # ── Test 5: Higher gamma → lower loss for confident predictions ─
    # Create confident predictions (high logit for true class)
    confident_logits = torch.zeros(4, 4)
    confident_logits[0, 0] = 10.0
    confident_logits[1, 1] = 10.0
    confident_logits[2, 2] = 10.0
    confident_logits[3, 3] = 10.0
    confident_targets = torch.arange(4)

    loss_g0 = FocalLoss(gamma=0.0)(confident_logits, confident_targets)
    loss_g2 = FocalLoss(gamma=2.0)(confident_logits, confident_targets)
    loss_g5 = FocalLoss(gamma=5.0)(confident_logits, confident_targets)

    assert loss_g5 <= loss_g2 <= loss_g0, (
        "Higher gamma should reduce loss for confident predictions"
    )
    print(f"✓ Gamma scaling: γ=0 ({loss_g0:.6f}) ≥ γ=2 ({loss_g2:.6f}) ≥ γ=5 ({loss_g5:.6f})")

    # ── Test 6: With label smoothing ────────────────────────────────
    criterion_smooth = FocalLoss(gamma=2.0, label_smoothing=0.1)
    loss_smooth = criterion_smooth(logits, targets)
    assert loss_smooth.item() > 0
    print(f"✓ Label smoothing (ε=0.1): {loss_smooth.item():.4f}")

    print(f"\n{'=' * 70}")
    print(f"  ✓ ALL FOCAL LOSS TESTS PASSED")
    print(f"{'=' * 70}")
