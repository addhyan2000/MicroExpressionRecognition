"""
supcon_loss.py — Supervised Contrastive Loss (SupCon)
=====================================================

Implements **Supervised Contrastive Loss** from:

    Khosla, P., Teterwak, P., Wang, C., et al. (2020).
    "Supervised Contrastive Learning."
    Advances in Neural Information Processing Systems (NeurIPS).

Theory:
    Standard cross-entropy operates on logits and treats each sample
    independently.  SupCon instead operates on *normalized embeddings*
    and exploits the label structure within a batch:

    For each anchor ``i``, we contrast it against every other sample
    ``j`` in the batch.  Samples sharing the same label as ``i`` are
    **positives** (pulled closer); all others are **negatives** (pushed
    apart).

    Formula::

        L_supcon = Σ_i  (-1 / |P(i)|)  Σ_{p ∈ P(i)}
                   log [ exp(z_i · z_p / τ) / Σ_{a ∈ A(i)} exp(z_i · z_a / τ) ]

    where:
        - ``z_i``   = L2-normalized projection of sample i
        - ``P(i)``  = set of positives (same label as i, excluding i)
        - ``A(i)``  = set of all samples excluding i
        - ``τ``     = temperature (controls cluster tightness)

    Lower τ → sharper contrast → tighter clusters (but harder to train).
    Default τ = 0.1 is standard in the literature.

Purpose (in our pipeline):
    SupCon anchors the Transformer's feature space with emotion-aware
    clustering.  This prevents the GRL's adversarial gradient from
    collapsing the feature space when λ → 1.0 at late training.

Author  : Addhyan
Stage   : 3 — Adversarial Domain Adaptation
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss for multi-class feature clustering.

    Computes the contrastive loss over a batch of L2-normalized
    projection-head embeddings, using their ground-truth emotion
    labels to define positive/negative pairs.

    Parameters
    ----------
    temperature : float
        Temperature scaling factor τ for the cosine similarity.
        Lower values produce sharper, tighter clusters.
        Default: 0.1 (standard from Khosla et al., 2020).

    Shape
    -----
    - features : ``[B, D]`` — L2-normalized embeddings from the
      projection head (D = projection dimension, e.g. 64).
    - labels   : ``[B]`` — integer class labels.
    - output   : scalar loss.

    Example
    -------
    >>> criterion = SupConLoss(temperature=0.1)
    >>> features = torch.randn(8, 64)
    >>> features = nn.functional.normalize(features, dim=1)
    >>> labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    >>> loss = criterion(features, labels)
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()

        if temperature <= 0:
            raise ValueError(
                f"Temperature must be positive, got {temperature}"
            )

        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the Supervised Contrastive Loss.

        Parameters
        ----------
        features : torch.Tensor
            L2-normalized embeddings of shape ``[B, D]``.
            **Must be pre-normalized** (the projection head handles this).
        labels : torch.Tensor
            Ground-truth class labels of shape ``[B]``.

        Returns
        -------
        torch.Tensor
            Scalar contrastive loss value.

        Implementation Notes
        --------------------
        We follow the numerically stable formulation:
        1. Compute the full similarity matrix S = Z @ Z^T / τ
        2. Subtract max for numerical stability (log-sum-exp trick)
        3. Mask out self-contrast (diagonal)
        4. Build positive mask from labels
        5. Compute log-softmax over negatives + positives
        6. Average over positives for each anchor
        """
        device = features.device
        batch_size = features.shape[0]

        # ── Guard: need at least 2 samples for contrastive loss ─────
        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # ── Validate inputs ─────────────────────────────────────────
        assert features.dim() == 2, (
            f"Expected 2D features [B, D], got {features.dim()}D: "
            f"{features.shape}"
        )
        assert labels.dim() == 1, (
            f"Expected 1D labels [B], got {labels.dim()}D: {labels.shape}"
        )
        assert features.shape[0] == labels.shape[0], (
            f"Batch size mismatch: features={features.shape[0]}, "
            f"labels={labels.shape[0]}"
        )

        # ────────────────────────────────────────────────────────────
        # Step 1: Pairwise cosine similarity matrix
        # ────────────────────────────────────────────────────────────
        # S[i, j] = (z_i · z_j) / τ
        # Since features are L2-normalized, dot product = cosine sim.
        # Shape: [B, B]
        similarity_matrix = torch.matmul(
            features, features.T
        ) / self.temperature

        # ────────────────────────────────────────────────────────────
        # Step 2: Numerical stability — subtract row-wise max
        # ────────────────────────────────────────────────────────────
        # This is the log-sum-exp trick: subtracting the max prevents
        # exp() overflow without changing the softmax result.
        sim_max, _ = similarity_matrix.max(dim=1, keepdim=True)
        similarity_matrix = similarity_matrix - sim_max.detach()

        # ────────────────────────────────────────────────────────────
        # Step 3: Self-contrast mask (exclude diagonal)
        # ────────────────────────────────────────────────────────────
        # self_mask[i, j] = 1 if i ≠ j, else 0
        # We never want a sample contrasted against itself.
        self_mask = ~torch.eye(
            batch_size, dtype=torch.bool, device=device
        )

        # ────────────────────────────────────────────────────────────
        # Step 4: Positive mask — same-label pairs (excluding self)
        # ────────────────────────────────────────────────────────────
        # positive_mask[i, j] = 1 if label_i == label_j AND i ≠ j
        labels_column = labels.unsqueeze(0)  # [1, B]
        labels_row = labels.unsqueeze(1)     # [B, 1]
        positive_mask = (labels_row == labels_column) & self_mask

        # ────────────────────────────────────────────────────────────
        # Step 5: Count positives per anchor
        # ────────────────────────────────────────────────────────────
        # num_positives[i] = number of same-class samples for anchor i
        num_positives = positive_mask.sum(dim=1)  # [B]

        # Guard: anchors with zero positives cannot contribute to loss.
        # This happens when a class has only 1 sample in the batch.
        has_positives = num_positives > 0

        # If no anchor has any positive, return zero loss
        if not has_positives.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # ────────────────────────────────────────────────────────────
        # Step 6: Denominator — log-sum-exp over all non-self pairs
        # ────────────────────────────────────────────────────────────
        # For each anchor i, sum exp(S[i, j]) over all j ≠ i
        # We mask out the diagonal by zeroing those entries before exp
        exp_similarity = torch.exp(similarity_matrix) * self_mask.float()
        log_denominator = torch.log(
            exp_similarity.sum(dim=1, keepdim=True) + 1e-12
        )  # [B, 1]  — add epsilon for numerical safety

        # ────────────────────────────────────────────────────────────
        # Step 7: Log-probability of each pair
        # ────────────────────────────────────────────────────────────
        # log_prob[i, j] = S[i, j] - log(Σ_{a≠i} exp(S[i, a]))
        # This is the log-softmax over all non-self pairs.
        log_prob = similarity_matrix - log_denominator  # [B, B]

        # ────────────────────────────────────────────────────────────
        # Step 8: Average log-probability over positives per anchor
        # ────────────────────────────────────────────────────────────
        # For anchor i: mean_log_prob_pos[i] =
        #   (1 / |P(i)|) * Σ_{p ∈ P(i)} log_prob[i, p]
        #
        # We zero-out non-positive entries, sum, then divide by count.
        positive_log_prob = (log_prob * positive_mask.float()).sum(
            dim=1
        )  # [B]

        # Avoid division by zero for anchors with no positives
        safe_num_positives = num_positives.float().clamp(min=1.0)
        mean_log_prob_pos = positive_log_prob / safe_num_positives  # [B]

        # ────────────────────────────────────────────────────────────
        # Step 9: Final loss — negate and average over valid anchors
        # ────────────────────────────────────────────────────────────
        # L = -mean over anchors that have positives
        loss = -(mean_log_prob_pos[has_positives]).mean()

        return loss

    def extra_repr(self) -> str:
        """String representation for print(model)."""
        return f"temperature={self.temperature}"


# ─────────────────────────────────────────────────────────────────────
# Standalone Verification
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch.nn.functional as F

    print("=" * 70)
    print("  SupConLoss — Standalone Verification")
    print("=" * 70)

    torch.manual_seed(42)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Test 1: Basic forward pass ──────────────────────────────────
    criterion = SupConLoss(temperature=0.1).to(DEVICE)
    feats = F.normalize(torch.randn(8, 64, device=DEVICE), dim=1)
    lbls = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], device=DEVICE)
    loss = criterion(feats, lbls)
    assert loss.dim() == 0, "Loss should be scalar"
    assert loss.item() > 0, "Loss should be positive"
    print(f"✓ Basic forward: loss = {loss.item():.4f}")

    # ── Test 2: Gradient flow ───────────────────────────────────────
    feats2_raw = torch.randn(8, 64, device=DEVICE, requires_grad=True)
    feats2 = F.normalize(feats2_raw, dim=1)
    lbls2 = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], device=DEVICE)
    loss2 = criterion(feats2, lbls2)
    loss2.backward()
    assert feats2_raw.grad is not None, "Gradients should flow to leaf tensor"
    assert not torch.isnan(feats2_raw.grad).any(), "Gradients should not be NaN"
    print(f"✓ Gradient flow verified (no NaN)")

    # ── Test 3: Perfect clustering → lower loss ─────────────────────
    # Create features where same-class samples are identical
    perfect_feats = torch.zeros(8, 64, device=DEVICE)
    perfect_feats[0:2, 0] = 1.0   # Class 0 cluster at dim 0
    perfect_feats[2:4, 1] = 1.0   # Class 1 cluster at dim 1
    perfect_feats[4:6, 2] = 1.0   # Class 2 cluster at dim 2
    perfect_feats[6:8, 3] = 1.0   # Class 3 cluster at dim 3
    perfect_feats = F.normalize(perfect_feats, dim=1)

    random_feats = F.normalize(torch.randn(8, 64, device=DEVICE), dim=1)

    loss_perfect = criterion(perfect_feats, lbls)
    loss_random = criterion(random_feats, lbls)

    assert loss_perfect.item() < loss_random.item(), (
        f"Perfect clustering should have lower loss: "
        f"{loss_perfect.item():.4f} vs {loss_random.item():.4f}"
    )
    print(
        f"✓ Perfect clustering ({loss_perfect.item():.4f}) "
        f"< random ({loss_random.item():.4f})"
    )

    # ── Test 4: Batch size = 1 returns zero ─────────────────────────
    single_feat = F.normalize(torch.randn(1, 64, device=DEVICE), dim=1)
    single_lbl = torch.tensor([0], device=DEVICE)
    loss_single = criterion(single_feat, single_lbl)
    assert loss_single.item() == 0.0, "Single sample should return 0 loss"
    print(f"✓ Single-sample guard: loss = {loss_single.item():.4f}")

    # ── Test 5: All same label ──────────────────────────────────────
    same_feats = F.normalize(torch.randn(4, 64, device=DEVICE), dim=1)
    same_lbls = torch.zeros(4, dtype=torch.long, device=DEVICE)
    loss_same = criterion(same_feats, same_lbls)
    assert loss_same.item() >= 0, "Loss should be non-negative"
    print(f"✓ All-same-label: loss = {loss_same.item():.4f}")

    # ── Test 6: No valid positives returns zero ─────────────────────
    unique_feats = F.normalize(torch.randn(4, 64, device=DEVICE), dim=1)
    unique_lbls = torch.tensor([0, 1, 2, 3], device=DEVICE)
    loss_unique = criterion(unique_feats, unique_lbls)
    assert loss_unique.item() == 0.0, "No positives should return 0 loss"
    print(f"✓ No-positives guard: loss = {loss_unique.item():.4f}")

    # ── Test 7: Temperature sensitivity ─────────────────────────────
    low_temp = SupConLoss(temperature=0.05)
    high_temp = SupConLoss(temperature=0.5)
    mixed_feats = F.normalize(torch.randn(8, 64, device=DEVICE), dim=1)
    loss_low_t = low_temp(mixed_feats, lbls)
    loss_high_t = high_temp(mixed_feats, lbls)
    print(
        f"✓ Temperature sensitivity: τ=0.05 → {loss_low_t.item():.4f}, "
        f"τ=0.5 → {loss_high_t.item():.4f}"
    )

    print(f"\n{'=' * 70}")
    print(f"  ✓ ALL SUPCON LOSS TESTS PASSED")
    print(f"{'=' * 70}")
