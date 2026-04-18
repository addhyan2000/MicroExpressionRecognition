"""
supcon_loss.py — Supervised Contrastive Loss with Cross-Batch Memory (XBM)
===========================================================================

Implements **Supervised Contrastive Loss** (Khosla et al., 2020) extended
with a **Cross-Batch Memory Bank** (Wang et al., 2020) to solve the
small-physical-batch-size bottleneck in VRAM-constrained video pipelines.

Problem Statement:
    When processing 5D video tensors (e.g., [B, 3, 32, 224, 224]), the
    physical batch size is restricted to B=2-4 to avoid CUDA OOM.
    Standard SupCon fails catastrophically at B=2 because the probability
    of having *any* positive pair (same-class sample) in a batch of 2 is
    extremely low — the loss degrades to near-zero and provides no
    gradient signal for the embedding space.

Solution — Cross-Batch Memory (XBM):
    We maintain a **FIFO queue** of projected features and their labels
    from the most recent ``memory_size`` training samples.  During the
    forward pass, the *anchors* are only the current batch, but the
    *contrastive targets* are the union of [current_batch, memory_bank].

    This effectively increases the number of contrastive comparisons
    from B² to B × (B + M), where M = memory_size, without any increase
    in GPU memory during the forward pass of the backbone (the features
    stored in the bank are .detach()-ed and require no gradient).

Mathematical Formulation:
    Let:
        z_i     = L2-normalized projection of anchor i (current batch)
        Z_curr  = [z_1, ..., z_B]          — current batch features   [B, D]
        Z_mem   = [m_1, ..., m_M]          — memory bank features     [M, D]
        Z_all   = concat(Z_curr, Z_mem)    — all contrastive targets  [B+M, D]
        L_all   = concat(L_curr, L_mem)    — corresponding labels     [B+M]
        τ       = temperature

    For each anchor i in the current batch:

        S[i, j] = (z_i · z_all_j) / τ          for j = 1, ..., B+M

        P(i) = { j | L_all[j] == L_curr[i]  AND  j ≠ i }    (positives)
        A(i) = { j | j ≠ i }                                 (all non-self)

        L_i  = (-1 / |P(i)|) Σ_{p ∈ P(i)}
               log [ exp(S[i,p]) / Σ_{a ∈ A(i)} exp(S[i,a]) ]

        L_supcon = (1/B) Σ_i  L_i        (averaged over valid anchors)

    For numerical stability, we apply the **log-sum-exp trick**:
        S[i,:] ← S[i,:] - max_j(S[i,j])

    This subtracts the row-wise maximum before exponentiation, preventing
    overflow in exp() without changing the softmax result.

Reference:
    - Khosla, P. et al. (2020). "Supervised Contrastive Learning." NeurIPS.
    - Wang, X. et al. (2020). "Cross-Batch Memory for Embedding Learning."
      CVPR.  (Original XBM paper.)

Author  : Addhyan
Stage   : 3 — Adversarial Domain Adaptation
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss with Cross-Batch Memory Bank (XBM).

    Maintains a FIFO queue of L2-normalized projected features and their
    labels from past training batches.  During the forward pass, the
    *anchors* are only the current batch, but the *contrastive targets*
    include both the current batch **and** the memory bank, dramatically
    increasing the effective number of positive and negative pairs.

    Memory Bank Details:
        - Stored as ``register_buffer`` → automatically moves to GPU
          with ``.to(device)`` / ``.cuda()`` and is saved/loaded with
          ``state_dict()`` (but NOT treated as a trainable parameter).
        - Updated with ``features.detach()`` after every **training**
          forward pass (no update during validation/eval to prevent
          information leakage).
        - Operates as a FIFO queue: newest samples are appended,
          oldest samples are dequeued when the bank exceeds
          ``memory_size``.

    Parameters
    ----------
    temperature : float
        Temperature scaling factor τ for cosine similarity.
        Lower values → sharper, tighter clusters (but harder to train).
        Default: 0.1 (standard from Khosla et al., 2020).
    memory_size : int
        Maximum number of samples to store in the memory bank.
        Default: 64.  Good starting point for B=2-4 physical batch size.
        The effective contrastive set becomes B + min(memory_size, steps_so_far × B).
    proj_dim : int
        Dimensionality of the projected features.  Must match the output
        of the projection head.  Default: 64.

    Shape
    -----
    - features : ``[B, D]`` — L2-normalized embeddings from the
      projection head (D = proj_dim).
    - labels   : ``[B]`` — integer class labels.
    - output   : scalar loss tensor (with gradient).

    Example
    -------
    >>> criterion = SupConLoss(temperature=0.1, memory_size=64, proj_dim=64)
    >>> features = torch.randn(4, 64)
    >>> features = nn.functional.normalize(features, dim=1)
    >>> labels = torch.tensor([0, 0, 1, 1])
    >>> loss = criterion(features, labels)

    Notes
    -----
    The memory bank is **not** updated during ``model.eval()`` mode.
    This ensures that validation batches do not pollute the training
    memory and that validation is deterministic w.r.t. the bank state.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        memory_size: int = 64,
        proj_dim: int = 64,
    ) -> None:
        super().__init__()

        if temperature <= 0:
            raise ValueError(
                f"Temperature must be positive, got {temperature}"
            )
        if memory_size < 1:
            raise ValueError(
                f"memory_size must be ≥ 1, got {memory_size}"
            )
        if proj_dim < 1:
            raise ValueError(
                f"proj_dim must be ≥ 1, got {proj_dim}"
            )

        self.temperature = temperature
        self.memory_size = memory_size
        self.proj_dim = proj_dim

        # ── Memory Bank Buffers ──────────────────────────────────────
        # register_buffer ensures:
        #   1. Automatic device transfer with .to() / .cuda()
        #   2. Included in state_dict for checkpoint save/resume
        #   3. NOT treated as trainable parameters (no gradient)
        #
        # feature_bank: [memory_size, proj_dim] — stores detached features
        # label_bank:   [memory_size]           — stores corresponding labels
        # bank_ptr:     scalar                  — current write pointer (FIFO)
        # bank_full:    scalar bool             — whether the bank has wrapped
        #
        # We pre-allocate the full bank to avoid dynamic resizing.
        # Until the bank is full, we track valid entries via bank_ptr.
        self.register_buffer(
            "feature_bank",
            torch.zeros(memory_size, proj_dim),
        )
        self.register_buffer(
            "label_bank",
            torch.full((memory_size,), fill_value=-1, dtype=torch.long),
        )
        self.register_buffer(
            "bank_ptr",
            torch.zeros(1, dtype=torch.long),
        )
        self.register_buffer(
            "bank_full",
            torch.zeros(1, dtype=torch.bool),
        )

    @property
    def _current_bank_size(self) -> int:
        """
        Number of valid entries currently in the memory bank.

        Returns ``memory_size`` once the bank has fully wrapped at least
        once, otherwise returns the current write pointer position.
        """
        if self.bank_full.item():
            return self.memory_size
        return int(self.bank_ptr.item())

    def _get_valid_bank(self) -> tuple:
        """
        Retrieve the valid (non-empty) portion of the memory bank.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            (features, labels) — both sliced to only valid entries.
            features shape: [K, D]  where K = current_bank_size
            labels shape:   [K]
        """
        k = self._current_bank_size
        if k == 0:
            # Empty bank — return zero-length tensors on the correct device
            return (
                self.feature_bank[:0],   # [0, D]
                self.label_bank[:0],     # [0]
            )

        if self.bank_full.item():
            # Bank has wrapped — all entries are valid
            return self.feature_bank, self.label_bank

        # Bank partially filled — only entries [0, ptr) are valid
        return (
            self.feature_bank[:k],
            self.label_bank[:k],
        )

    @torch.no_grad()
    def _enqueue(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Enqueue new features and labels into the FIFO memory bank.

        Uses a circular buffer strategy:
            1. Write new entries starting at ``bank_ptr``.
            2. If entries would exceed ``memory_size``, wrap around.
            3. Advance ``bank_ptr`` by the number of new entries.

        Parameters
        ----------
        features : torch.Tensor
            Detached, L2-normalized features of shape ``[N, D]``.
        labels : torch.Tensor
            Detached labels of shape ``[N]``.

        Notes
        -----
        This method runs inside ``torch.no_grad()`` — no gradient
        tape is created for the enqueue operation.  The ``.detach()``
        is applied by the caller before passing features here.
        """
        batch_size = features.shape[0]
        ptr = int(self.bank_ptr.item())

        if batch_size >= self.memory_size:
            # If the batch is larger than the bank, just take the last
            # memory_size entries.
            self.feature_bank[:] = features[-self.memory_size:]
            self.label_bank[:] = labels[-self.memory_size:]
            self.bank_ptr[0] = 0
            self.bank_full[0] = True
            return

        # How many entries fit before we need to wrap?
        remaining = self.memory_size - ptr

        if batch_size <= remaining:
            # Everything fits without wrapping
            self.feature_bank[ptr : ptr + batch_size] = features
            self.label_bank[ptr : ptr + batch_size] = labels
            new_ptr = ptr + batch_size
            if new_ptr == self.memory_size:
                new_ptr = 0
                self.bank_full[0] = True
        else:
            # Split: fill the end, then wrap around to the beginning
            self.feature_bank[ptr:] = features[:remaining]
            self.label_bank[ptr:] = labels[:remaining]

            overflow = batch_size - remaining
            self.feature_bank[:overflow] = features[remaining:]
            self.label_bank[:overflow] = labels[remaining:]

            new_ptr = overflow
            self.bank_full[0] = True

        self.bank_ptr[0] = new_ptr

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Supervised Contrastive Loss with Cross-Batch Memory.

        The forward pass proceeds in three stages:

        Stage 1 — Build Extended Target Set:
            Concatenate the current batch features/labels with valid
            entries from the memory bank to form the full contrastive
            target set.

        Stage 2 — Compute SupCon Loss:
            For each anchor in the current batch, compute the InfoNCE
            loss against the extended target set using the log-sum-exp
            trick for numerical stability.

        Stage 3 — Update Memory Bank:
            Enqueue the current batch's detached features and labels
            into the FIFO memory bank (training mode only).

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
            Scalar contrastive loss value (with gradient).

        Implementation Notes
        --------------------
        1. The similarity matrix has shape [B, B+M] where:
           - B = current batch size (anchors, rows)
           - M = current valid memory bank size (extra targets, cols)
        2. The self-contrast mask only affects the [B, B] top-left block
           (diagonal), since memory bank entries are always "other" samples.
        3. The positive mask is built from extended labels: anchors match
           against both current-batch and memory-bank labels.
        4. The memory bank is updated AFTER loss computation so that the
           current batch's own features appear in the bank for *future*
           batches, not the current one's denominator (which already
           includes them directly in the current-batch portion).
        """
        device = features.device
        batch_size = features.shape[0]

        # ── Guard: need at least 2 samples for contrastive loss ─────
        if batch_size < 2:
            # Still update the bank even for batch_size=1 during training
            if self.training:
                self._enqueue(features.detach(), labels.detach())
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

        # ════════════════════════════════════════════════════════════
        # Stage 1: Build the extended contrastive target set
        # ════════════════════════════════════════════════════════════
        #
        # Anchors      = features (current batch):          [B, D]
        # Targets      = concat(features, memory_bank):     [B+M, D]
        # Target labels = concat(labels, memory_labels):    [B+M]
        #
        # The memory bank features are .detach()-ed (no gradient)
        # so they do not affect the backward pass of the backbone.

        bank_features, bank_labels = self._get_valid_bank()
        mem_size = bank_features.shape[0]  # 0 if bank is empty

        if mem_size > 0:
            # Concatenate current batch with memory bank
            # bank_features are already detached (stored with .detach())
            all_targets = torch.cat(
                [features, bank_features.detach()], dim=0,
            )  # [B + M, D]
            all_labels = torch.cat(
                [labels, bank_labels.detach()], dim=0,
            )  # [B + M]
        else:
            # First step: memory bank is empty — fall back to batch-only
            all_targets = features       # [B, D]
            all_labels = labels          # [B]

        total_targets = all_targets.shape[0]  # B + M (or just B)

        # ════════════════════════════════════════════════════════════
        # Stage 2: Supervised Contrastive Loss (InfoNCE)
        # ════════════════════════════════════════════════════════════

        # ────────────────────────────────────────────────────────────
        # Step 2a: Pairwise cosine similarity (anchors × targets)
        # ────────────────────────────────────────────────────────────
        # S[i, j] = (z_i · z_all_j) / τ
        # Shape: [B, B+M]
        # Since features are L2-normalized, matmul = cosine similarity.
        similarity_matrix = torch.matmul(
            features, all_targets.T,
        ) / self.temperature  # [B, B+M]

        # ────────────────────────────────────────────────────────────
        # Step 2b: Numerical stability — subtract row-wise max
        # ────────────────────────────────────────────────────────────
        # log-sum-exp trick: subtract the maximum from each row.
        # This prevents exp() overflow without changing softmax.
        #
        # IMPORTANT: We .detach() the max so it does not affect
        # the gradient computation.  The subtraction is a constant
        # shift per row and is mathematically neutral for softmax.
        sim_max, _ = similarity_matrix.max(dim=1, keepdim=True)
        similarity_matrix = similarity_matrix - sim_max.detach()

        # ────────────────────────────────────────────────────────────
        # Step 2c: Self-contrast mask (exclude diagonal of anchor block)
        # ────────────────────────────────────────────────────────────
        # Shape: [B, B+M]
        #
        # self_mask[i, j] = False only when i == j AND j < B
        # (i.e., the anchor vs. itself in the current batch).
        # All memory bank columns (j ≥ B) are always valid targets.
        #
        # We build this as: all ones, then zero out the diagonal
        # of the [B, B] top-left sub-block.
        self_mask = torch.ones(
            batch_size, total_targets, dtype=torch.bool, device=device,
        )
        self_mask[:batch_size, :batch_size].fill_diagonal_(False)

        # ────────────────────────────────────────────────────────────
        # Step 2d: Positive mask — same-label pairs (excluding self)
        # ────────────────────────────────────────────────────────────
        # positive_mask[i, j] = 1 iff:
        #   - labels_curr[i] == all_labels[j]
        #   - AND j ≠ i (ensured by AND-ing with self_mask)
        #
        # Shape: [B, B+M]
        anchor_labels = labels.unsqueeze(1)      # [B, 1]
        target_labels = all_labels.unsqueeze(0)   # [1, B+M]
        positive_mask = (anchor_labels == target_labels) & self_mask

        # ────────────────────────────────────────────────────────────
        # Step 2e: Count positives per anchor
        # ────────────────────────────────────────────────────────────
        num_positives = positive_mask.sum(dim=1)  # [B]

        # Anchors with zero positives cannot contribute to loss.
        has_positives = num_positives > 0

        if not has_positives.any():
            # No positive pairs found (even with memory bank) — return
            # zero loss but still update the bank.
            if self.training:
                self._enqueue(features.detach(), labels.detach())
            return torch.tensor(0.0, device=device, requires_grad=True)

        # ────────────────────────────────────────────────────────────
        # Step 2f: Denominator — log-sum-exp over all non-self targets
        # ────────────────────────────────────────────────────────────
        # For anchor i: Σ_{j ∈ A(i)} exp(S[i,j])
        # where A(i) = all targets except self (self_mask = True)
        #
        # We zero out the self-entry by multiplying with self_mask.
        exp_similarity = torch.exp(similarity_matrix) * self_mask.float()
        log_denominator = torch.log(
            exp_similarity.sum(dim=1, keepdim=True) + 1e-12
        )  # [B, 1]  — epsilon for numerical safety

        # ────────────────────────────────────────────────────────────
        # Step 2g: Log-probability of each pair
        # ────────────────────────────────────────────────────────────
        # log_prob[i, j] = S[i, j] - log(Σ_{a ∈ A(i)} exp(S[i, a]))
        log_prob = similarity_matrix - log_denominator  # [B, B+M]

        # ────────────────────────────────────────────────────────────
        # Step 2h: Average log-probability over positives per anchor
        # ────────────────────────────────────────────────────────────
        # For anchor i:
        #   mean_log_prob_pos[i] = (1/|P(i)|) Σ_{p ∈ P(i)} log_prob[i, p]
        positive_log_prob = (log_prob * positive_mask.float()).sum(
            dim=1,
        )  # [B]

        safe_num_positives = num_positives.float().clamp(min=1.0)
        mean_log_prob_pos = positive_log_prob / safe_num_positives  # [B]

        # ────────────────────────────────────────────────────────────
        # Step 2i: Final loss — negate and average over valid anchors
        # ────────────────────────────────────────────────────────────
        loss = -(mean_log_prob_pos[has_positives]).mean()

        # ════════════════════════════════════════════════════════════
        # Stage 3: Update Memory Bank (training mode only)
        # ════════════════════════════════════════════════════════════
        #
        # We update the bank AFTER computing the loss so that:
        #   1. The current batch's features are already included in the
        #      target set (as the first B columns), not double-counted.
        #   2. Future batches will benefit from these features.
        #
        # During validation (self.training = False), we do NOT update
        # the bank to prevent information leakage from val → train.
        if self.training:
            self._enqueue(features.detach(), labels.detach())

        return loss

    def extra_repr(self) -> str:
        """String representation for print(model)."""
        return (
            f"temperature={self.temperature}, "
            f"memory_size={self.memory_size}, "
            f"proj_dim={self.proj_dim}, "
            f"bank_fill={self._current_bank_size}/{self.memory_size}"
        )


# ─────────────────────────────────────────────────────────────────────
# Standalone Verification
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch.nn.functional as F

    print("=" * 70)
    print("  SupConLoss + XBM — Standalone Verification")
    print("=" * 70)

    torch.manual_seed(42)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Test 1: Basic forward pass ──────────────────────────────────
    criterion = SupConLoss(temperature=0.1, memory_size=32, proj_dim=64).to(DEVICE)
    feats = F.normalize(torch.randn(8, 64, device=DEVICE), dim=1)
    lbls = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], device=DEVICE)
    criterion.train()
    loss = criterion(feats, lbls)
    assert loss.dim() == 0, "Loss should be scalar"
    assert loss.item() > 0, "Loss should be positive"
    print(f"✓ Basic forward: loss = {loss.item():.4f}")

    # ── Test 2: Memory bank was populated ───────────────────────────
    assert criterion._current_bank_size == 8, (
        f"Expected 8 entries in bank, got {criterion._current_bank_size}"
    )
    print(f"✓ Memory bank populated: {criterion._current_bank_size}/32 entries")

    # ── Test 3: Second batch leverages memory bank ──────────────────
    feats2 = F.normalize(torch.randn(4, 64, device=DEVICE), dim=1)
    lbls2 = torch.tensor([0, 1, 2, 3], device=DEVICE)
    loss2 = criterion(feats2, lbls2)
    assert loss2.item() > 0, "Loss with memory bank should be positive"
    assert criterion._current_bank_size == 12, (
        f"Expected 12 entries in bank, got {criterion._current_bank_size}"
    )
    print(f"✓ Memory-augmented forward: loss = {loss2.item():.4f}, "
          f"bank = {criterion._current_bank_size}/32")

    # ── Test 4: Gradient flow ───────────────────────────────────────
    criterion_grad = SupConLoss(temperature=0.1, memory_size=16, proj_dim=64).to(DEVICE)
    criterion_grad.train()
    feats3_raw = torch.randn(8, 64, device=DEVICE, requires_grad=True)
    feats3 = F.normalize(feats3_raw, dim=1)
    lbls3 = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], device=DEVICE)
    loss3 = criterion_grad(feats3, lbls3)
    loss3.backward()
    assert feats3_raw.grad is not None, "Gradients should flow to leaf tensor"
    assert not torch.isnan(feats3_raw.grad).any(), "Gradients should not be NaN"
    print(f"✓ Gradient flow verified (no NaN)")

    # ── Test 5: Empty bank (first step) works correctly ─────────────
    criterion_empty = SupConLoss(temperature=0.1, memory_size=16, proj_dim=64).to(DEVICE)
    criterion_empty.train()
    assert criterion_empty._current_bank_size == 0, "Bank should start empty"
    feats4 = F.normalize(torch.randn(4, 64, device=DEVICE), dim=1)
    lbls4 = torch.tensor([0, 0, 1, 1], device=DEVICE)
    loss4 = criterion_empty(feats4, lbls4)
    assert loss4.item() > 0, "First-step loss should still work"
    print(f"✓ Empty bank (first step): loss = {loss4.item():.4f}")

    # ── Test 6: FIFO overflow — oldest entries are evicted ──────────
    criterion_fifo = SupConLoss(temperature=0.1, memory_size=10, proj_dim=64).to(DEVICE)
    criterion_fifo.train()
    for i in range(5):
        f_i = F.normalize(torch.randn(4, 64, device=DEVICE), dim=1)
        l_i = torch.tensor([0, 1, 2, 3], device=DEVICE)
        _ = criterion_fifo(f_i, l_i)
    # 5 batches × 4 = 20 entries attempted, but bank only holds 10
    assert criterion_fifo._current_bank_size == 10, (
        f"FIFO should cap at memory_size=10, got {criterion_fifo._current_bank_size}"
    )
    print(f"✓ FIFO overflow: bank = {criterion_fifo._current_bank_size}/10 (capped)")

    # ── Test 7: Eval mode does NOT update bank ──────────────────────
    criterion_eval = SupConLoss(temperature=0.1, memory_size=16, proj_dim=64).to(DEVICE)
    criterion_eval.train()
    f_train = F.normalize(torch.randn(4, 64, device=DEVICE), dim=1)
    l_train = torch.tensor([0, 0, 1, 1], device=DEVICE)
    _ = criterion_eval(f_train, l_train)
    assert criterion_eval._current_bank_size == 4
    criterion_eval.eval()
    f_val = F.normalize(torch.randn(4, 64, device=DEVICE), dim=1)
    l_val = torch.tensor([0, 0, 1, 1], device=DEVICE)
    _ = criterion_eval(f_val, l_val)
    assert criterion_eval._current_bank_size == 4, (
        f"Bank should not grow during eval, got {criterion_eval._current_bank_size}"
    )
    print(f"✓ Eval mode: bank unchanged at {criterion_eval._current_bank_size}")

    # ── Test 8: Perfect clustering → lower loss ─────────────────────
    criterion_cluster = SupConLoss(temperature=0.1, memory_size=32, proj_dim=64).to(DEVICE)
    criterion_cluster.train()
    perfect_feats = torch.zeros(8, 64, device=DEVICE)
    perfect_feats[0:2, 0] = 1.0   # Class 0 cluster at dim 0
    perfect_feats[2:4, 1] = 1.0   # Class 1 cluster at dim 1
    perfect_feats[4:6, 2] = 1.0   # Class 2 cluster at dim 2
    perfect_feats[6:8, 3] = 1.0   # Class 3 cluster at dim 3
    perfect_feats = F.normalize(perfect_feats, dim=1)
    random_feats = F.normalize(torch.randn(8, 64, device=DEVICE), dim=1)
    loss_perfect = criterion_cluster(perfect_feats, lbls)
    # Reset bank to have fair comparison
    criterion_cluster2 = SupConLoss(temperature=0.1, memory_size=32, proj_dim=64).to(DEVICE)
    criterion_cluster2.train()
    loss_random = criterion_cluster2(random_feats, lbls)
    assert loss_perfect.item() < loss_random.item(), (
        f"Perfect clustering should have lower loss: "
        f"{loss_perfect.item():.4f} vs {loss_random.item():.4f}"
    )
    print(
        f"✓ Perfect clustering ({loss_perfect.item():.4f}) "
        f"< random ({loss_random.item():.4f})"
    )

    # ── Test 9: Batch size = 1 returns zero but updates bank ────────
    criterion_bs1 = SupConLoss(temperature=0.1, memory_size=16, proj_dim=64).to(DEVICE)
    criterion_bs1.train()
    single_feat = F.normalize(torch.randn(1, 64, device=DEVICE), dim=1)
    single_lbl = torch.tensor([0], device=DEVICE)
    loss_single = criterion_bs1(single_feat, single_lbl)
    assert loss_single.item() == 0.0, "Single sample should return 0 loss"
    assert criterion_bs1._current_bank_size == 1, "Bank should still be updated for bs=1"
    print(f"✓ Single-sample guard: loss = {loss_single.item():.4f}, bank = 1")

    # ── Test 10: No valid positives returns zero ─────────────────────
    criterion_nopos = SupConLoss(temperature=0.1, memory_size=16, proj_dim=64).to(DEVICE)
    criterion_nopos.train()
    unique_feats = F.normalize(torch.randn(4, 64, device=DEVICE), dim=1)
    unique_lbls = torch.tensor([0, 1, 2, 3], device=DEVICE)
    loss_unique = criterion_nopos(unique_feats, unique_lbls)
    assert loss_unique.item() == 0.0, "No positives should return 0 loss"
    print(f"✓ No-positives guard: loss = {loss_unique.item():.4f}")

    # ── Test 11: Temperature sensitivity ─────────────────────────────
    low_temp = SupConLoss(temperature=0.05, memory_size=16, proj_dim=64).to(DEVICE)
    high_temp = SupConLoss(temperature=0.5, memory_size=16, proj_dim=64).to(DEVICE)
    low_temp.train()
    high_temp.train()
    mixed_feats = F.normalize(torch.randn(8, 64, device=DEVICE), dim=1)
    loss_low_t = low_temp(mixed_feats, lbls)
    loss_high_t = high_temp(mixed_feats, lbls)
    print(
        f"✓ Temperature sensitivity: τ=0.05 → {loss_low_t.item():.4f}, "
        f"τ=0.5 → {loss_high_t.item():.4f}"
    )

    # ── Test 12: repr shows bank fill status ─────────────────────────
    print(f"✓ repr: {criterion!r}")

    # ── Test 13: B=2 with memory bank finds positives ────────────────
    criterion_b2 = SupConLoss(temperature=0.1, memory_size=32, proj_dim=64).to(DEVICE)
    criterion_b2.train()
    # Seed the bank with class 0 and class 1 samples
    for _ in range(4):
        seed_feats = F.normalize(torch.randn(4, 64, device=DEVICE), dim=1)
        seed_lbls = torch.tensor([0, 0, 1, 1], device=DEVICE)
        _ = criterion_b2(seed_feats, seed_lbls)
    # Now test B=2 — this is the real use case
    tiny_feats = F.normalize(torch.randn(2, 64, device=DEVICE), dim=1)
    tiny_lbls = torch.tensor([0, 1], device=DEVICE)
    loss_b2 = criterion_b2(tiny_feats, tiny_lbls)
    assert loss_b2.item() > 0, (
        "B=2 with seeded memory bank should find positives from the bank!"
    )
    print(
        f"✓ B=2 + memory bank: loss = {loss_b2.item():.4f} "
        f"(positives found from bank!)"
    )

    print(f"\n{'=' * 70}")
    print(f"  ✓ ALL SUPCON LOSS + XBM TESTS PASSED")
    print(f"{'=' * 70}")
