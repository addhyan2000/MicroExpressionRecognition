"""
adversarial_wrapper.py — Adversarial MER Model with SupCon Projection Head
============================================================================

Wraps the Stage 2 ``HybridMERModel`` in an adversarial framework that
splits the ``[B, 96]`` feature representation into three paths:

    ┌─────────────────────────────────────────────────────────────────┐
    │              AdversarialMERWrapper Pipeline                      │
    │                                                                 │
    │   [B, 3, 32, 224, 224]                                          │
    │         │                                                       │
    │         ▼                                                       │
    │   ┌──────────────────────────────────┐                          │
    │   │  HybridMERModel (Stage 2)        │                          │
    │   │  (Backbone→SimAM→Transformer)    │                          │
    │   │  → bypassed classifier           │                          │
    │   └──────────────┬───────────────────┘                          │
    │                  │ [B, 96]  (identity-entangled features)       │
    │                  │                                              │
    │         ┌────────┼────────┐                                     │
    │         │        │        │                                     │
    │         ▼        ▼        ▼                                     │
    │   ┌─────────┐ ┌──────┐ ┌──────────────┐                        │
    │   │Proj Head│ │Emo   │ │    GRL        │ ← gradient × (-λ)     │
    │   │(MLP +   │ │Head  │ │      ↓        │                        │
    │   │ L2Norm) │ │      │ │ Identity Head │                        │
    │   └────┬────┘ └──┬───┘ └──────┬───────┘                        │
    │        │         │            │                                  │
    │        ▼         ▼            ▼                                  │
    │   [B, proj_dim] [B, E]   [B, num_S]                             │
    │                                                                 │
    │   Projected     Emotion   Identity                              │
    │   Features      Logits    Logits                                │
    └─────────────────────────────────────────────────────────────────┘

Theory:
    The Projection Head maps features to a lower-dimensional space
    for Supervised Contrastive Loss (Khosla et al., 2020).  SupCon
    clusters same-emotion features and separates different emotions,
    anchoring the latent space against GRL-induced feature collapse.

    The Emotion Head optimizes for correct emotion classification.
    The Identity Head (behind the GRL) forces the Transformer to
    *unlearn* subject-specific features.

Author  : Addhyan
Stage   : 3 — Adversarial Domain Adaptation
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Wire up Stage 2 imports ─────────────────────────────────────────
_STAGE3_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _STAGE3_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "Stage2_Architecture"))
sys.path.insert(0, str(_PROJECT_ROOT / "Stage1_DataPipeline"))

from models.hybrid_model import HybridMERModel  # noqa: E402
from .grl import GradientReversalLayer           # noqa: E402

from utils.logger import get_logger              # noqa: E402


class AdversarialMERWrapper(nn.Module):
    """
    Adversarial wrapper with SupCon projection head for identity-invariant
    micro-expression recognition.

    This module:
        1. Instantiates the Stage 2 HybridMERModel.
        2. Bypasses and freezes its final classifier head.
        3. Routes the [B, 96] features to three separate paths:
           - **Projection Head**: MLP → L2-Norm → [B, proj_dim] for SupConLoss
           - **Emotion Head**: LayerNorm → Dropout → Linear(96, num_emotions)
           - **Identity Head**: GRL → LayerNorm → Dropout → Linear(96, num_subjects)

    Parameters
    ----------
    num_emotions : int
        Number of emotion classes (default: 4 for Negative/Positive/Surprise/Others).
    num_subjects : int
        Number of unique subject identities.
    grl_lambda : float
        Initial reversal coefficient for the GRL.
    feature_dim : int
        Dimension of the feature vector from the Transformer.
        Must match ``d_model`` (default: 96).
    proj_dim : int
        Projection head output dimension for contrastive learning.
        Default: 64 (following SimCLR/SupCon conventions).
    head_dropout : float
        Dropout rate for classification heads.
    backbone_kwargs : dict, optional
        Additional keyword arguments passed to ``HybridMERModel.__init__``.

    Example
    -------
    >>> model = AdversarialMERWrapper(num_emotions=4, num_subjects=26)
    >>> x = torch.randn(2, 3, 32, 224, 224)
    >>> projected, emotion_logits, identity_logits = model(x)
    >>> assert projected.shape == (2, 64)
    >>> assert emotion_logits.shape == (2, 4)
    >>> assert identity_logits.shape == (2, 26)
    """

    def __init__(
        self,
        num_emotions: int = 4,
        num_subjects: int = 26,
        grl_lambda: float = 1.0,
        feature_dim: int = 96,
        proj_dim: int = 64,
        head_dropout: float = 0.3,
        **backbone_kwargs,
    ) -> None:
        super().__init__()

        self._log = get_logger(
            "AdversarialMERWrapper",
            log_dir=_STAGE3_DIR / "logs",
            log_filename="stage3_training.log",
        )

        self.num_emotions = num_emotions
        self.num_subjects = num_subjects
        self.feature_dim = feature_dim
        self.proj_dim = proj_dim

        # ────────────────────────────────────────────────────────────
        # Component 1: Stage 2 Backbone (HybridMERModel)
        # ────────────────────────────────────────────────────────────
        # We instantiate HybridMERModel but will bypass its classifier.
        # The classifier inside HybridMERModel is NOT used — we replace
        # it with our triple-path architecture.
        self.backbone = HybridMERModel(
            num_classes=num_emotions,  # Not actually used (bypassed)
            **backbone_kwargs,
        )

        # ── CRITICAL: Freeze the unused Stage 2 classifier ──────────
        # This prevents unused-parameter errors in DDP / autograd,
        # since the classifier is never called in our forward pass.
        for param in self.backbone.classifier.parameters():
            param.requires_grad = False

        self._log.info(
            "Instantiated HybridMERModel backbone "
            "(classifier bypassed & frozen)"
        )
        self._log.info("  Feature dimension (d_model): %d", feature_dim)
        self._log.info("  Projection dimension: %d", proj_dim)

        # ────────────────────────────────────────────────────────────
        # Component 2: Projection Head for SupCon
        # ────────────────────────────────────────────────────────────
        self.projection_head = self._build_projection_head(
            feature_dim, proj_dim
        )

        # ────────────────────────────────────────────────────────────
        # Components 3 & 4: Classification Heads
        # ────────────────────────────────────────────────────────────
        self.emotion_head, self.identity_head = (
            self._build_classification_heads(
                feature_dim, num_emotions, num_subjects, head_dropout
            )
        )

        # ────────────────────────────────────────────────────────────
        # Component 5: Gradient Reversal Layer
        # ────────────────────────────────────────────────────────────
        self.grl = GradientReversalLayer(initial_lambda=grl_lambda)
        self._log.info("GRL initialized with λ=%.4f", grl_lambda)

        # ── Weight initialization for all new heads ─────────────────
        self._init_all_heads()

    def _build_projection_head(
        self,
        feature_dim: int,
        proj_dim: int,
    ) -> nn.Sequential:
        """
        Build the MLP projection head for Supervised Contrastive Learning.

        Architecture:
            Linear(feature_dim, feature_dim) → ReLU → Linear(feature_dim, proj_dim)

        The output is NOT normalized here — normalization is applied
        in the forward pass after the MLP, so the projection parameters
        can be optimized without normalization in the computation graph
        of the MLP itself.

        Parameters
        ----------
        feature_dim : int
            Input dimension (d_model from Transformer, typically 96).
        proj_dim : int
            Output dimension for contrastive space (typically 64).

        Returns
        -------
        nn.Sequential
            The projection MLP (without L2 normalization).
        """
        head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, proj_dim),
        )

        self._log.info(
            "Projection Head: Linear(%d, %d) → ReLU → Linear(%d, %d) → L2Norm",
            feature_dim, feature_dim, feature_dim, proj_dim,
        )
        return head

    def _build_classification_heads(
        self,
        feature_dim: int,
        num_emotions: int,
        num_subjects: int,
        head_dropout: float,
    ) -> Tuple[nn.Sequential, nn.Sequential]:
        """
        Build the emotion and identity classification heads.

        Both heads share the same architecture:
            LayerNorm(feature_dim) → Dropout(p) → Linear(feature_dim, out_dim)

        The identity head receives GRL-reversed features in the
        forward pass (the GRL is applied externally, not inside this head).

        Parameters
        ----------
        feature_dim : int
            Input dimension from Transformer (d_model).
        num_emotions : int
            Number of emotion classes.
        num_subjects : int
            Number of subject identities.
        head_dropout : float
            Dropout rate for both heads.

        Returns
        -------
        tuple of (nn.Sequential, nn.Sequential)
            ``(emotion_head, identity_head)``
        """
        emotion_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(head_dropout),
            nn.Linear(feature_dim, num_emotions),
        )

        identity_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(head_dropout),
            nn.Linear(feature_dim, num_subjects),
        )

        self._log.info(
            "Emotion Head: LayerNorm(%d) → Dropout(%.2f) → Linear(%d, %d)",
            feature_dim, head_dropout, feature_dim, num_emotions,
        )
        self._log.info(
            "Identity Head: GRL → LayerNorm(%d) → Dropout(%.2f) → Linear(%d, %d)",
            feature_dim, head_dropout, feature_dim, num_subjects,
        )

        return emotion_head, identity_head

    def _init_all_heads(self) -> None:
        """
        Initialize weights for the projection, emotion, and identity heads.

        Strategy:
            - Linear: Xavier uniform (consistent with Stage 2)
            - LayerNorm: weight=1, bias=0
        """
        for head in [self.projection_head, self.emotion_head, self.identity_head]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        self._log.info(
            "All head weights initialized "
            "(Xavier uniform + LayerNorm defaults)"
        )

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the HybridMERModel backbone up to the Transformer output,
        bypassing the final classifier.

        This replicates the forward pass of HybridMERModel but stops
        before the ``self.backbone.classifier`` call, returning the
        raw ``[B, 96]`` feature vector.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``[B, 3, 32, 224, 224]``.

        Returns
        -------
        torch.Tensor
            Feature tensor of shape ``[B, 96]``.
        """
        # Step 1: Backbone (STSTNet → SimAM → Concat)
        features = self.backbone.backbone(x)           # [B, 96, 32, 112, 112]

        # Step 2: Adaptive Spatial Pooling
        features = self.backbone.spatial_pool(features) # [B, 96, 32, 1, 1]

        # Step 3: Reshape to sequence format
        features = features.squeeze(-1).squeeze(-1)     # [B, 96, 32]
        features = features.permute(0, 2, 1)            # [B, 32, 96]

        # Step 4: Transformer Encoder (without classifier)
        temporal_repr = self.backbone.transformer(features)  # [B, 96]

        return temporal_repr

    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full adversarial forward pass with contrastive projection.

        Shape Transformation:
            1. Input:             [B, 3, 32, 224, 224]
            2. Backbone→Trans:    [B, 96]
            3a. Projection Head:  [B, 96] → MLP → L2Norm → [B, proj_dim]
            3b. Emotion Head:     [B, 96] → [B, num_emotions]
            3c. GRL→ID Head:      [B, 96] → GRL → [B, num_subjects]

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``[B, 3, 32, 224, 224]``.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor, torch.Tensor)
            ``(projected_features, emotion_logits, identity_logits)`` where:
            - ``projected_features`` has shape ``[B, proj_dim]`` (L2-normalized)
            - ``emotion_logits``     has shape ``[B, num_emotions]``
            - ``identity_logits``    has shape ``[B, num_subjects]``
        """
        # ── Extract features (bypass classifier) ────────────────────
        features = self._extract_features(x)  # [B, 96]

        # ── Projection Head (for SupCon) ────────────────────────────
        # MLP projection followed by L2 normalization.
        # SupCon requires normalized embeddings (cosine similarity).
        projected = self.projection_head(features)       # [B, proj_dim]
        projected = F.normalize(projected, dim=1, p=2)   # L2 normalize

        # ── Emotion Head (direct path) ──────────────────────────────
        emotion_logits = self.emotion_head(features)  # [B, num_emotions]

        # ── Identity Head (through GRL) ─────────────────────────────
        reversed_features = self.grl(features)         # [B, 96] (identity in fwd)
        identity_logits = self.identity_head(reversed_features)  # [B, num_subjects]

        return projected, emotion_logits, identity_logits

    def set_grl_lambda(self, new_lambda: float) -> None:
        """Convenience method to update the GRL's reversal coefficient."""
        self.grl.set_lambda(new_lambda)

    def schedule_grl_lambda(self, progress: float, gamma: float = 10.0) -> float:
        """
        Anneal the GRL lambda using the sigmoid schedule.

        Parameters
        ----------
        progress : float
            Training progress in [0, 1].
        gamma : float
            Sigmoid steepness.

        Returns
        -------
        float
            The new lambda value.
        """
        return self.grl.schedule_lambda(progress, gamma)

    def count_parameters(self) -> Dict[str, int]:
        """
        Count parameters by component for thesis documentation.

        Returns
        -------
        dict
            Parameter counts per component and total.
        """
        def _count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        def _count_trainable(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        breakdown = {
            "backbone_total": _count(self.backbone),
            "backbone_trainable": _count_trainable(self.backbone),
            "projection_head_total": _count(self.projection_head),
            "projection_head_trainable": _count_trainable(self.projection_head),
            "emotion_head_total": _count(self.emotion_head),
            "emotion_head_trainable": _count_trainable(self.emotion_head),
            "identity_head_total": _count(self.identity_head),
            "identity_head_trainable": _count_trainable(self.identity_head),
            "model_total": _count(self),
            "model_trainable": _count_trainable(self),
        }
        return breakdown


# ─────────────────────────────────────────────────────────────────────
# Standalone Verification
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  AdversarialMERWrapper — Standalone Verification (SupCon)")
    print("=" * 70)

    NUM_EMOTIONS = 4
    NUM_SUBJECTS = 26
    PROJ_DIM = 64
    BATCH_SIZE = 2

    model = AdversarialMERWrapper(
        num_emotions=NUM_EMOTIONS,
        num_subjects=NUM_SUBJECTS,
        grl_lambda=1.0,
        feature_dim=96,
        proj_dim=PROJ_DIM,
        head_dropout=0.3,
    )

    print(f"\nModel instantiated.")

    # ── Parameter count ─────────────────────────────────────────────
    params = model.count_parameters()
    print(f"\n{'─' * 50}")
    print(f"  Parameter Breakdown")
    print(f"{'─' * 50}")
    print(f"  Backbone (Stage 2):     {params['backbone_trainable']:>10,}")
    print(f"  Projection Head:        {params['projection_head_trainable']:>10,}")
    print(f"  Emotion Head:           {params['emotion_head_trainable']:>10,}")
    print(f"  Identity Head:          {params['identity_head_trainable']:>10,}")
    print(f"{'─' * 50}")
    print(f"  TOTAL Trainable:        {params['model_trainable']:>10,}")
    print(f"{'─' * 50}")

    # ── Forward pass ────────────────────────────────────────────────
    print(f"\nRunning forward pass...")
    x = torch.randn(BATCH_SIZE, 3, 32, 224, 224)

    model.eval()
    with torch.no_grad():
        projected, emotion_logits, identity_logits = model(x)

    print(f"\n  Input shape          : {list(x.shape)}")
    print(f"  Projected features   : {list(projected.shape)}")
    print(f"  Emotion logits       : {list(emotion_logits.shape)}")
    print(f"  Identity logits      : {list(identity_logits.shape)}")

    # ── Shape assertions ────────────────────────────────────────────
    assert projected.shape == (BATCH_SIZE, PROJ_DIM), (
        f"Projected shape mismatch: {projected.shape}"
    )
    assert emotion_logits.shape == (BATCH_SIZE, NUM_EMOTIONS), (
        f"Emotion shape mismatch: {emotion_logits.shape}"
    )
    assert identity_logits.shape == (BATCH_SIZE, NUM_SUBJECTS), (
        f"Identity shape mismatch: {identity_logits.shape}"
    )

    # ── Verify L2 normalization ─────────────────────────────────────
    norms = torch.norm(projected, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
        f"Projected features should be L2-normalized, norms: {norms}"
    )
    print(f"  ✓ Projected features are L2-normalized (norms ≈ 1.0)")

    # ── Verify Stage 2 classifier is frozen ─────────────────────────
    for p in model.backbone.classifier.parameters():
        assert not p.requires_grad, (
            "Stage 2 classifier parameters should be frozen!"
        )
    print(f"  ✓ Stage 2 classifier is frozen (requires_grad=False)")

    # ── GRL lambda scheduling ──────────────────────────────────────
    print(f"\nGRL Lambda Schedule:")
    for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
        lam = model.schedule_grl_lambda(p)
        print(f"  progress={p:.2f}  →  λ={lam:.4f}")

    print(f"\n{'=' * 70}")
    print(f"  ✓ ADVERSARIAL WRAPPER VERIFICATION PASSED (SupCon)")
    print(f"{'=' * 70}")
