"""Transfer Learning Loss Functions for MTMNet.

Implements Loss Inequality Regularization (LIR) with teacher gradient isolation,
Cross-Domain Shared Center Alignment, and vectorised Semi-Hard Triplet Mining to 
transfer knowledge robustly from the Macro to the Micro domain.

Author  : Addhyan Pant
Project : Master's Thesis — Micro-Expression Recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

# Assuming joint_loss.py is dynamically importable in this environment
from src.losses.joint_loss import JointLoss

__all__ = ["SemiHardTripletLoss", "MTMTransferLoss"]


class SemiHardTripletLoss(nn.Module):
    """Vectorized Triplet Loss with Dynamic Semi-Hard Negative Mining.
    
    Uses Cosine Distance (1 - CosineSimilarity) to prioritize the Transformer's 
    directional embeddings. Dynamically pairs each macro/micro batch to mine 
    Semi-Hard Negatives where: d(A, P) < d(A, N) < d(A, P) + margin.
    
    If no semi-hard negative is found, falls back to the hardest valid negative.
    """

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        micro_features: torch.Tensor,
        micro_labels: torch.Tensor,
        macro_features: torch.Tensor,
        macro_labels: torch.Tensor
    ) -> torch.Tensor:
        """Vectorized computation of the cosine semi-hard triplet margin loss."""
        # 1. Cosine Distance Calculation: 1 - CosineSimilarity
        micro_norm = F.normalize(micro_features, p=2, dim=1)
        macro_norm = F.normalize(macro_features, p=2, dim=1)
        
        # Dot product of normalized vectors gives cosine similarity
        cos_sim = torch.mm(micro_norm, macro_norm.t())
        
        # Convert similarity bounded [-1, 1] to distance bounded [0, 2]
        dists = 1.0 - cos_sim
        
        # Logical mask matrices for matching and differing labels
        labels_eq = (micro_labels.unsqueeze(1) == macro_labels.unsqueeze(0))
        labels_neq = ~labels_eq

        # FP16/AMP Compatibility: Cast safe_max explicitly to dists footprint
        # utilizing torch.where to avoid implicit scalar masked_fill castings
        safe_max = torch.tensor(1e4, dtype=dists.dtype, device=dists.device)
        invalid_pos = torch.tensor(-1.0, dtype=dists.dtype, device=dists.device)

        # 2. Hardest Positives (maximize distance to same-class macro)
        # Use invalid_pos (-1.0) for invalids so max() gracefully skips them
        pos_dists = torch.where(labels_eq, dists, invalid_pos)
        hardest_pos, _ = pos_dists.max(dim=1)  # Shape: (B_micro,)
        
        # Valid anchors must have at least one valid positive candidate
        valid_pos_mask = (hardest_pos >= 0.0)

        # 3. Semi-Hard Negatives: d(A, P) < d(A, N) < d(A, P) + margin
        d_ap = hardest_pos.unsqueeze(1)  # Broadcast bounds: (B_micro, 1)
        semi_hard_mask = labels_neq & (dists > d_ap) & (dists < d_ap + self.margin)
        
        # Pick the closest semi-hard negative. 
        neg_dists_semi = torch.where(semi_hard_mask, dists, safe_max)
        semi_hard_neg, _ = neg_dists_semi.min(dim=1)  # Shape: (B_micro,)
        
        # 4. Fallback Hardest Negatives (minimize distance to diff-class macro)
        neg_dists_all = torch.where(labels_neq, dists, safe_max)
        hardest_neg, _ = neg_dists_all.min(dim=1)  # Shape: (B_micro,)
        
        # Select Semi-Hard if available (< safe_max); otherwise, drop to Hardest Negative
        has_semi_hard = (semi_hard_neg < safe_max)
        selected_neg = torch.where(has_semi_hard, semi_hard_neg, hardest_neg)
        
        # 5. Filter Valid Triplets
        # Must have a positive candidate AND a negative candidate
        has_any_neg = (hardest_neg < safe_max)
        valid_triplets = valid_pos_mask & has_any_neg
        
        # Graph Stability: Return zero-loss tied to graph if no triplets found
        if not valid_triplets.any():
            return 0.0 * micro_features.sum()
            
        final_d_ap = hardest_pos[valid_triplets]
        final_d_an = selected_neg[valid_triplets]
        
        # Compute ReLU Margin loss
        losses = F.relu(final_d_ap - final_d_an + self.margin)
        return losses.mean()


class MTMTransferLoss(nn.Module):
    """Unified Transfer Loss Wrapper for Step 4.
    
    Combines:
    1. Cross-Domain Joint (Center + Focal) Loss.
    2. Loss Inequality Regularization (LIR) w/ Gradient Isolation on Macro.
    3. Semi-Hard Triplet Loss.
    4. Domain Adversarial BCE Loss.
    """

    def __init__(
        self,
        num_classes: int,
        feat_dim: int = 128,
        triplet_margin: float = 1.0,
        joint_lambda_c: float = 0.1,
        lir_alpha: float = 0.5,
        triplet_beta: float = 1.0,
        adv_gamma: float = 1.0,
        alpha: Optional[torch.Tensor] = None
    ) -> None:
        super().__init__()
        # Shared Centroids: A single JointLoss handles CenterLoss for BOTH domains.
        # Gradients from Macro and Micro align toward these same learnable centers.
        self.joint_loss = JointLoss(
            num_classes=num_classes, 
            feat_dim=feat_dim, 
            alpha=alpha,
            lambda_c=joint_lambda_c
        )
        
        self.triplet_loss = SemiHardTripletLoss(margin=triplet_margin)
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Hyperparameter weights
        self.lir_alpha = lir_alpha
        self.triplet_beta = triplet_beta
        self.adv_gamma = adv_gamma

    def forward(
        self,
        micro_logits: torch.Tensor,
        micro_features_raw: torch.Tensor,
        micro_features_proj: torch.Tensor,
        micro_labels: torch.Tensor,
        micro_domain_logits: torch.Tensor,
        macro_logits: torch.Tensor,
        macro_features_raw: torch.Tensor,
        macro_features_proj: torch.Tensor,
        macro_labels: torch.Tensor,
        macro_domain_logits: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        
        # 1. Micro-Expression Classification (Student)
        L_micro = self.joint_loss(micro_logits, micro_features_proj, micro_labels)
        
        # 2. Macro-Expression Classification (Teacher)
        L_macro_train = self.joint_loss(macro_logits, macro_features_proj, macro_labels)
        L_macro_detached = L_macro_train.detach()
            
        # 3. Loss Inequality Regularization (LIR)
        # Added eps for numerical stability resolving infinitely small float gaps in FP16
        eps = torch.tensor(1e-6, dtype=L_micro.dtype, device=L_micro.device)
        L_LIR = F.relu(L_micro - L_macro_detached + eps)
        
        # 4. Cross-Domain Center Alignment (Combined classification)
        L_cls_total = L_micro + L_macro_train
        
        # 5. Semi-Hard Triplet Loss
        L_triplet = self.triplet_loss(micro_features_raw, micro_labels, macro_features_raw, macro_labels)
        
        # 6. Domain Adversarial Loss
        # Explicit tensor creation avoiding `ones_like` implicity for strict AMP alignment
        domain_targets_micro = torch.ones(
            micro_domain_logits.size(), 
            dtype=micro_domain_logits.dtype, 
            device=micro_domain_logits.device
        )
        domain_targets_macro = torch.zeros(
            macro_domain_logits.size(), 
            dtype=macro_domain_logits.dtype, 
            device=macro_domain_logits.device
        )
        
        domain_logits = torch.cat([micro_domain_logits, macro_domain_logits], dim=0)
        domain_targets = torch.cat([domain_targets_micro, domain_targets_macro], dim=0)
        L_adv = self.bce_loss(domain_logits, domain_targets)
        
        # Total Weighted Sum
        L_total = (
            L_cls_total 
            + self.lir_alpha * L_LIR 
            + self.triplet_beta * L_triplet 
            + self.adv_gamma * L_adv
        )
        
        return {
            "Total_Loss": L_total,
            "L_micro": L_micro.detach(),
            "L_macro": L_macro_detached,
            "L_LIR": L_LIR.detach(),
            "L_triplet": L_triplet.detach(),
            "L_adv": L_adv.detach()
        }
