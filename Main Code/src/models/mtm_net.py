"""MTMNet Wrapper for Macro-to-Micro Transfer Learning.

Implements the EIDNet principles and domain adaptation logic for the MER thesis.
Integrates a Gradient Reversal Layer (GRL) and Domain Discriminator to align
Macro-expression domains with Micro-expression domains based on abstract 
affective motion features.

Identity Disentanglement is natively established by the prerequisite step 
which extracts Optical Flow (horizontal, vertical, optical strain), and is 
further explicitly enforced by the Identity Invariant Projector. By discarding
the raw RGB frames and normalizing feature magnitudes, static identity cues 
(bone structure, lighting, skin color) are fundamentally stripped from the pipeline.

Author  : Addhyan Pandey
Project : Master's Thesis — Micro-Expression Recognition
"""

import logging
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)

__all__ = ["GradientReversalLayer", "GRLModule", "IdentityInvariantProjector", "DomainDiscriminator", "MTMNet"]


class GradientReversalLayer(torch.autograd.Function):
    """Gradient Reversal Layer (GRL).
    
    Acts as an identity function during the forward pass.
    During the backward pass, multiplies the gradient by a negative scalar 
    (-lambda) to reverse the optimization direction, forcing the preceding 
    encoder to 'unlearn' domain-specific information.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features.
        lambda_ : float
            The scaling factor for gradient reversal.
            
        Returns
        -------
        torch.Tensor
            Unmodified input features.
        """
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Backward pass."""
        # Multiplies gradient by -lambda
        grad_input = grad_output.clone()
        return -ctx.lambda_ * grad_input, None


class GRLModule(nn.Module):
    """GRL Module Wrapper.
    
    Encapsulates the GradientReversalLayer autograd function and elegantly manages
    the global adversarial scaling weight (`alpha` and `current_lambda`) to avoid 
    stray parameters in every forward pass of the main network.
    
    Parameters
    ----------
    alpha : float
        A base multiplier for the scaled gradient penalty. Default is 1.0.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        # Formally register as a buffer so it persists inherently within state_dict
        self.register_buffer('current_lambda', torch.tensor(0.0))

    def set_lambda(self, val: float) -> None:
        """Dynamically update the GRL scaling factor during training epochs."""
        # Clamp lambda to prevent adversarial gradient explosion
        clamped_val = max(0.0, min(1.0, val))
        self.current_lambda.fill_(clamped_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input features.

        Returns
        -------
        torch.Tensor
            Unmodified features going forward, reversed gradients coming back.
        """
        # Apply the internally tracked lambda
        return GradientReversalLayer.apply(x, self.current_lambda.item() * self.alpha)


class IdentityInvariantProjector(nn.Module):
    """Identity Invariant Projector.
    
    A small MLP bottleneck that explicitly strips subject-specific intensity biases 
    that survive the optical flow extraction. Uses LayerNorm for native 1D feature 
    normalization, treating the feature vector as an instance sequence and stripping 
    its magnitude explicitly.
    
    Parameters
    ----------
    input_dim : int
        The dimension of the input sequence embedding (e.g., 128).
    hidden_dim : int
        The bottleneck dimension (e.g., 64).
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64) -> None:
        super().__init__()
        # Using native nn.LayerNorm efficiently normalizes identical to InstanceNorm1d(1)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),  # Safe here since inputs to linear branch sequentially
            nn.Linear(hidden_dim, input_dim)
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Embedding of shape `(B, input_dim)`.

        Returns
        -------
        torch.Tensor
            Projected identity-agnostic embedding of shape `(B, input_dim)`.
        """
        # Linear + LayerNorm prevents mutating `x` inplace, preserving raw_features
        return self.net(x)


class DomainDiscriminator(nn.Module):
    """Domain Discriminator for Adversarial Alignment.
    
    Classifies sequence embeddings as 'Macro' (0) or 'Micro' (1).
    Utilizes a lightweight architecture and strong dropout to prevent the discriminator 
    from overpowering the encoder on small datasets (e.g., 53 samples).
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 32) -> None:
        super().__init__()
        self.grl = GRLModule()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),  # Optimized inplace operation post-linear
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, 1)  # Binary logit output
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Output layer
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through GRL then Discriminator.

        Parameters
        ----------
        x : torch.Tensor
            Embedded features of shape `(B, input_dim)`.
            
        Returns
        -------
        torch.Tensor
            Raw logits for domain classification `(B, 1)`.
        """
        x_reversed = self.grl(x)
        return self.net(x_reversed)


class MTMNet(nn.Module):
    """Macro-to-Micro Transfer Learning Network (MTMNet).
    
    A unified wrapper serving as both MacroNet (Teacher) and MicroNet (Student)
    by sharing the exact same spatial-temporal backbone. Exposes logits for 
    emotion classification and domain classification. Allows Teacher-Student 
    weight cloning and freezing natively.
    
    Parameters
    ----------
    spatial_encoder : nn.Module
        The initialized Stage 1 Spatial Encoder (Conv3d/SimAM).
    temporal_aggregator : nn.Module
        The initialized Stage 2 Temporal Modeler (SLSTT-LSTM).
    classification_head : nn.Module
        The initialized Stage 3 Emotion Classifier.
    """

    def __init__(
        self,
        spatial_encoder: nn.Module,
        temporal_aggregator: nn.Module,
        classification_head: nn.Module,
    ) -> None:
        super().__init__()
        
        # Determine feature dimension explicitly to harden coupling
        if not hasattr(temporal_aggregator, "out_features"):
            raise AttributeError("`temporal_aggregator` must explicitly define `out_features`.")
        feature_dim = temporal_aggregator.out_features
        
        # Robust Feature Coupling Validation
        class_head_in_features = None
        # Deep inspection for custom non-linear sequential heads
        for module in classification_head.modules():
            if isinstance(module, nn.Linear):
                class_head_in_features = module.in_features
                break
            
        if class_head_in_features is not None:
            if feature_dim != class_head_in_features:
                raise ValueError(
                    f"Dimension Mismatch: Temporal Aggregator outputs {feature_dim}-D, "
                    f"but Classification Head strictly expects {class_head_in_features}-D."
                )
        else:
            logger.warning(
                "Could not explicitly detect `in_features` in classification_head. "
                "Bypassing strict dimension coupling check for custom head."
            )

        # Backbone (Shared across domains)
        self.spatial_encoder = spatial_encoder
        self.temporal_aggregator = temporal_aggregator
        
        # Explicit Identity Disentanglement Projector
        self.identity_projector = IdentityInvariantProjector(
            input_dim=feature_dim, hidden_dim=feature_dim // 2
        )
        
        # Emotion Classifier (Shared across domains)
        self.classification_head = classification_head
        
        # Lightweight Adversarial Domain Discriminator
        self.domain_discriminator = DomainDiscriminator(
            input_dim=feature_dim
        )

    def freeze_as_teacher(self) -> None:
        """Freeze all parameters to act as the exact frozen MacroNet Teacher."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def clone_from_teacher(self, teacher_model: "MTMNet") -> None:
        """Clone weights directly from the Teacher model to instantiate the Student.
        
        Parameters
        ----------
        teacher_model : MTMNet
            The explicitly trained MacroNet Teacher model.
        """
        # Strict state loading guarantees perfectly synchronized architecture states
        self.load_state_dict(teacher_model.state_dict(), strict=True)

    def update_grl_lambda(self, val: float) -> None:
        """Update the internal Gradient Reversal scaling factor linearly."""
        self.domain_discriminator.grl.set_lambda(val)

    def extract_features(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract both raw and identity-projected embeddings.
        
        Parameters
        ----------
        x : torch.Tensor
            Optical flow video tensor `(B, T, 3, H, W)`.
        lengths : torch.Tensor, optional
            Valid lengths for variable-length video sequence packing.
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - raw_features: Embedding before ID projection `(B, 128)`.
            - proj_features: Embedding after ID projection `(B, 128)`.
        """
        # Extract Spatial Patches -> (B, T, P, 16)
        spatial_features = self.spatial_encoder(x)
        # Aggregate Temporal Dynamics -> (B, 128)
        raw_features = self.temporal_aggregator(spatial_features, lengths=lengths)
        # Explicitly Disentangle Identity -> (B, 128)
        proj_features = self.identity_projector(raw_features)
        return raw_features, proj_features

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for MTMNet.

        Parameters
        ----------
        x : torch.Tensor
            Optical flow video tensor `(B, T, 3, H, W)`.
        lengths : torch.Tensor, optional
            Valid lengths for variable-length video sequence packing.
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - 'features_raw': Embedding before identity projection (for Triplet Loss).
            - 'features_proj': Embedding after identity projection (for distill/classification).
            - 'class_logits': Emotion classification logits `(B, num_classes)`.
            - 'domain_logits': Binary domain classification logits `(B, 1)`.
        """
        # 1. Feature Extraction & Identity Disentanglement
        raw_features, proj_features = self.extract_features(x, lengths)
        
        # 2. Emotion Classification (uses projected identity-agnostic features)
        class_logits = self.classification_head(proj_features)
        
        # 3. Domain Discrimination (Adversarial, uses projected features)
        domain_logits = self.domain_discriminator(proj_features)
        
        return {
            "features_raw": raw_features,
            "features_proj": proj_features,
            "class_logits": class_logits,
            "domain_logits": domain_logits
        }
