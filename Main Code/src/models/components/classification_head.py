"""Classification Head for Micro-Expression Recognition.

Implements a lightweight Multilayer Perceptron (MLP) to project the
128-dimensional sequence embedding from the temporal aggregator into
class logits.

Author  : Addhyan Pant
Project : Master's Thesis — Micro-Expression Recognition
"""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Lightweight MLP for projecting sequence embeddings to class logits.

    Uses strong dropout and LayerNorm to prevent overfitting on small
    micro-expression datasets.  LayerNorm is used instead of BatchNorm1d
    to avoid crashes when the DataLoader yields a batch size of 1 (which
    happens at the tail end of tiny datasets like CAS(ME)²'s 53 samples).
    It also aligns with the upstream Transformer's normalization scheme.

    Weight initialization is layer-aware:
    - The hidden Linear layer uses Kaiming Normal (fan_out, relu) because
      it feeds into a ReLU activation.
    - The final logit layer uses a small Normal(0, 0.01) to ensure
      near-uniform initial class probabilities — preventing gradient
      explosions in the downstream Focal Loss / Softmax.

    Parameters
    ----------
    input_dim : int, optional
        Dimensionality of the input sequence embedding (default: 128).
    hidden_dim : int, optional
        Dimensionality of the hidden layer (default: 64).
    num_classes : int, optional
        Number of output classes. CASME II uses 5, CAS(ME)^2 uses 3 (default: 5).
    dropout : float, optional
        Dropout probability applied after the ReLU activation (default: 0.5).

    Shape
    -----
    - Input: Tensor of shape ``(Batch, input_dim)``
    - Output: Tensor of shape ``(Batch, num_classes)`` containing raw unactivated logits.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        num_classes: int = 5,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Sequential layout:
        #   [0] nn.Linear(input_dim, hidden_dim)   — hidden layer
        #   [1] nn.LayerNorm(hidden_dim)
        #   [2] nn.ReLU(inplace=True)
        #   [3] nn.Dropout(p=dropout)
        #   [4] nn.Linear(hidden_dim, num_classes)  — logit layer
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Apply layer-aware weight initialization
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply layer-aware initialization.

        - Hidden Linear [0]: Kaiming Normal (fan_out, relu) — variance-
          preserving through the ReLU.
        - LayerNorm [1]: weight = 1.0, bias = 0.0.
        - Logit Linear [4]: Normal(0, 0.01) — keeps initial logits
          tightly clustered near zero so that softmax outputs ≈ 1/C,
          preventing gradient explosions in epoch 1.
        """
        # Hidden layer — feeds into ReLU, so Kaiming is correct
        nn.init.kaiming_normal_(
            self.classifier[0].weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.zeros_(self.classifier[0].bias)

        # LayerNorm
        nn.init.ones_(self.classifier[1].weight)
        nn.init.zeros_(self.classifier[1].bias)

        # Final logit layer — feeds into softmax/cross-entropy, NOT ReLU.
        # Small std keeps initial predictions near-uniform across classes.
        nn.init.normal_(self.classifier[4].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier[4].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute class logits.

        Parameters
        ----------
        x : torch.Tensor
            Sequence embeddings from the temporal aggregator. Shape: ``(Batch, input_dim)``.

        Returns
        -------
        torch.Tensor
            Raw class logits. Shape: ``(Batch, num_classes)``.
        """
        # Ensure we have a valid 2D tensor of (Batch, input_dim)
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(
                f"Expected input of shape (Batch, {self.input_dim}), but got {x.shape}"
            )

        return self.classifier(x)
