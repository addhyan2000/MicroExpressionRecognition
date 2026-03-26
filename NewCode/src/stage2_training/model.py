# -*- coding: utf-8 -*-
"""STSTNet — Shallow Triple Stream Three-dimensional CNN.

This module implements the STSTNet architecture adapted for 3D
spatio-temporal input tensors of shape ``(B, C=3, T=32, H=224, W=224)``.

Architecture Overview (per the Liong et al. 2019 paper):
    The network splits the 3-channel input into three independent streams,
    one per feature channel (Flow-X, Flow-Y, Optical Strain).  Each stream
    applies a single 3D convolutional layer followed by ReLU, Batch
    Normalisation, 3D Max Pooling, and Dropout.  The three branch outputs
    are concatenated along the channel dimension and fed through a global
    3D Average Pool and a single Fully Connected layer to produce class
    logits.

Branch Filter Counts (matching the reference implementation):
    - Branch 1 (Flow-X):         3 filters
    - Branch 2 (Flow-Y):         5 filters
    - Branch 3 (Optical Strain): 8 filters
    Total after concatenation:   16 feature maps

Reference:
    Liong et al., "Shallow Triple Stream Three-dimensional CNN (STSTNet)
    for Micro-expression Recognition", FG 2019.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Helper: Single 3D convolutional branch
# ──────────────────────────────────────────────────────────────────────

class _ConvBranch3D(nn.Module):
    """One branch of the triple-stream architecture.

    Pipeline: Conv3d → BatchNorm3d → ReLU → MaxPool3d → Dropout.

    Parameters
    ----------
    in_channels : int
        Number of input channels (always 1 — a single feature stream).
    out_channels : int
        Number of convolutional filters.
    kernel_size : int | tuple
        Spatial-temporal kernel size for Conv3d.
    dropout_p : float
        Dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        kernel_size: int = 3,
        dropout_p: float = 0.5,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # 'same'-style padding
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # MaxPool3d (kernel=3, stride=3, padding=1) performs an aggressive reduction.
        # Spatial dims: 224 -> floor((224 + 2 - 3)/3) + 1 = 75. Temporal: 32 -> 11.
        self.pool = nn.MaxPool3d(kernel_size=3, stride=3, padding=1)
        self.dropout = nn.Dropout3d(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the branch.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, 1, T, H, W)``.

        Returns
        -------
        Tensor
            Shape ``(B, out_channels, T', H', W')`` after pooling.
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


# ──────────────────────────────────────────────────────────────────────
# Main Model
# ──────────────────────────────────────────────────────────────────────

class STSTNet(nn.Module):
    """Shallow Triple Stream Three-dimensional CNN for MER.

    The network expects input of shape ``(B, 3, T, H, W)`` where the 3
    channels correspond to:
        - Channel 0: Horizontal optical flow  (u)
        - Channel 1: Vertical optical flow    (v)
        - Channel 2: Optical strain           (ε)

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 4: negative, positive,
        surprise, others).
    in_temporal : int
        Number of temporal frames in each sample (default 32).
    in_height : int
        Spatial height of each frame (default 224).
    in_width : int
        Spatial width of each frame (default 224).
    dropout_p : float
        Dropout probability used in each branch (default 0.5).
    """

    def __init__(
        self,
        num_classes: int = 4,
        in_temporal: int = 32,
        in_height: int = 224,
        in_width: int = 224,
        dropout_p: float = 0.5,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.in_temporal = in_temporal
        self.in_height = in_height
        self.in_width = in_width
        self._logged_latent_shape = False

        # ── Three parallel 3D convolutional branches ──────────────────
        # Filter counts mirror the original 2D STSTNet: 3, 5, 8
        self.branch1 = _ConvBranch3D(in_channels=1, out_channels=3, dropout_p=dropout_p)
        self.branch2 = _ConvBranch3D(in_channels=1, out_channels=5, dropout_p=dropout_p)
        self.branch3 = _ConvBranch3D(in_channels=1, out_channels=8, dropout_p=dropout_p)

        # ── Global Average Pool + Classifier ──────────────────────────
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # After concatenation: 3 + 5 + 8 = 16 channels
        self.fc = nn.Linear(in_features=16, out_features=num_classes)

        # ── Kaiming He initialisation (optimal for ReLU networks) ─────
        self._init_weights()



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the triple-stream network.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, 3, T, H, W)``.

        Returns
        -------
        Tensor
            Raw logits of shape ``(B, num_classes)``.
        """
        # Defensive Input Assertion: 3D-CNNs are highly sensitive to input shapes
        expected_shape = (x.shape[0], 3, self.in_temporal, self.in_height, self.in_width)
        if x.shape != expected_shape:
            raise ValueError(f"STSTNet expects input shape {expected_shape}, got {tuple(x.shape)}")

        # Split the 3 input channels into separate streams.
        # Each slice: (B, 1, T, H, W)
        x1 = x[:, 0:1, :, :, :]   # Flow-X
        x2 = x[:, 1:2, :, :, :]   # Flow-Y
        x3 = x[:, 2:3, :, :, :]   # Optical Strain

        # Pass through independent branches
        x1 = self.branch1(x1)      # (B, 3, T', H', W')
        x2 = self.branch2(x2)      # (B, 5, T', H', W')
        x3 = self.branch3(x3)      # (B, 8, T', H', W')

        # Concatenate along the channel dimension → (B, 16, T', H', W')
        x = torch.cat((x1, x2, x3), dim=1)

        # Global average pooling → (B, 16, 1, 1, 1)
        x = self.avgpool(x)

        if not self._logged_latent_shape:
            logger.debug("Latent Feature Dimension after Global Pool: %s", tuple(x.shape))
            self._logged_latent_shape = True

        # Flatten → (B, 16) and classify
        x = x.view(x.size(0), -1)
        logits = self.fc(x)        # (B, num_classes)

        return logits

    # ------------------------------------------------------------------ #
    #  Weight Initialisation
    # ------------------------------------------------------------------ #

    def _init_weights(self) -> None:
        """Apply Kaiming He initialisation to all Conv3d and Linear layers.

        This is the standard initialisation for networks using ReLU
        activations (He et al., 2015).  BatchNorm layers are initialised
        with weight=1, bias=0.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def reset_weights(self) -> None:
        """Re-initialise all learnable parameters.

        Useful when training in a leave-one-subject-out cross-validation
        loop to prevent weight leakage between folds.  Uses the same
        Kaiming initialisation strategy as ``_init_weights``.
        """
        self._init_weights()
