"""Spatial Feature Encoder with SimAM Attention for Micro-Expression Recognition.

This module implements the first stage of a hybrid CNN-Transformer pipeline:
a triple-stream convolutional spatial encoder enhanced with the parameter-free
SimAM attention mechanism. It extracts patch-level spatial features from
optical-flow video sequences while preserving the temporal dimension for
downstream Spatio-Temporal Transformer processing (SLSTT, Step 2).

Architecture Overview
---------------------
  Input  (B, T, 3, 28, 28)      # Optical flow cube: Horizontal, Vertical, Strain
    |
    v
  Reshape to (B*T, 3, 28, 28)   # Merge batch and time for spatial processing
    |
    +-- Stream 1: Conv2d(3→3)  -> ReLU -> SimAM -> MaxPool  -> (B*T,  3, 10, 10)
    +-- Stream 2: Conv2d(3→5)  -> ReLU -> SimAM -> MaxPool  -> (B*T,  5, 10, 10)
    +-- Stream 3: Conv2d(3→8)  -> ReLU -> SimAM -> MaxPool  -> (B*T,  8, 10, 10)
    |
    v
  Concatenate along channels     -> (B*T, 16, 10, 10)
    |
    v
  Reshape to (B, T, 100, 16)    # 100 spatial patches × 16-dim embedding
    |
    v
  Output: ready for Transformer

Reference Papers
-----------------
  - STSTNet: Liong et al., "Shallow Triple Stream Three-dimensional CNN
    for Micro-expression Recognition", FG 2019.
  - SimAM: Yang et al., "SimAM: A Simple, Parameter-Free Attention Module
    for Convolutional Neural Networks", ICML 2021.

Author  : Addhyan Pandey
Project : Master's Thesis — Micro-Expression Recognition
"""

import torch
import torch.nn as nn

__all__ = ["SimAM", "SpatialEncoder"]


# ---------------------------------------------------------------------------
# SimAM: Parameter-Free Spatial Attention
# ---------------------------------------------------------------------------

class SimAM(nn.Module):
    """SimAM (Simple, Parameter-Free Attention Module).

    Computes 3-D (channel × spatial) attention weights using an analytical
    energy function inspired by neuroscientific spatial suppression, without
    introducing any learnable parameters.

    The closed-form solution for a single neuron's importance score is:

        E_inv(x) = (x - μ)² / (4 · (σ² + ε)) + 0.5

    where μ and σ² are the spatial mean and variance of the feature map.
    The output is the element-wise product of the input and sigmoid(E_inv).

    Parameters
    ----------
    e_lambda : float, optional
        Small constant for numerical stability (default: 1e-4).

    Shape
    -----
    - Input:  (N, C, H, W)
    - Output: (N, C, H, W)  — same shape, re-weighted by attention.

    Notes
    -----
    This module has **zero** learnable parameters, making it ideal for
    extremely small datasets such as CASME II (~247 samples) where
    parameter-heavy attention (SE, CBAM) would cause overfitting.
    """

    def __init__(self, e_lambda: float = 1e-4) -> None:
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SimAM attention to the input feature map.

        Parameters
        ----------
        x : torch.Tensor
            Feature tensor of shape ``(N, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Attention-weighted tensor of shape ``(N, C, H, W)``.

        Tensor Flow
        -----------
        (N, C, H, W)  →  compute E_inv  →  sigmoid  →  element-wise mul
        """
        # Spatial size for unbiased variance (exclude the neuron itself)
        # x.shape = (N, C, H, W)
        _, _, h, w = x.shape
        n = h * w - 1  # scalar

        # Squared deviation from the spatial mean
        # mean over H, W → keepdim for broadcasting
        # d.shape = (N, C, H, W)
        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)

        # Spatial variance (unbiased estimate)
        # v.shape = (N, C, 1, 1)
        v = d.sum(dim=[2, 3], keepdim=True) / n

        # Inverse energy: higher value ⇒ neuron stands out more
        # E_inv.shape = (N, C, H, W)
        e_inv = d / (4.0 * (v + self.e_lambda)) + 0.5

        # Sigmoid gating — re-weight the original features
        return x * torch.sigmoid(e_inv)

    def extra_repr(self) -> str:
        return f"e_lambda={self.e_lambda}"


# ---------------------------------------------------------------------------
# SpatialEncoder: Triple-Stream CNN with SimAM
# ---------------------------------------------------------------------------

class SpatialEncoder(nn.Module):
    """Triple-stream spatial feature encoder based on STSTNet with SimAM.

    Processes each frame of an optical-flow video through three parallel
    shallow Conv2d streams (3, 5, and 8 filters), each followed by ReLU
    activation, SimAM attention, and max-pooling.  The three streams are
    concatenated along the channel axis to produce a 16-channel feature
    map, which is then unfolded into a sequence of spatial patch embeddings
    suitable for a downstream Transformer.

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels per frame (default: 3 for optical-flow
        triplet: horizontal, vertical, strain).
    e_lambda : float, optional
        SimAM stability constant forwarded to each SimAM block (default: 1e-4).

    Shape
    -----
    - Input:  ``(B, T, 3, 28, 28)``
              B = batch size, T = number of temporal frames.
    - Output: ``(B, T, 100, 16)``
              100 = 10×10 spatial patches, 16 = embedding dimension.

    Design Rationale
    ----------------
    * **Shallow streams (3 / 5 / 8 filters):**  Proven in the STSTNet
      baseline to prevent overfitting on micro-expression datasets that
      contain fewer than 300 training samples.
    * **SimAM after ReLU:**  Enhances subtle, localised facial Action
      Units (e.g., AU4 — brow lowerer) without adding learnable params.
    * **Patch-sequence output:**  Unlike STSTNet's terminal average
      pooling that collapses spatial information, we preserve every
      10×10 spatial patch so the subsequent Spatio-Temporal Transformer
      can attend to fine-grained muscle deformations across time.
    """

    def __init__(
        self,
        in_channels: int = 3,
        e_lambda: float = 1e-4,
    ) -> None:
        super().__init__()

        # ---- Stream 1: 3 filters ----
        self.stream1 = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SimAM(e_lambda=e_lambda),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
        )

        # ---- Stream 2: 5 filters ----
        self.stream2 = nn.Sequential(
            nn.Conv2d(in_channels, 5, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SimAM(e_lambda=e_lambda),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
        )

        # ---- Stream 3: 8 filters ----
        self.stream3 = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SimAM(e_lambda=e_lambda),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
        )

        # Total output channels after concatenation: 3 + 5 + 8 = 16
        self._out_channels = 16

    @property
    def out_channels(self) -> int:
        """Total number of output feature channels (3 + 5 + 8)."""
        return self._out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial patch embeddings from a video sequence.

        Parameters
        ----------
        x : torch.Tensor
            Optical-flow video tensor of shape ``(B, T, 3, 28, 28)``.

        Returns
        -------
        torch.Tensor
            Spatial patch embeddings of shape ``(B, T, 100, 16)``.

        Tensor Flow (step-by-step)
        --------------------------
        1. ``(B, T, 3, 28, 28)``  →  reshape to ``(B*T, 3, 28, 28)``
        2. Three parallel streams, each producing ``(B*T, C_i, 10, 10)``
           with C_i ∈ {3, 5, 8}.
        3. Concatenate → ``(B*T, 16, 10, 10)``
        4. Reshape     → ``(B, T, 16, 10, 10)``
        5. Permute     → ``(B, T, 10, 10, 16)``
        6. Flatten H×W → ``(B, T, 100, 16)``
        """
        b, t, c, h, w = x.shape

        # ---- 1. Merge batch & time for frame-level spatial processing ----
        # (B, T, 3, 28, 28) → (B*T, 3, 28, 28)
        x = x.reshape(b * t, c, h, w)

        # ---- 2. Triple-stream convolution ----
        # Each stream: Conv2d → ReLU → SimAM → MaxPool
        # Input  per stream: (B*T, 3, 28, 28)
        # Output per stream: (B*T, C_i, 10, 10)
        s1 = self.stream1(x)  # (B*T, 3, 10, 10)
        s2 = self.stream2(x)  # (B*T, 5, 10, 10)
        s3 = self.stream3(x)  # (B*T, 8, 10, 10)

        # ---- 3. Concatenate along channel axis ----
        # (B*T, 16, 10, 10)
        out = torch.cat([s1, s2, s3], dim=1)

        # ---- 4. Restore the temporal dimension ----
        # (B*T, 16, 10, 10) → (B, T, 16, 10, 10)
        _, c_out, h_out, w_out = out.shape
        out = out.reshape(b, t, c_out, h_out, w_out)

        # ---- 5–6. Rearrange to patch-sequence format ----
        # (B, T, 16, 10, 10) → (B, T, 10, 10, 16) → (B, T, 100, 16)
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        out = out.reshape(b, t, h_out * w_out, c_out)

        return out
