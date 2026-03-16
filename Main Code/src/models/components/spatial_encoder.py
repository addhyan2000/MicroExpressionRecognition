"""Spatial Feature Encoder with SimAM Attention for Micro-Expression Recognition.

This module implements the first stage of a hybrid CNN-Transformer pipeline:
a multi-stream 3D convolutional spatial encoder enhanced with the parameter-free
SimAM attention mechanism. It extracts short-term spatiotemporal features from
optical-flow video sequences while preserving the temporal dimension intact for
downstream Spatio-Temporal Transformer processing (SLSTT, Step 2).

Architecture Overview
---------------------
  Input  (B, T, 3, H, W)        # Optical flow cube: Horizontal, Vertical, Strain
    |
    v
  Permute to (B, 3, T, H, W)   # Channel-first 3D processing
    |
    +-- Stream 1: Conv3d(3→F1) -> IN3d -> ReLU -> SimAM3d -> AdaptivePool3d -> (B, F1, T, P_H, P_W)
    +-- Stream 2: Conv3d(3→F2) -> IN3d -> ReLU -> SimAM3d -> AdaptivePool3d -> (B, F2, T, P_H, P_W)
    ...
    +-- Stream N: Conv3d(3→FN) -> IN3d -> ReLU -> SimAM3d -> AdaptivePool3d -> (B, FN, T, P_H, P_W)
    |
    v
  Concatenate along channels     -> (B, sum(F), T, P_H, P_W)
    |
    v
  1x1x1 Projection (feature_proj)-> (B, embed_dim, T, P_H, P_W)
    |
    v
  Permute & Reshape              -> (B, T, P, embed_dim)  # P = P_H × P_W spatial patches
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
from torch.cuda.amp import autocast

__all__ = ["SimAM", "SpatialEncoder"]


# ---------------------------------------------------------------------------
# SimAM: Parameter-Free Spatial Attention
# ---------------------------------------------------------------------------

class SimAM(nn.Module):
    """SimAM (Simple, Parameter-Free Attention Module).

    Computes 4-D (channel × depth × height × width) attention weights using an 
    analytical energy function inspired by neuroscientific spatial suppression, 
    without introducing any learnable parameters.

    The closed-form solution for a single neuron's importance score is:

        E_inv(x) = (x - μ)² / (4 · (σ² + ε)) + 0.5

    where μ and σ² are the spatiotemporal mean and variance of the feature map.
    The output is the element-wise product of the input and sigmoid(E_inv).

    Parameters
    ----------
    e_lambda : float, optional
        Small constant for numerical stability (default: 1e-4).

    Shape
    -----
    - Input:  (N, C, D, H, W)
    - Output: (N, C, D, H, W)  — same shape, re-weighted by attention.

    Notes
    -----
    This module has **zero** learnable parameters, making it ideal for
    extremely small datasets such as CASME II (~247 samples) where
    parameter-heavy attention (SE, CBAM) would cause overfitting.
    """

    def __init__(self, e_lambda: float = 1e-4) -> None:
        super().__init__()
        self.e_lambda = e_lambda

    @autocast(enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SimAM attention to the input feature map.

        Calculated with FP32 precision to prevent potential NaN overflows
        during Automatic Mixed Precision (AMP) training. Disables autocast
        defensively to maintain scaling context.

        Parameters
        ----------
        x : torch.Tensor
            Feature tensor of shape ``(N, C, D, H, W)``.

        Returns
        -------
        torch.Tensor
            Attention-weighted tensor of shape ``(N, C, D, H, W)``.
        """
        orig_dtype = x.dtype
        x_f32 = x.float()

        _, _, d, h, w = x_f32.shape
        n = max(d * h * w - 1, 1)  # Defensively clamp to prevent division by zero

        # Calculate deviation from the spatiotemporal mean
        dev = (x_f32 - x_f32.mean(dim=[2, 3, 4], keepdim=True)).pow(2)

        # Calculate spatial variance directly from deviation to save VRAM
        v = dev.sum(dim=[2, 3, 4], keepdim=True) / n

        # Inverse energy: higher value ⇒ neuron stands out more
        e_inv = dev / (4.0 * (v + self.e_lambda)) + 0.5

        # Sigmoid gating — re-weight the original features (out-of-place)
        out = x_f32 * torch.sigmoid(e_inv)
        return out.to(orig_dtype)

    def extra_repr(self) -> str:
        return f"e_lambda={self.e_lambda}"


# ---------------------------------------------------------------------------
# SpatialEncoder: Multi-Stream CNN with SimAM
# ---------------------------------------------------------------------------

class SpatialEncoder(nn.Module):
    """Multi-stream 3D spatial feature encoder based on STSTNet with SimAM.

    Processes each framewise sequence of an optical-flow video through parallel
    shallow Conv3d streams, each followed by Instance Normalization, ReLU 
    activation, 3D SimAM attention, and sequence-preserving adaptive max-pooling. 
    The streams are concatenated along the channel axis to produce a fused feature 
    map, projected to a target embedding dimension via 1x1x1 Conv3d, and unfolded 
    into a sequence of spatial patch embeddings before feeding into a downstream 
    Transformer.

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels per frame (default: 3 for optical-flow
        triplet: horizontal, vertical, strain).
    e_lambda : float, optional
        SimAM stability constant forwarded to each SimAM block (default: 1e-4).
    stream_configs : tuple, optional
        Configurations for the parallel 3D streams in the format 
        ``(out_channels, (kernel_T, kernel_H, kernel_W))``.
        (default: ((3, (3, 3, 3)), (5, (3, 3, 3)), (8, (1, 3, 3)))).
    patch_grid : tuple, optional
        Spatial structural dimensions (H, W) for the adaptive patch grid 
        (default: (10, 10)).
    embed_dim : int, optional
        Projection dimension for the downstream Spatio-Temporal Transformer
        (default: 16).

    Shape
    -----
    - Input:  ``(B, T, 3, H_in, W_in)``
              B = batch size, T = number of temporal frames.
    - Output: ``(B, T, P, embed_dim)``
              Produces exactly P patches regardless of input size, 
              where P = patch_grid[0] × patch_grid[1].

    Design Rationale
    ----------------
    * **Shallow 3D streams:** Restores STSTNet's 3D convolutions to process 
      short-term temporal structure dynamically instead of independently processing 
      collapsed 2D images.
    * **Instance Normalization:** Replaces BatchNorm to evaluate spatial statistics
      of each frame sequence independently, preventing "temporal leakage" and 
      perfectly preserving the intensity gradient from neutral onset to peak apex.
    * **3D SimAM after ReLU:** Enhances subtle, localised facial Action Units and
      dynamics simultaneously.
    * **Adaptive 3D Max Pooling:** Configured to perfectly preserve sequence length
      `T` while forcing spatial constraints to guarantee exact patch layouts,
      making the network resolution-agnostic.
    * **1x1x1 Feature Projection:** Decouples the CNN filter capacity from the
      Transformer's d_model dimension, ensuring safe hyperparameter scalability.
    """

    def __init__(
        self,
        in_channels: int = 3,
        e_lambda: float = 1e-4,
        stream_configs: tuple = ((3, (3, 3, 3)), (5, (3, 3, 3)), (8, (1, 3, 3))),
        patch_grid: tuple = (10, 10),
        embed_dim: int = 16,
    ) -> None:
        super().__init__()
        
        self.streams = nn.ModuleList()
        for out_c, kernel in stream_configs:
            # Padding handles temporal kernel and spatial kernel 
            # Assumes spatial kernel is always 3, padding=1 for spatial
            pad_t = kernel[0] // 2
            self.streams.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_c, kernel_size=kernel, padding=(pad_t, 1, 1), bias=False),
                    nn.InstanceNorm3d(out_c, affine=True),
                    nn.ReLU(inplace=True),
                    SimAM(e_lambda=e_lambda),
                    nn.AdaptiveMaxPool3d((None, patch_grid[0], patch_grid[1]))
                )
            )

        self._out_channels = embed_dim
        sum_filters = sum(config[0] for config in stream_configs)

        # 1x1x1 Projection layer to decouple Convolution capacity from Transformer embed_dim
        self.feature_proj = nn.Conv3d(sum_filters, embed_dim, kernel_size=1, bias=False)

        # Apply robust weight initialization
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize module weights to ensure proper gradient convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm3d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    @property
    def out_channels(self) -> int:
        """Total number of output feature channels."""
        return self._out_channels

    def freeze_encoder(self) -> None:
        """Freeze the spatial encoder parameters for transfer learning."""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze the spatial encoder parameters."""
        self.train()
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial patch embeddings from a video sequence.

        Parameters
        ----------
        x : torch.Tensor
            Optical-flow video tensor of shape ``(B, T, 3, H_in, W_in)``.

        Returns
        -------
        torch.Tensor
            Spatial patch embeddings of shape ``(B, T, P, embed_dim)`` 
            where P = patch_grid[0] × patch_grid[1].
        """
        b, t, c, h, w = x.shape
        assert c == 3, f"Expected 3 input channels (Horizontal Flow, Vertical Flow, Optical Strain), but got {c}."

        # Defensively cast to float32 to prevent float64 mismatches
        x = x.float()

        # ---- 1. Prepare for 3D processing ----
        # (B, T, 3, H_in, W_in) → (B, 3, T, H_in, W_in)
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        # ---- 2. Dynamic multi-stream 3D convolution ----
        # Input  per stream: (B, 3, T, H_in, W_in)
        # Output per stream: (B, F_i, T, P_H, P_W)
        # ---- 3. Concatenate along channel axis ----
        # (B, sum(F), T, P_H, P_W)
        out = torch.cat([stream(x) for stream in self.streams], dim=1)
        
        # ---- 4. 1x1x1 Projection to embed_dim ----
        # (B, sum(F), T, P_H, P_W) -> (B, embed_dim, T, P_H, P_W)
        out = self.feature_proj(out)
        out = out.contiguous()

        # ---- 5. Sequence Reshaping ----
        # (B, embed_dim, T, P_H, P_W) → (B, T, P_H, P_W, embed_dim)
        b, c_out, t, h_out, w_out = out.shape
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        
        # (B, T, P_H, P_W, embed_dim) → (B, T, P, embed_dim)
        out = out.view(b, t, h_out * w_out, c_out)

        return out
