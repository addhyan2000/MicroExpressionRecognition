"""SLSTT-LSTM Temporal Modeler for Micro-Expression Recognition.

This module implements the second stage of the hybrid CNN-Transformer pipeline:
a lightweight Spatio-Temporal Transformer followed by an LSTM sequence
aggregator.  It consumes the spatial patch embeddings produced by Step 1
(SpatialEncoder) and outputs a single fixed-length sequence embedding per
video clip for downstream classification.

Pipeline Position
-----------------
  Step 1  SpatialEncoder   :  (B, T, 3, 28, 28)  →  (B, T, 100, 16)
  Step 2  SLSTTLSTM        :  (B, T, 100, 16)    →  (B, 128)     ← this module
  Step 3  ClassificationHead  (future)

Architecture Overview
---------------------
  Input  (B, T, 100, 16)        # 100 spatial patches × 16-dim embeddings
    |
    v
  pack_padded_sequence           # Mask zero-padded frames (variable length)
    |                            # Extracts valid frames -> (Total_Valid, 100, 16)
    v
  + CLS Token                    # (Total_Valid, 1, 16) -> (Total_Valid, 101, 16)
  + Learnable Position Embed     # (1, 101, 16)
    |
    v
  Positional Dropout               # Regularise spatial embeddings
    |
    v
  TransformerEncoder             # 2 layers, 4 heads, d_model=16, ff=64
    |                            # Captures long-range spatial relationships
    v                            # across all 100 facial patches + CLS token
  Extract CLS Token              → (Total_Valid, 16)
    |
    v
  Reconstruct PackedSequence     # Restore packed format for LSTM
    |
    v
  BiLSTM (in=16, hidden=64)      # Bidirectional processing of temp dynamics
    |
    v
  Concat Final State             → (B, 128)
    |
    v
  Terminal Dropout               # Prevent overfitting on tiny datasets
    |
    v
  Output Sequence Embedding      → (B, 128)

Reference Papers
-----------------
  - SLSTT: Zhang et al., "Short and Long Range Relation Based
    Spatio-Temporal Transformer for Micro-Expression Recognition",
    IEEE Transactions on Affective Computing, 2022.
  - STSTNet: Liong et al., "Shallow Triple Stream Three-dimensional CNN
    for Micro-expression Recognition", FG 2019.

Author  : Addhyan Pant
Project : Master's Thesis — Micro-Expression Recognition
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence

__all__ = ["SLSTTLSTM"]


class SLSTTLSTM(nn.Module):
    """SLSTT-LSTM: Spatio-Temporal Transformer with LSTM Aggregation.

    Processes a sequence of spatial patch embeddings through a lightweight
    Transformer encoder (capturing intra-frame spatial relationships) and
    then a BiLSTM (capturing inter-frame temporal dynamics). Supports
    variable-length video sequences via ``pack_padded_sequence``.

    Parameters
    ----------
    num_patches : int, optional
        Number of spatial patches per frame (default: 100, i.e. 10×10).
    embed_dim : int, optional
        Dimension of each patch embedding (default: 16).
    num_heads : int, optional
        Number of attention heads in the Transformer (default: 4).
        Must evenly divide ``embed_dim``.
    num_transformer_layers : int, optional
        Number of TransformerEncoderLayer blocks (default: 2).
    ff_dim : int, optional
        Hidden dimension of the feed-forward network inside each
        Transformer layer (default: 64).
    lstm_hidden : int, optional
        Hidden-state size of the LSTM aggregator (default: 64).
    lstm_layers : int, optional
        Number of stacked LSTM layers (default: 1).
    dropout : float, optional
        Dropout rate for the Transformer encoder, positional
        embedding, and terminal output (default: 0.1).
    lstm_dropout : float, optional
        Dropout rate applied between stacked LSTM layers (default: 0.0).
        Only effective when ``lstm_layers > 1``; PyTorch ignores this
        value for single-layer LSTMs.

    Shape
    -----
    - Input:  ``(B, T, 100, 16)``
              B = batch size, T = number of temporal frames (may be padded).
    - lengths (optional): ``(B,)``
              True (unpadded) frame count per video in the batch.
    - Output: ``(B, 128)``
              A fixed-length sequence embedding per video clip (bidirectional).
    """

    def __init__(
        self,
        num_patches: int = 100,
        embed_dim: int = 16,
        num_heads: int = 4,
        num_transformer_layers: int = 2,
        ff_dim: int = 64,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        lstm_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.lstm_hidden = lstm_hidden

        # ---- Class Token ---------------------------------------------------
        # Avoids spatial dilution of GAP. Learns to aggregate the highly
        # discriminative facial signals via self-attention across the patches.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ---- Learnable Positional Embedding --------------------------------
        # Encodes spatial position. +1 accounts for the CLS token.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ---- Positional Dropout --------------------------------------------
        self.pos_drop = nn.Dropout(p=dropout)

        # ---- Spatio-Temporal Transformer Encoder ---------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
            norm=nn.LayerNorm(embed_dim)
        )

        # ---- BiLSTM Temporal Aggregator ------------------------------------
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            bidirectional=True
        )

        # ---- Terminal Dropout ----------------------------------------------
        self.terminal_drop = nn.Dropout(p=dropout)

        # ---- Weight Initialization -----------------------------------------
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize Transformer and LSTM weights for stable training."""
        # ---- Transformer weight init ----
        for module in self.transformer_encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # ---- LSTM weight init ----
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                hidden_size = self.lstm_hidden
                # Apply orthogonal init to each of the 4 gates independently
                for i in range(0, 4 * hidden_size, hidden_size):
                    nn.init.orthogonal_(param.data[i:i + hidden_size, :])
            elif "bias" in name:
                nn.init.zeros_(param)
                if "bias_ih" in name:
                    # Set forget gate bias to 1.0 only for bias_ih (avoids 2.0 sum).
                    hidden_size = self.lstm_hidden
                    param.data[hidden_size:2 * hidden_size].fill_(1.0)

    @property
    def out_features(self) -> int:
        """Dimensionality of the output sequence embedding."""
        return self.lstm_hidden * 2

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Produce a fixed-length sequence embedding from spatial patches."""
        b, t, p, d = x.shape

        assert p == self.num_patches and d == self.embed_dim, (
            f"Expected input patches={self.num_patches} and "
            f"embed_dim={self.embed_dim}, but got patches={p} and "
            f"embed_dim={d}. Check SpatialEncoder output shape."
        )

        if lengths is not None:
            # Robust tensor casting
            lengths_cpu = torch.as_tensor(lengths, dtype=torch.int64, device='cpu')
            
            # Pack the 4D tensor
            packed = pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            valid_frames = packed.data  # shape: (Total_Valid, 100, 16)
            total_valid = valid_frames.shape[0]

            # Broadast CLS token and concatenate
            cls_tokens = self.cls_token.expand(total_valid, -1, -1)
            valid_frames = torch.cat((cls_tokens, valid_frames), dim=1)

            # Positional embeddings + dropout
            valid_frames = self.pos_drop(valid_frames + self.pos_embed)

            # Transformer Encoder
            valid_frames = self.transformer_encoder(valid_frames)

            # Extract CLS token state
            pooled_data = valid_frames[:, 0, :].contiguous()

            # Reconstruct Packed Sequence
            lstm_input = PackedSequence(
                pooled_data, packed.batch_sizes, packed.sorted_indices, packed.unsorted_indices
            )
        else:
            x_flat = x.contiguous().view(b * t, p, d)
            total_valid = b * t

            cls_tokens = self.cls_token.expand(total_valid, -1, -1)
            x_flat = torch.cat((cls_tokens, x_flat), dim=1)

            x_flat = self.pos_drop(x_flat + self.pos_embed)
            x_flat = self.transformer_encoder(x_flat)
            pooled_data = x_flat[:, 0, :].contiguous()
            lstm_input = pooled_data.view(b, t, d)

        # BiLSTM processing
        _, (hn, _) = self.lstm(lstm_input)

        # Extract forward and backward final states
        forward_state = hn[-2, :, :]
        backward_state = hn[-1, :, :]
        out = torch.cat([forward_state, backward_state], dim=-1)

        # Terminal Dropout
        out = self.terminal_drop(out)

        return out
