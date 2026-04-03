"""
flow_strain_extractor.py — Farnebäck Optical Flow & Strain Computation
======================================================================

Implements ``FlowStrainExtractor``, which computes three physics-based
motion modalities from a sequence of grayscale frames:

    1. **Horizontal Optical Flow (u)** — pixel displacement in x.
    2. **Vertical Optical Flow (v)**   — pixel displacement in y.
    3. **Optical Strain (os)**         — deformation magnitude from
       spatial gradients of the flow field.

Optical Strain Formulation:
    Given the flow field (u, v) between two consecutive frames, the
    strain tensor components are:

        ε_xx = ∂u/∂x          (horizontal normal strain)
        ε_yy = ∂v/∂y          (vertical normal strain)
        ε_xy = ½(∂u/∂y + ∂v/∂x)  (shear strain)

    The scalar strain magnitude is:

        os = √(ε_xx² + ε_yy² + ε_xy²)

    This captures the *intensity* of facial deformation regardless of
    direction — exactly what distinguishes micro-expressions from the
    neutral baseline.

Normalization:
    Each channel (u, v, os) is independently Min-Max normalized to
    [0, 1] across the entire temporal sequence.  This prevents the
    strain channel (which has a different numerical range than raw
    flow) from dominating during training.

Output Shape:
    ``[3, T, H, W]`` where T = number of frame pairs (L − 1 = 32).

Author  : Addhyan
Stage   : 1 — Data Pipeline / Step 2 — Extraction
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class FlowStrainExtractor:
    """
    Computes optical flow and optical strain from a frame sequence.

    Parameters
    ----------
    farneback_params : dict, optional
        Keyword arguments passed to ``cv2.calcOpticalFlowFarneback``.
        If None, uses carefully-tuned defaults for micro-expression
        analysis (see Notes).

    Notes
    -----
    Default Farnebäck parameters:
        • pyr_scale = 0.5  — standard pyramid scale
        • levels = 3       — sufficient for 224×224 images
        • winsize = 15     — balances sensitivity vs. noise
        • iterations = 3   — convergence iterations
        • poly_n = 5       — polynomial expansion neighbourhood
        • poly_sigma = 1.2 — Gaussian smoothing for poly expansion
        • flags = 0        — no special flags

    These defaults follow the STSTNet paper's preprocessing and are
    validated for micro-expression datasets where motion amplitudes
    are inherently small (sub-pixel to a few pixels).

    Thread Safety:
        This class calls ``cv2.setNumThreads(0)`` at construction
        to prevent OpenCV's internal threading from clashing with
        Python's ``ProcessPoolExecutor``.  This is a *per-process*
        setting and is safe to call multiple times.
    """

    def __init__(
        self,
        farneback_params: Optional[dict] = None,
    ) -> None:
        # ── CRITICAL: Disable OpenCV internal threading ─────────────
        # Without this, ProcessPoolExecutor + OpenCV = deadlock.
        cv2.setNumThreads(0)

        # ── Farnebäck parameters ────────────────────────────────────
        self._fb_params = farneback_params or {
            "pyr_scale": 0.5,
            "levels": 3,
            "winsize": 15,
            "iterations": 3,
            "poly_n": 5,
            "poly_sigma": 1.2,
            "flags": 0,
        }

    # ─────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────

    def extract(
        self, frames: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Compute flow + strain tensor from a frame sequence.

        Parameters
        ----------
        frames : np.ndarray
            Grayscale frame sequence of shape ``(L, H, W)`` with
            dtype float32 and values in [0, 255].

        Returns
        -------
        np.ndarray or None
            Float32 tensor of shape ``(3, T, H, W)`` where:
                • Channel 0 = normalized horizontal flow (u)
                • Channel 1 = normalized vertical flow   (v)
                • Channel 2 = normalized optical strain   (os)
            T = L − 1 (number of consecutive frame pairs).
            All values in [0, 1] after Min-Max normalization.
            Returns None if extraction fails.

        Raises
        ------
        ValueError
            If ``frames`` has fewer than 2 frames.
        """
        if frames.ndim != 3:
            raise ValueError(
                f"Expected frames of shape (L, H, W), got {frames.shape}"
            )
        L, H, W = frames.shape
        if L < 2:
            raise ValueError(
                f"Need at least 2 frames for optical flow (got {L})."
            )

        T = L - 1  # Number of frame pairs

        # ── Pre-allocate output arrays ──────────────────────────────
        u_seq = np.zeros((T, H, W), dtype=np.float32)
        v_seq = np.zeros((T, H, W), dtype=np.float32)
        os_seq = np.zeros((T, H, W), dtype=np.float32)

        # ── Compute flow + strain for each consecutive pair ─────────
        for t in range(T):
            frame_prev = frames[t].astype(np.uint8)
            frame_curr = frames[t + 1].astype(np.uint8)

            # ── Farnebäck dense optical flow ────────────────────────
            flow = cv2.calcOpticalFlowFarneback(
                prev=frame_prev,
                next=frame_curr,
                flow=None,
                **self._fb_params,
            )
            # flow shape: (H, W, 2) — channel 0 = u (horiz), 1 = v (vert)

            u = flow[:, :, 0]   # Horizontal displacement
            v = flow[:, :, 1]   # Vertical displacement

            # ── Optical Strain computation ──────────────────────────
            #
            #  ε_xx = ∂u/∂x       (horizontal normal strain)
            #  ε_yy = ∂v/∂y       (vertical normal strain)
            #  ε_xy = ½(∂u/∂y + ∂v/∂x)   (shear strain)
            #
            #  Magnitude: os = √(ε_xx² + ε_yy² + ε_xy²)
            #
            #  np.gradient returns gradient along each axis:
            #    np.gradient(u) returns [∂u/∂y, ∂u/∂x] for a 2D array
            #    (axis 0 = rows = y, axis 1 = cols = x)
            # ────────────────────────────────────────────────────────
            du_dy, du_dx = np.gradient(u)
            dv_dy, dv_dx = np.gradient(v)

            eps_xx = du_dx
            eps_yy = dv_dy
            eps_xy = 0.5 * (du_dy + dv_dx)

            strain_magnitude = np.sqrt(
                eps_xx ** 2 + eps_yy ** 2 + eps_xy ** 2
            )

            u_seq[t] = u
            v_seq[t] = v
            os_seq[t] = strain_magnitude

        # ── Min-Max Normalization per channel ───────────────────────
        #
        #  Each channel is normalized independently across all T
        #  timesteps.  This is critical because:
        #    • u, v can be negative (leftward/upward motion)
        #    • strain is always non-negative
        #    • Their numerical ranges differ significantly
        #
        #  Formula: x_norm = (x - x_min) / (x_max - x_min + ε)
        #  ε = 1e-8 prevents division by zero for static clips.
        # ────────────────────────────────────────────────────────────
        u_norm = self._minmax_normalize(u_seq)
        v_norm = self._minmax_normalize(v_seq)
        os_norm = self._minmax_normalize(os_seq)

        # ── Stack into (3, T, H, W) ────────────────────────────────
        tensor = np.stack([u_norm, v_norm, os_norm], axis=0)

        return tensor

    # ─────────────────────────────────────────────────────────────────
    #  Private Helpers
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _minmax_normalize(
        arr: np.ndarray,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """
        Min-Max normalize an array to [0, 1].

        Parameters
        ----------
        arr : np.ndarray
            Input array of any shape.
        eps : float
            Small constant to prevent division by zero.

        Returns
        -------
        np.ndarray
            Normalized array with same shape, dtype float32.
        """
        arr_min = arr.min()
        arr_max = arr.max()
        return ((arr - arr_min) / (arr_max - arr_min + eps)).astype(np.float32)
