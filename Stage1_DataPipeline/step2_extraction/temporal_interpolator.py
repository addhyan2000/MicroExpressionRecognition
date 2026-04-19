"""
temporal_interpolator.py — TIM-Inspired Frame Sampling & Loading
================================================================

Implements ``TemporalInterpolator``, which solves the fundamental
problem of heterogeneous temporal length across datasets.

Problem Statement:
    Preprocessed EVM `.avi` videos have varying frame counts (e.g., 56 frames).
    The downstream Spatio-Temporal Transformer requires a fixed-length tensor input.

Solution (Temporal Interpolation Model — TIM):
    We sample exactly L=33 uniformly-spaced frame *indices*. This
    yields 33 frames, which produce 32 consecutive pairs for optical-
    flow computation — matching the required temporal dimension T=32.

    We read the entire video into RAM, buffer it, then compute indices
    from 0 to (total_frames - 1).

Author  : Addhyan
Stage   : 1 — Data Pipeline / Step 2 — Extraction
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

class TemporalInterpolator:
    """
    Generates L uniformly-spaced frame indices and loads/resizes them from `.avi`.

    Parameters
    ----------
    target_length : int
        Number of frames to sample.  Default is 33 (yielding 32 pairs
        for optical flow).
    spatial_size : Tuple[int, int]
        Target (height, width) for resized frames.  Default (224, 224).

    Notes
    -----
    This class is designed to be instantiated once and called many
    times — it carries no mutable state between calls.  It is safe
    to use from ``ProcessPoolExecutor`` worker processes.
    """

    def __init__(
        self,
        target_length: int = 33,
        spatial_size: Tuple[int, int] = (224, 224),
    ) -> None:
        if target_length < 2:
            raise ValueError(
                f"target_length must be >= 2 (got {target_length}).  "
                f"Need at least 2 frames to form 1 optical-flow pair."
            )
        self._L = target_length
        self._H, self._W = spatial_size

    # ─────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────

    def compute_indices(
        self, onset: int, offset: int
    ) -> np.ndarray:
        """
        Compute L uniformly-spaced frame indices in [onset, offset].

        Uses ``numpy.linspace`` for maximally uniform spacing, then
        rounds to the nearest integer.  Duplicate indices *are*
        permitted when the clip is shorter than L (supersampling).

        Parameters
        ----------
        onset : int
            First frame number of the expression clip.
        offset : int
            Last frame number of the expression clip.

        Returns
        -------
        np.ndarray
            Integer array of shape ``(L,)`` with frame indices.
        """
        if offset < onset:
            raise ValueError(
                f"offset ({offset}) must be >= onset ({onset})."
            )

        indices = np.linspace(onset, offset, num=self._L)
        indices = np.round(indices).astype(np.int64)
        return indices

    def load_frames(
        self,
        frames_dir: str,
        onset: int,
        offset: int,
        dataset: str,
    ) -> Optional[np.ndarray]:
        """
        Load L frames from a pre-cropped .avi video file, resized to (H, W) grayscale.

        Note: `onset`, `offset`, and `dataset` arguments are maintained for API 
        compatibility with the orchestrator, but are ignored. The video is already
        cropped, so we calculate indices dynamically over [0, total_frames - 1].

        Parameters
        ----------
        frames_dir : str
            Absolute path to the .avi video file.
        onset : int
            Ignored. Maintained for interface compatibility.
        offset : int
            Ignored. Maintained for interface compatibility.
        dataset : str
            Ignored. Maintained for interface compatibility.

        Returns
        -------
        np.ndarray or None
            Float32 array of shape ``(L, H, W)`` with pixel values in
            [0, 255], or ``None`` if the video fails to load or is empty.
        """
        filepath = Path(frames_dir)
        
        if not filepath.exists() or not filepath.is_file():
            logger.error(f"Video file not found or invalid: {filepath}")
            return None

        cap = cv2.VideoCapture(str(filepath))
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {filepath}")
            return None

        # ── Read all frames into RAM (Avoid Seeking Bugs) ────────────
        all_frames: List[np.ndarray] = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Preprocessing: Grayscale, resize, float32
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(
                gray_frame,
                (self._W, self._H),
                interpolation=cv2.INTER_LINEAR,
            )
            all_frames.append(resized_frame.astype(np.float32))

        cap.release()

        total_frames = len(all_frames)
        if total_frames == 0:
            logger.error(f"Video file is empty: {filepath}")
            return None

        # ── Dynamic Indexing ─────────────────────────────────────────
        # Video is PRE-CROPPED. Calculate indices from exactly what we loaded.
        indices = self.compute_indices(0, total_frames - 1)
        
        # ── Extract and Return ───────────────────────────────────────
        selected_frames = []
        for idx in indices:
            # Bounds checking
            clamped_idx = max(0, min(int(idx), total_frames - 1))
            selected_frames.append(all_frames[clamped_idx])
            
        return np.stack(selected_frames, axis=0)
