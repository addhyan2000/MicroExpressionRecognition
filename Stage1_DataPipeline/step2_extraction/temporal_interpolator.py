"""
temporal_interpolator.py — TIM-Inspired Frame Sampling & Loading
================================================================

Implements ``TemporalInterpolator``, which solves the fundamental
problem of heterogeneous temporal length across CASME II (200 fps)
and CAS(ME)^2 (30 fps).

Problem Statement:
    CASME II micro-expression clips have mean ~67 frames, while
    CAS(ME)^2 clips have mean ~35 frames.  The downstream
    Spatio-Temporal Transformer requires a fixed-length tensor input.

Solution (Temporal Interpolation Model — TIM):
    We sample exactly L=33 uniformly-spaced frame *indices* between
    Onset_Frame and Offset_Frame (inclusive of both endpoints).  This
    yields 33 frames, which produce 32 consecutive pairs for optical-
    flow computation — matching the required temporal dimension T=32.

    For a clip with N raw frames (Offset − Onset + 1):
        • If N >= 33: we *subsample* (drop intermediate frames).
        • If N < 33:  we *supersample* (some frames are reused).
    In both cases, ``numpy.linspace`` + ``numpy.round`` gives us
    exactly 33 integer indices with maximal temporal coverage.

Frame Naming Conventions (verified empirically):
    • CASME II:    ``reg_img{N}.jpg``  where N = frame number
    • CAS(ME)^2:   ``img{N}.jpg``      where N = frame number

Author  : Addhyan
Stage   : 1 — Data Pipeline / Step 2 — Extraction
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


class TemporalInterpolator:
    """
    Generates L uniformly-spaced frame indices and loads/resizes them.

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

    # ── Frame filename patterns per dataset ─────────────────────────
    _FILENAME_PATTERNS = {
        "CASME_II":       "reg_img{idx}.jpg",
        "CASME2_Squared": "img{idx}.jpg",
    }

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

        Examples
        --------
        >>> ti = TemporalInterpolator(target_length=5)
        >>> ti.compute_indices(10, 18)
        array([10, 12, 14, 16, 18])

        >>> ti.compute_indices(10, 12)   # Supersampling: only 3 raw frames
        array([10, 10, 11, 12, 12])
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
        Load L frames from disk, resized to (H, W) grayscale.

        Parameters
        ----------
        frames_dir : str
            Absolute path to the directory containing frame images.
        onset : int
            Onset frame number.
        offset : int
            Offset frame number.
        dataset : str
            Dataset identifier (``"CASME_II"`` or ``"CASME2_Squared"``)
            used to determine the filename pattern.

        Returns
        -------
        np.ndarray or None
            Float32 array of shape ``(L, H, W)`` with pixel values in
            [0, 255], or ``None`` if any frame fails to load.

        Raises
        ------
        ValueError
            If the dataset identifier is unrecognized.
        """
        # ── Resolve filename pattern ────────────────────────────────
        pattern = self._FILENAME_PATTERNS.get(dataset)
        if pattern is None:
            raise ValueError(
                f"Unknown dataset '{dataset}'.  "
                f"Expected one of: {list(self._FILENAME_PATTERNS.keys())}"
            )

        # ── Compute target indices ──────────────────────────────────
        indices = self.compute_indices(onset, offset)
        dir_path = Path(frames_dir)

        # ── Load each frame ─────────────────────────────────────────
        frames: List[np.ndarray] = []
        for i, idx in enumerate(indices):
            filename = pattern.format(idx=int(idx))
            filepath = dir_path / filename

            if not filepath.exists():
                # Attempt alternate filename formats as fallback
                alt_loaded = self._try_alternate_load(dir_path, int(idx))
                if alt_loaded is not None:
                    frames.append(alt_loaded)
                    continue
                # If we still can't find it, return None
                return None

            # Read as grayscale
            img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
            if img is None:
                # Corrupt or unreadable image
                return None

            # Resize to target spatial dimensions
            img = cv2.resize(
                img,
                (self._W, self._H),
                interpolation=cv2.INTER_LINEAR,
            )

            frames.append(img.astype(np.float32))

        # ── Stack into (L, H, W) ───────────────────────────────────
        return np.stack(frames, axis=0)

    # ─────────────────────────────────────────────────────────────────
    #  Private Helpers
    # ─────────────────────────────────────────────────────────────────

    def _try_alternate_load(
        self, dir_path: Path, idx: int
    ) -> Optional[np.ndarray]:
        """
        Try alternate frame filename conventions.

        Some datasets may have slight naming variations (e.g. zero-
        padded frame numbers).  This method attempts common patterns
        before giving up.

        Parameters
        ----------
        dir_path : Path
            Directory containing frames.
        idx : int
            Frame index to locate.

        Returns
        -------
        np.ndarray or None
            Resized grayscale frame, or None if not found.
        """
        # Common alternate patterns
        alt_patterns = [
            f"img{idx:04d}.jpg",        # Zero-padded 4 digits
            f"reg_img{idx:04d}.jpg",     # Zero-padded variant
            f"img{idx}.png",             # PNG format
            f"reg_img{idx}.png",
            f"frame{idx}.jpg",           # Generic "frame" prefix
            f"{idx}.jpg",                # Bare number
        ]

        for alt_name in alt_patterns:
            alt_path = dir_path / alt_name
            if alt_path.exists():
                img = cv2.imread(str(alt_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(
                        img,
                        (self._W, self._H),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    return img.astype(np.float32)

        return None
