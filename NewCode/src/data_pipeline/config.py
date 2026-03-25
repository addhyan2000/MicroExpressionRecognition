# -*- coding: utf-8 -*-
"""Pipeline configuration via a frozen dataclass.

This module defines ``PipelineConfig``, the single source of truth for every
tunable parameter in the preprocessing pipeline.  Configs can be instantiated
directly, loaded from a plain ``dict`` (e.g. parsed from YAML/JSON), or
validated programmatically before a run begins.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable configuration for a single preprocessing run.

    Attributes:
        dataset_name: Human-readable identifier for the dataset being
            processed (e.g. ``"CASME_II"`` or ``"CAS_ME_2"``).
        input_dir: Absolute path to the root directory that contains the
            raw subject / video folders.
        output_dir: Absolute path to the directory where processed
            ``.npy`` tensors will be saved.
        target_fps: Frames-per-second to which every clip will be
            temporally resampled.
        target_frames: Fixed number of frames that each clip will be
            interpolated (stretched / compressed) to.
        target_height: Spatial height to resize each frame to.
        target_width: Spatial width to resize each frame to.
        evm_alpha: Amplification factor for Eulerian Video Magnification.
        evm_low_omega: Lower cutoff frequency (Hz) for the EVM bandpass
            filter.
        evm_high_omega: Upper cutoff frequency (Hz) for the EVM bandpass
            filter.
        evm_pyramid_levels: Number of Gaussian/Laplacian pyramid levels
            used during EVM decomposition.
        checkpoint_file: Absolute path to the JSON checkpoint file.
            If None, defaults to ``output_dir / "processed_state.json"``.
        log_file: Absolute path to the pipeline log file.
            If None, defaults to ``output_dir / "preprocessing.log"``.
        num_workers: Number of parallel workers for I/O-bound loading.
    """

    dataset_name: str
    input_dir: Path
    output_dir: Path

    # Temporal resampling
    target_fps: int = 30
    target_frames: int = 32

    # Spatial resampling
    target_height: int = 224
    target_width: int = 224

    # Eulerian Video Magnification
    evm_alpha: float = 20.0
    evm_low_omega: float = 0.4
    evm_high_omega: float = 3.0
    evm_pyramid_levels: int = 3

    # Infrastructure
    checkpoint_file: Optional[Path] = field(default=None)
    log_file: Optional[Path] = field(default=None)
    num_workers: int = 0

    def __post_init__(self) -> None:
        """Coerce string paths to ``Path`` and validate configuration."""
        # frozen=True requires object.__setattr__ for post-init fixups.
        object.__setattr__(self, "input_dir", Path(self.input_dir).resolve())
        object.__setattr__(self, "output_dir", Path(self.output_dir).resolve())

        # 1. Strict Path Validation
        if not self.input_dir.exists():
            raise FileNotFoundError(
                f"Input directory does not exist: {self.input_dir}"
            )

        # 2. Frequency Logic Check
        if self.evm_low_omega >= self.evm_high_omega:
            raise ValueError(
                f"evm_low_omega ({self.evm_low_omega}) must be strictly less "
                f"than evm_high_omega ({self.evm_high_omega})."
            )

        # 3. Dimension Validation
        dimensions = {
            "target_fps": self.target_fps,
            "target_height": self.target_height,
            "target_width": self.target_width,
            "target_frames": self.target_frames,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"{name} must be a positive integer, got {value}."
                )

        # 4. Flow Minimum Validation
        if self.target_frames < 2:
            raise ValueError(
                f"target_frames must be >= 2 for Optical Flow extraction, "
                f"got {self.target_frames}."
            )

        # 5. Scientific Nyquist Validation (Aliasing Prevention)
        nyquist_freq = self.target_fps / 2.0
        if self.evm_high_omega >= nyquist_freq:
            raise ValueError(
                f"Scientific Error: evm_high_omega ({self.evm_high_omega} Hz) "
                f"exceeds the Nyquist limit ({nyquist_freq} Hz) for "
                f"target_fps={self.target_fps}. Higher frequencies will alias."
            )

        # Resolve/Default infrastructure paths
        if self.checkpoint_file is None:
            object.__setattr__(
                self,
                "checkpoint_file",
                (self.output_dir / "processed_state.json").resolve(),
            )
        else:
            object.__setattr__(
                self, "checkpoint_file", Path(self.checkpoint_file).resolve()
            )

        if self.log_file is None:
            object.__setattr__(
                self, "log_file", (self.output_dir / "preprocessing.log").resolve()
            )
        else:
            object.__setattr__(self, "log_file", Path(self.log_file).resolve())

    def setup_filesystem(self) -> None:
        """Create necessary directories for the pipeline."""
        if self.output_dir.exists() and self.output_dir.is_file():
            raise RuntimeError(
                f"Directory Conflict: Output path {self.output_dir} exists "
                f"but is a file. The pipeline requires a directory."
            )
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> PipelineConfig:
        """Construct a ``PipelineConfig`` from a plain dictionary."""
        # Filter to only recognised field names.
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in cfg.items() if k in valid_keys}
        return cls(**filtered)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable multi-line summary of this config."""
        lines = [
            f"{'='*60}",
            f"  Pipeline Configuration — {self.dataset_name}",
            f"{'='*60}",
            f"  Input dir        : {self.input_dir}",
            f"  Output dir       : {self.output_dir}",
            f"  Target FPS       : {self.target_fps}",
            f"  Target frames    : {self.target_frames}",
            f"  Spatial size     : {self.target_height}×{self.target_width}",
            f"  EVM α            : {self.evm_alpha}",
            f"  EVM ω_low        : {self.evm_low_omega} Hz",
            f"  EVM ω_high       : {self.evm_high_omega} Hz",
            f"  EVM pyr levels   : {self.evm_pyramid_levels}",
            f"  Checkpoint file  : {self.checkpoint_file}",
            f"  Log file         : {self.log_file}",
            f"{'='*60}",
        ]
        return "\n".join(lines)
