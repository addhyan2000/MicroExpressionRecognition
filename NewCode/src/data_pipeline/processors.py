# -*- coding: utf-8 -*-
"""Production-grade processor classes for the MER preprocessing pipeline.

Each class encapsulates a single processing stage and exposes a ``process()``
method that accepts *and returns* NumPy ``float32`` arrays.  All methods
enforce strict shape and dtype validation before computation.

Classes:
    TemporalInterpolator: Resample a clip to a fixed frame count via linear
        interpolation along the temporal axis.
    EVMProcessor: Eulerian Video Magnification (colour amplification).
        *Math is a placeholder; type/shape contracts are enforced.*
    FlowExtractor: Dense Farnebäck optical flow + optical strain estimation.
"""

from __future__ import annotations

import logging
from typing import Dict

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d


logger = logging.getLogger("mer_preprocessing")


# ─── helper ──────────────────────────────────────────────────────────────
def _validate_video_tensor(
    frames: NDArray[np.float32],
    *,
    caller: str,
    allow_3d: bool = False,
) -> None:
    """Validate that *frames* is a well-formed video tensor.

    Args:
        frames: Array to validate.
        caller: Name of the calling class (used in error messages).
        allow_3d: If ``True``, also accept 3-D tensors ``(T, H, W)``.

    Raises:
        TypeError: If *frames* is not an ``ndarray``.
        ValueError: If the shape, dtype, or dimensionality contract
            is violated.
    """
    if not isinstance(frames, np.ndarray):
        raise TypeError(
            f"[{caller}] Expected np.ndarray, got {type(frames).__name__}."
        )

    if frames.dtype != np.float32:
        raise ValueError(
            f"[{caller}] Expected dtype float32, got {frames.dtype}."
        )

    valid_ndims = {4, 3} if allow_3d else {4}
    if frames.ndim not in valid_ndims:
        expected = "3-D (T, H, W) or 4-D (T, H, W, C)" if allow_3d else "4-D (T, H, W, C)"
        raise ValueError(
            f"[{caller}] Expected {expected} tensor, "
            f"got {frames.ndim}-D with shape {frames.shape}."
        )

    if frames.shape[0] == 0:
        raise ValueError(
            f"[{caller}] Temporal dimension T must be > 0, "
            f"got shape {frames.shape}."
        )


# ======================================================================
# Temporal Interpolation
# ======================================================================

class TemporalInterpolator:
    """Resample a variable-length clip to a fixed number of frames.

    Uses ``scipy.interpolate.interp1d`` with **linear** interpolation
    along the temporal axis (``axis=0``) for sub-frame accuracy.  Spatial
    dimensions and channel count are preserved exactly.

    Attributes:
        target_frames: The fixed number of output frames.
    """

    def __init__(self, target_frames: int) -> None:
        """Initialise the interpolator.

        Args:
            target_frames: Desired number of output frames (e.g. 32).

        Raises:
            ValueError: If *target_frames* is not a positive integer.
        """
        if target_frames <= 0:
            raise ValueError(
                f"target_frames must be > 0, got {target_frames}"
            )
        self.target_frames: int = target_frames
        logger.debug(
            "TemporalInterpolator initialised — target_frames=%d",
            self.target_frames,
        )

    def process(self, frames: NDArray[np.float32]) -> NDArray[np.float32]:
        """Temporally resample *frames* to ``self.target_frames``.

        Linear interpolation is applied independently to every spatial
        position and every channel, guaranteeing that the output has
        shape ``(target_frames, H, W, C)`` with dtype ``float32``.

        Args:
            frames: Input video tensor of shape ``(T_in, H, W, C)``
                with dtype ``float32``.

        Returns:
            Resampled tensor of shape ``(target_frames, H, W, C)``,
            dtype ``float32``.

        Raises:
            TypeError: If *frames* is not an ``ndarray``.
            ValueError: If the shape or dtype contract is violated.
        """
        _validate_video_tensor(frames, caller="TemporalInterpolator")

        t_in: int = frames.shape[0]

        # Fast path: no interpolation needed
        if t_in == self.target_frames:
            logger.debug(
                "TemporalInterpolator: T_in == target (%d) — passthrough.",
                t_in,
            )
            return frames.copy().astype(np.float32)

        # Normalised positions of original and target frames in [0, 1]
        src_positions: NDArray[np.float32] = np.linspace(
            0.0, 1.0, num=t_in, dtype=np.float32,
        )
        dst_positions: NDArray[np.float32] = np.linspace(
            0.0, 1.0, num=self.target_frames, dtype=np.float32,
        )

        # Reshape to (T, -1) so interp1d treats spatial+channel as a
        # single vector — avoids nested loops and is memory-friendly.
        spatial_shape = frames.shape[1:]                     # (H, W, C)
        flat: NDArray[np.float32] = frames.reshape(t_in, -1)  # (T, H*W*C)

        interpolator = interp1d(
            src_positions,
            flat,
            axis=0,
            kind="linear",
            assume_sorted=True,
        )
        interpolated_flat: NDArray[np.float32] = interpolator(
            dst_positions,
        ).astype(np.float32)

        output: NDArray[np.float32] = interpolated_flat.reshape(
            self.target_frames, *spatial_shape,
        )

        logger.debug(
            "TemporalInterpolator: %d → %d frames (linear interp1d).",
            t_in,
            self.target_frames,
        )
        return output


# ======================================================================
# Eulerian Video Magnification
# ======================================================================

class EVMProcessor:
    """Colour-amplified Eulerian Video Magnification (EVM) processor.

    The final implementation will:
    1. Build a Gaussian pyramid per frame.
    2. Apply a temporal ideal bandpass filter (FFT) within the
       ``[low_omega, high_omega]`` band.
    3. Amplify the filtered signal by ``alpha``.
    4. Reconstruct the full-resolution magnified video.

    **Math is a placeholder** — the type, dtype, and shape contracts
    are fully enforced so that downstream consumers can depend on the
    interface.

    Attributes:
        alpha: Amplification factor.
        low_omega: Lower cutoff frequency in Hz.
        high_omega: Upper cutoff frequency in Hz.
        pyramid_levels: Number of Gaussian pyramid levels.
    """

    def __init__(
        self,
        alpha: float,
        low_omega: float,
        high_omega: float,
        pyramid_levels: int = 3,
    ) -> None:
        """Initialise the EVM processor.

        Args:
            alpha: Amplification factor (e.g. 20.0).
            low_omega: Lower cutoff frequency in Hz (e.g. 0.4).
            high_omega: Upper cutoff frequency in Hz (e.g. 3.0).
            pyramid_levels: Number of levels for the Gaussian /
                Laplacian pyramid decomposition.

        Raises:
            ValueError: If ``low_omega >= high_omega`` or ``alpha <= 0``.
        """
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if low_omega >= high_omega:
            raise ValueError(
                f"low_omega ({low_omega}) must be < high_omega ({high_omega})"
            )

        self.alpha: float = alpha
        self.low_omega: float = low_omega
        self.high_omega: float = high_omega
        self.pyramid_levels: int = pyramid_levels

        logger.debug(
            "EVMProcessor initialised — α=%.1f, ω=[%.2f, %.2f] Hz, "
            "levels=%d",
            self.alpha,
            self.low_omega,
            self.high_omega,
            self.pyramid_levels,
        )

    def process(
        self,
        frames: NDArray[np.float32],
        fps: int,
    ) -> NDArray[np.float32]:
        """Apply Eulerian Video Magnification to *frames*.

        Args:
            frames: Input video tensor of shape ``(T, H, W, C)``
                with dtype ``float32``.
            fps: Frames-per-second of the input clip (needed by the
                temporal bandpass filter).

        Returns:
            Magnified video tensor of the **same shape** as *frames*,
            dtype ``float32``.

        Raises:
            TypeError: If *frames* is not an ``ndarray``.
            ValueError: If the shape contract is violated.

        TODO:
            Wire in the full Gaussian-pyramid + FFT bandpass +
            amplification + reconstruction pipeline from ``PyEVM``.
        """
        _validate_video_tensor(frames, caller="EVMProcessor")

        logger.debug(
            "EVMProcessor: passthrough on tensor %s @ %d fps.  "
            "(Implementation pending.)",
            frames.shape,
            fps,
        )
        # ── Placeholder: identity passthrough ────────────────────────
        return frames.copy().astype(np.float32)


# ======================================================================
# Optical Flow + Strain Extraction
# ======================================================================

class FlowExtractor:
    """Dense optical-flow and optical-strain extractor.

    Computes frame-to-frame Farnebäck dense optical flow via
    ``cv2.calcOpticalFlowFarneback``, then derives the 2-D optical
    strain tensor from the spatial gradients of the flow field:

    .. math::

        \\varepsilon_{xx} = \\frac{\\partial u}{\\partial x}, \\quad
        \\varepsilon_{yy} = \\frac{\\partial v}{\\partial y}, \\quad
        \\varepsilon_{xy} = \\frac{1}{2}\\left(
            \\frac{\\partial u}{\\partial y} +
            \\frac{\\partial v}{\\partial x}
        \\right)

    The scalar strain magnitude returned is:

    .. math::

        \\|\\varepsilon\\| = \\sqrt{
            \\varepsilon_{xx}^2 +
            \\varepsilon_{yy}^2 +
            \\varepsilon_{xy}^2
        }

    The ``process`` method returns a **single** ``float32`` array of
    shape ``(T-1, H, W, 3)`` with channels ``[flow_x, flow_y, strain]``.

    Attributes:
        flow_params: Parameters forwarded to the OpenCV flow estimator.
    """

    def __init__(self) -> None:
        """Initialise the flow extractor with default Farnebäck params."""
        self.flow_params: Dict[str, float | int] = {
            "pyr_scale": 0.5,
            "levels": 3,
            "winsize": 15,
            "iterations": 3,
            "poly_n": 5,
            "poly_sigma": 1.2,
        }
        logger.debug("FlowExtractor initialised with default parameters.")

    # ── private helper ───────────────────────────────────────────────
    def _compute_optical_strain(
        self,
        flow: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Compute the scalar optical-strain magnitude from a flow field.

        Args:
            flow: Dense optical flow of shape ``(H, W, 2)`` where
                ``flow[..., 0]`` is *u* (horizontal) and
                ``flow[..., 1]`` is *v* (vertical).

        Returns:
            Strain magnitude of shape ``(H, W)``, dtype ``float32``.
        """
        u: NDArray[np.float32] = flow[..., 0]  # horizontal component
        v: NDArray[np.float32] = flow[..., 1]  # vertical component

        # Spatial gradients — np.gradient returns (∂/∂y, ∂/∂x) for a 2-D
        # array because axis-0 corresponds to rows (y) and axis-1 to
        # columns (x).
        du_dy, du_dx = np.gradient(u)  # partials of u
        dv_dy, dv_dx = np.gradient(v)  # partials of v

        epsilon_xx: NDArray[np.float32] = du_dx.astype(np.float32)
        epsilon_yy: NDArray[np.float32] = dv_dy.astype(np.float32)
        epsilon_xy: NDArray[np.float32] = (
            0.5 * (du_dy + dv_dx)
        ).astype(np.float32)

        strain: NDArray[np.float32] = np.sqrt(
            epsilon_xx ** 2 + epsilon_yy ** 2 + epsilon_xy ** 2,
        ).astype(np.float32)

        return strain

    # ── public API ───────────────────────────────────────────────────
    def process(
        self,
        frames: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Compute optical flow and strain for consecutive frame pairs.

        Args:
            frames: Input video tensor of shape ``(T, H, W, C)`` or
                ``(T, H, W)`` for pre-greyscale data, dtype ``float32``.
                Pixel values may be in ``[0.0, 1.0]`` (will be scaled
                to ``[0, 255]``) or ``[0.0, 255.0]`` range.

        Returns:
            Stacked ``float32`` array of shape ``(T-1, H, W, 3)`` with
            channels ordered as ``[flow_x, flow_y, optical_strain]``.

        Raises:
            TypeError: If *frames* is not an ``ndarray``.
            ValueError: If the shape contract is violated or
                ``T < 2`` (need at least two frames for one flow pair).
        """
        _validate_video_tensor(
            frames, caller="FlowExtractor", allow_3d=True,
        )

        t: int = frames.shape[0]
        if t < 2:
            raise ValueError(
                f"[FlowExtractor] Need at least 2 frames to compute flow, "
                f"got T={t}."
            )

        h: int = frames.shape[1]
        w: int = frames.shape[2]
        n_pairs: int = t - 1

        # Pre-allocate output buffer — (T-1, H, W, 3)
        output: NDArray[np.float32] = np.empty(
            (n_pairs, h, w, 3), dtype=np.float32,
        )

        # ── Lazy scaling flag (no global tensor copy) ────────────────
        needs_scaling: bool = frames.max() <= 1.0
        if needs_scaling:
            logger.debug(
                "FlowExtractor: detected [0, 1] range — will rescale "
                "per-frame inside _to_grey.",
            )

        # ── Determine if grayscale conversion is needed ──────────────
        is_already_grey: bool = (
            frames.ndim == 3                              # (T, H, W)
            or (frames.ndim == 4 and frames.shape[3] == 1)  # (T, H, W, 1)
        )

        def _to_grey(frame_f32: NDArray[np.float32]) -> NDArray[np.uint8]:
            """Convert a single frame to uint8 greyscale."""
            if needs_scaling:
                frame_f32 = frame_f32 * np.float32(255.0)
            frame_u8: NDArray[np.uint8] = np.clip(
                frame_f32, 0.0, 255.0,
            ).astype(np.uint8)
            if is_already_grey:
                # Squeeze away a potential trailing channel dim
                return frame_u8.reshape(h, w)
            # 3-channel BGR → greyscale
            return cv2.cvtColor(frame_u8, cv2.COLOR_BGR2GRAY)

        # ── Iterative flow computation (no list comprehension) ───────
        prev_grey: NDArray[np.uint8] = _to_grey(frames[0])

        for i in range(n_pairs):
            curr_grey: NDArray[np.uint8] = _to_grey(frames[i + 1])

            flow: NDArray[np.float32] = cv2.calcOpticalFlowFarneback(
                prev_grey,
                curr_grey,
                None,  # type: ignore[arg-type]
                pyr_scale=self.flow_params["pyr_scale"],
                levels=int(self.flow_params["levels"]),
                winsize=int(self.flow_params["winsize"]),
                iterations=int(self.flow_params["iterations"]),
                poly_n=int(self.flow_params["poly_n"]),
                poly_sigma=self.flow_params["poly_sigma"],
                flags=0,
            )  # shape: (H, W, 2), dtype float32

            flow_x: NDArray[np.float32] = flow[..., 0]
            flow_y: NDArray[np.float32] = flow[..., 1]
            strain: NDArray[np.float32] = self._compute_optical_strain(flow)

            output[i, :, :, 0] = flow_x
            output[i, :, :, 1] = flow_y
            output[i, :, :, 2] = strain

            # Carry forward — avoids redundant greyscale conversion
            prev_grey = curr_grey

        logger.debug(
            "FlowExtractor: computed flow+strain for %d pairs — "
            "output shape %s.",
            n_pairs,
            output.shape,
        )
        return output
