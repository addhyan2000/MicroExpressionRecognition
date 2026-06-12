"""
mpi_data_pipeline.py — MPI Facial Expression Database → Tensor Pipeline
=========================================================================

Self-contained preprocessing pipeline that converts the MPI Facial
Expression Database (raw PNG frame sequences) into the same ``[3, 32, 224, 224]``
optical-flow / optical-strain tensors consumed by ``MERAblationDataset``.

It also generates a CSV file (``mpi_labels.csv``) with the **exact same
schema** as ``master_thesis_labels.csv``, so the existing dataset loader
works without modification.

MPI Directory Structure (as present on disk)::

    MPI Datast/
      MPI_large_centralcam_hi_{participant}_complete/
        Subset {NN}/
          {expression_name}/
            {participant}_{expression}_{framenum:03d}.png

Each ``(participant, subset, expression)`` triple becomes one sample.

The pipeline:
    1. **Scans** the directory tree to discover all expression sequences.
    2. **Loads** PNG frames → grayscale → 224×224 → float32.
    3. **Uniformly samples** 33 frames (→ 32 flow pairs) via TIM strategy.
    4. **Computes** Farnebäck optical flow (u, v) and optical strain per pair.
    5. **Saves** a ``[3, 32, 224, 224]`` ``.npy`` tensor per sequence.
    6. **Writes** a master CSV with matching schema.

Optical flow / strain math is reimplemented inline (identical to
``Stage1_DataPipeline/step2_extraction/flow_strain_extractor.py``) so this
module is fully self-contained within the Ablation_Study folder.

Usage::

    python Ablation_Study/mpi_data_pipeline.py
    python Ablation_Study/mpi_data_pipeline.py --mpi_root "DATASETS/MPI Datast" --workers 8
    python Ablation_Study/mpi_data_pipeline.py --dry_run     # scan only, no processing

Author : Addhyan
Stage  : Ablation Study — MPI Data Adaptation
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gc
import logging
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Project root (same anchor as ablation_config.py)
# ─────────────────────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent


# ═════════════════════════════════════════════════════════════════════════════
# 1.  MPI DIRECTORY SCANNER
# ═════════════════════════════════════════════════════════════════════════════

# Regex to extract the 4-character participant ID from folder names like
# "MPI_large_centralcam_hi_cawm_complete"
_PARTICIPANT_RE = re.compile(
    r"MPI_large_centralcam_hi_([a-z]{3}[mf])_complete", re.IGNORECASE
)

# Regex to extract subset number from "Subset 01", "Subset 13", etc.
_SUBSET_RE = re.compile(r"Subset\s+(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class MPISequenceInfo:
    """Metadata for a single MPI expression sequence discovered on disk."""
    participant_id: str       # e.g. "cawm"
    subset_idx: int           # e.g. 1
    expression: str           # e.g. "happy_laughing"
    frames_dir: str           # absolute path to the expression folder
    num_frames: int           # number of PNG files found
    video_id: str             # unique ID for tensor naming


def scan_mpi_dataset(mpi_root: Path) -> List[MPISequenceInfo]:
    """
    Walk the MPI directory tree and discover all expression sequences.

    Returns a sorted list of ``MPISequenceInfo`` objects — one per
    ``(participant, subset, expression)`` triple.
    """
    sequences: List[MPISequenceInfo] = []

    if not mpi_root.exists():
        raise FileNotFoundError(f"MPI root directory not found: {mpi_root}")

    for participant_dir in sorted(mpi_root.iterdir()):
        if not participant_dir.is_dir():
            continue
        m = _PARTICIPANT_RE.match(participant_dir.name)
        if not m:
            continue
        participant_id = m.group(1).lower()

        for subset_dir in sorted(participant_dir.iterdir()):
            if not subset_dir.is_dir():
                continue
            sm = _SUBSET_RE.match(subset_dir.name)
            if not sm:
                continue
            subset_idx = int(sm.group(1))

            for expr_dir in sorted(subset_dir.iterdir()):
                if not expr_dir.is_dir():
                    continue
                expression = expr_dir.name

                # Count PNG frames
                png_files = sorted(expr_dir.glob("*.png"))
                if len(png_files) < 3:
                    # Need at least 3 frames (sampled to 2+ for flow)
                    continue

                video_id = f"{participant_id}_{expression}_s{subset_idx:02d}"
                sequences.append(MPISequenceInfo(
                    participant_id=participant_id,
                    subset_idx=subset_idx,
                    expression=expression,
                    frames_dir=str(expr_dir),
                    num_frames=len(png_files),
                    video_id=video_id,
                ))

    return sequences


# ═════════════════════════════════════════════════════════════════════════════
# 2.  FRAME LOADING + TEMPORAL SAMPLING
# ═════════════════════════════════════════════════════════════════════════════

def load_and_sample_frames(
    frames_dir: str,
    target_length: int = 33,
    spatial_size: Tuple[int, int] = (224, 224),
) -> Optional[np.ndarray]:
    """
    Load PNG frames from a directory, uniformly sample ``target_length``
    frames, resize to ``spatial_size``, and return as grayscale float32.

    This implements the same TIM (Temporal Interpolation Model) strategy
    as ``Stage1_DataPipeline/step2_extraction/temporal_interpolator.py``
    but reads individual PNGs instead of .avi videos.

    Parameters
    ----------
    frames_dir : str
        Absolute path to directory containing ``*.png`` frames.
    target_length : int
        Number of frames to sample. 33 → 32 flow pairs.
    spatial_size : tuple
        Target ``(H, W)`` for resized frames.

    Returns
    -------
    np.ndarray or None
        Float32 array of shape ``(target_length, H, W)`` with values
        in [0, 255], or None if loading fails.
    """
    H, W = spatial_size
    expr_path = Path(frames_dir)

    # Sort numerically by the frame number suffix
    png_files = sorted(expr_path.glob("*.png"))
    if len(png_files) < 2:
        return None

    total_frames = len(png_files)

    # Uniform temporal sampling: pick target_length indices from [0, total-1]
    indices = np.linspace(0, total_frames - 1, num=target_length)
    indices = np.round(indices).astype(np.int64)
    indices = np.clip(indices, 0, total_frames - 1)

    frames = np.zeros((target_length, H, W), dtype=np.float32)

    for i, idx in enumerate(indices):
        img = cv2.imread(str(png_files[idx]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        resized = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        frames[i] = resized.astype(np.float32)

    return frames


# ═════════════════════════════════════════════════════════════════════════════
# 3.  OPTICAL FLOW + STRAIN EXTRACTION (self-contained, same math as Stage 1)
# ═════════════════════════════════════════════════════════════════════════════

def compute_flow_strain_tensor(
    frames: np.ndarray,
    farneback_params: Optional[dict] = None,
) -> Optional[np.ndarray]:
    """
    Compute optical flow (u, v) and optical strain from a frame sequence.

    This is functionally identical to
    ``Stage1_DataPipeline/step2_extraction/flow_strain_extractor.py``
    but reimplemented inline for self-containment.

    Parameters
    ----------
    frames : np.ndarray
        Grayscale frame sequence of shape ``(L, H, W)`` with dtype float32,
        values in [0, 255].
    farneback_params : dict, optional
        Keyword arguments for ``cv2.calcOpticalFlowFarneback``.

    Returns
    -------
    np.ndarray or None
        Float32 tensor of shape ``(3, T, H, W)`` where T = L - 1:
            - Channel 0: normalized horizontal flow (u)
            - Channel 1: normalized vertical flow (v)
            - Channel 2: normalized optical strain (os)
        All values in [0, 1] after Min-Max normalization.
    """
    if frames.ndim != 3 or frames.shape[0] < 2:
        return None

    fb = farneback_params or {
        "pyr_scale": 0.5,
        "levels": 3,
        "winsize": 15,
        "iterations": 3,
        "poly_n": 5,
        "poly_sigma": 1.2,
        "flags": 0,
    }

    L, H, W = frames.shape
    T = L - 1

    frames_uint8 = frames.astype(np.uint8)
    u_seq = np.zeros((T, H, W), dtype=np.float32)
    v_seq = np.zeros((T, H, W), dtype=np.float32)

    for t in range(T):
        flow = cv2.calcOpticalFlowFarneback(
            prev=frames_uint8[t],
            next=frames_uint8[t + 1],
            flow=None,
            **fb,
        )
        u_seq[t] = flow[:, :, 0]
        v_seq[t] = flow[:, :, 1]

    # Optical strain computation (vectorized)
    _, du_dy, du_dx = np.gradient(u_seq)
    _, dv_dy, dv_dx = np.gradient(v_seq)

    eps_xx = du_dx
    eps_yy = dv_dy
    eps_xy = 0.5 * (du_dy + dv_dx)
    os_seq = np.sqrt(eps_xx ** 2 + eps_yy ** 2 + eps_xy ** 2)

    del du_dy, du_dx, dv_dy, dv_dx, eps_xx, eps_yy, eps_xy, frames_uint8

    # Min-Max normalization per channel
    def _minmax(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        a_min, a_max = arr.min(), arr.max()
        return ((arr - a_min) / (a_max - a_min + eps)).astype(np.float32)

    tensor = np.stack([_minmax(u_seq), _minmax(v_seq), _minmax(os_seq)], axis=0)
    return np.ascontiguousarray(tensor)


# ═════════════════════════════════════════════════════════════════════════════
# 4.  WORKER FUNCTION (for multiprocessing)
# ═════════════════════════════════════════════════════════════════════════════

def _process_sequence_worker(
    seq_info: Dict[str, Any],
    tensors_dir: str,
    target_length: int,
    spatial_size: Tuple[int, int],
) -> Dict[str, Any]:
    """
    Worker function for ProcessPoolExecutor.

    Loads frames, computes flow/strain, saves tensor.

    Parameters
    ----------
    seq_info : dict
        Serialisable representation of ``MPISequenceInfo``.
    tensors_dir : str
        Output directory for ``.npy`` tensors.
    target_length : int
        Number of frames to sample (33 for T=32).
    spatial_size : tuple
        Target (H, W).

    Returns
    -------
    dict
        Status report with keys: status, video_id, [error], [save_path].
    """
    cv2.setNumThreads(0)  # Prevent OpenCV threading in child process

    video_id = seq_info["video_id"]
    frames_dir = seq_info["frames_dir"]

    try:
        # Load and temporally sample frames
        frames = load_and_sample_frames(
            frames_dir, target_length=target_length, spatial_size=spatial_size
        )
        if frames is None:
            return {"status": "error", "video_id": video_id,
                    "error": f"Failed to load frames from {frames_dir}"}

        # Compute optical flow + strain tensor
        tensor = compute_flow_strain_tensor(frames)
        del frames
        if tensor is None:
            return {"status": "error", "video_id": video_id,
                    "error": "Flow/strain extraction returned None"}

        # Save as .npy
        out_path = Path(tensors_dir) / f"MPI_{video_id}.npy"
        np.save(str(out_path), tensor)
        del tensor
        gc.collect()

        return {"status": "success", "video_id": video_id,
                "save_path": str(out_path)}

    except Exception as exc:
        gc.collect()
        return {"status": "error", "video_id": video_id,
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}


# ═════════════════════════════════════════════════════════════════════════════
# 5.  CSV GENERATION (compatible with master_thesis_labels.csv schema)
# ═════════════════════════════════════════════════════════════════════════════

def generate_mpi_csv(
    sequences: List[MPISequenceInfo],
    csv_path: Path,
    emotion_map: Optional[Dict[str, str]] = None,
) -> None:
    """
    Write a CSV file with the same schema as ``master_thesis_labels.csv``.

    Parameters
    ----------
    sequences : list of MPISequenceInfo
        All discovered sequences.
    csv_path : Path
        Output CSV path.
    emotion_map : dict or None
        Maps MPI expression folder names → unified emotion labels
        (e.g., ``"happy_laughing" → "Positive"``). If None, uses the
        raw expression name as-is.
    """
    # Import label maps (deferred to avoid circular imports at module level)
    if emotion_map is None:
        try:
            from ablation_config import MPI_EMOTION_MAP_3CLASS
            emotion_map = MPI_EMOTION_MAP_3CLASS
        except ImportError:
            emotion_map = {}

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "Dataset", "Subject_ID", "Video_ID",
        "Onset_Frame", "Apex_Frame", "Offset_Frame",
        "Raw_Emotion", "Unified_Emotion", "Action_Units",
        "Expression_Type", "Sequence_Length", "FPS",
        "Frames_Directory", "Frames_Exist", "OF_Processed",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()

        for seq in sequences:
            # Map raw expression → unified emotion
            raw_emotion = seq.expression
            unified = emotion_map.get(raw_emotion, raw_emotion)

            writer.writerow({
                "Dataset": "MPI",
                "Subject_ID": seq.participant_id,
                "Video_ID": seq.video_id,
                "Onset_Frame": 1,
                "Apex_Frame": seq.num_frames // 2,
                "Offset_Frame": seq.num_frames,
                "Raw_Emotion": raw_emotion,
                "Unified_Emotion": unified,
                "Action_Units": "N/A",
                "Expression_Type": "expression",
                "Sequence_Length": seq.num_frames,
                "FPS": 25,  # MPI standard capture rate
                "Frames_Directory": seq.frames_dir,
                "Frames_Exist": True,
                "OF_Processed": False,
            })


# ═════════════════════════════════════════════════════════════════════════════
# 6.  PIPELINE ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

class MPIDataPipeline:
    """
    End-to-end MPI dataset → tensor pipeline.

    Scans the MPI directory, generates CSV, and computes optical
    flow/strain tensors in parallel.

    Parameters
    ----------
    mpi_root : Path
        Root directory of the MPI dataset.
    output_dir : Path
        Directory for output tensors.
    csv_path : Path
        Output CSV path.
    target_length : int
        Temporal sampling length (33 → 32 flow pairs).
    spatial_size : tuple
        Frame resize target (H, W).
    max_workers : int
        Number of parallel processes.
    emotion_map : dict or None
        Expression → unified emotion mapping for the CSV.
    """

    def __init__(
        self,
        mpi_root: Path,
        output_dir: Path,
        csv_path: Path,
        target_length: int = 33,
        spatial_size: Tuple[int, int] = (224, 224),
        max_workers: int = 8,
        emotion_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.mpi_root = Path(mpi_root)
        self.output_dir = Path(output_dir)
        self.csv_path = Path(csv_path)
        self.target_length = target_length
        self.spatial_size = spatial_size
        self.max_workers = max_workers
        self.emotion_map = emotion_map

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._log = self._make_logger()

    @staticmethod
    def _make_logger() -> logging.Logger:
        logger = logging.getLogger("MPIDataPipeline")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            fmt = logging.Formatter(
                "%(asctime)s | %(name)s | %(message)s", "%H:%M:%S"
            )
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            logger.addHandler(sh)
        return logger

    def run(self, dry_run: bool = False, limit: Optional[int] = None) -> None:
        """
        Execute the full pipeline.

        Parameters
        ----------
        dry_run : bool
            If True, scan and generate CSV but skip tensor computation.
        limit : int, optional
            Limit the number of processed sequences (useful for debugging).
        """
        start = time.time()

        # ── Step 1: Scan ──
        self._log.info("=" * 78)
        self._log.info("  MPI Data Pipeline — Scanning %s", self.mpi_root)
        self._log.info("=" * 78)

        sequences = scan_mpi_dataset(self.mpi_root)
        self._log.info(
            "Discovered %d sequences across %d participants",
            len(sequences),
            len({s.participant_id for s in sequences}),
        )

        if not sequences:
            self._log.error("No sequences found. Check MPI root directory.")
            return

        if limit is not None and limit > 0:
            sequences = sequences[:limit]
            self._log.info("Limiting processing to the first %d sequences.", limit)

        # Log per-participant summary
        from collections import Counter
        participant_counts = Counter(s.participant_id for s in sequences)
        for pid, cnt in sorted(participant_counts.items()):
            self._log.info("  %s: %d sequences", pid, cnt)

        # ── Step 2: Generate CSV ──
        self._log.info("Generating CSV: %s", self.csv_path)
        generate_mpi_csv(sequences, self.csv_path, self.emotion_map)
        self._log.info("CSV written: %d rows", len(sequences))

        if dry_run:
            self._log.info("DRY RUN — skipping tensor computation.")
            return

        # ── Step 3: Filter out already-processed sequences ──
        pending = []
        skipped = 0
        for seq in sequences:
            out_path = self.output_dir / f"MPI_{seq.video_id}.npy"
            if out_path.exists():
                skipped += 1
            else:
                pending.append(seq)

        self._log.info(
            "Tensors: %d pending | %d already exist (skipped)",
            len(pending), skipped,
        )

        if not pending:
            self._log.info("All tensors already computed. Nothing to do.")
            self._update_csv_of_processed(sequences)
            return

        # ── Step 4: Compute tensors in parallel ──
        self._log.info(
            "Starting tensor extraction with %d workers...", self.max_workers
        )

        success_count = 0
        error_count = 0

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            future_map = {}
            for seq in pending:
                seq_dict = {
                    "video_id": seq.video_id,
                    "frames_dir": seq.frames_dir,
                    "participant_id": seq.participant_id,
                    "expression": seq.expression,
                }
                fut = executor.submit(
                    _process_sequence_worker,
                    seq_dict,
                    str(self.output_dir),
                    self.target_length,
                    self.spatial_size,
                )
                future_map[fut] = seq.video_id

            completed = 0
            total = len(pending)
            for fut in concurrent.futures.as_completed(future_map):
                completed += 1
                vid = future_map[fut]
                try:
                    result = fut.result()
                    if result["status"] == "success":
                        success_count += 1
                        self._log.info(
                            "[%d/%d] ✔ %s → %s",
                            completed, total, vid, result["save_path"],
                        )
                    else:
                        error_count += 1
                        self._log.error(
                            "[%d/%d] ✖ %s: %s",
                            completed, total, vid, result["error"],
                        )
                except Exception as exc:
                    error_count += 1
                    self._log.error(
                        "[%d/%d] ✖ %s: CRITICAL %s",
                        completed, total, vid, exc,
                    )

        # ── Step 5: Update CSV with OF_Processed flags ──
        self._update_csv_of_processed(sequences)

        elapsed = time.time() - start
        self._log.info("=" * 78)
        self._log.info(
            "  Pipeline complete: %d success | %d errors | %.1f seconds",
            success_count, error_count, elapsed,
        )
        self._log.info("  Tensors: %s", self.output_dir)
        self._log.info("  CSV:     %s", self.csv_path)
        self._log.info("=" * 78)

    def _update_csv_of_processed(
        self, sequences: List[MPISequenceInfo]
    ) -> None:
        """Re-read the CSV and update OF_Processed based on tensor existence."""
        import pandas as pd

        if not self.csv_path.exists():
            return

        df = pd.read_csv(self.csv_path, encoding="utf-8-sig")
        modified = 0
        for idx, row in df.iterrows():
            vid = row["Video_ID"]
            tensor_path = self.output_dir / f"MPI_{vid}.npy"
            new_val = tensor_path.exists()
            if row["OF_Processed"] != new_val:
                df.at[idx, "OF_Processed"] = new_val
                modified += 1

        if modified > 0:
            df.to_csv(self.csv_path, index=False, encoding="utf-8-sig")
            self._log.info("Updated OF_Processed for %d rows in CSV.", modified)


# ═════════════════════════════════════════════════════════════════════════════
# 7.  CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MPI Facial Expression Database → Tensor Pipeline"
    )
    p.add_argument(
        "--mpi_root", type=Path,
        default=PROJECT_ROOT / "DATASETS" / "MPI Datast",
        help="Root directory of the MPI dataset.",
    )
    p.add_argument(
        "--output_dir", type=Path,
        default=PROJECT_ROOT / "Processed_Data" / "mpi_tensors",
        help="Output directory for .npy tensors.",
    )
    p.add_argument(
        "--csv_path", type=Path,
        default=PROJECT_ROOT / "Processed_Data" / "mpi_labels.csv",
        help="Output CSV path.",
    )
    p.add_argument(
        "--target_length", type=int, default=33,
        help="Temporal sampling length (33 → 32 flow pairs).",
    )
    p.add_argument(
        "--spatial_h", type=int, default=224,
        help="Spatial height for frame resizing.",
    )
    p.add_argument(
        "--spatial_w", type=int, default=224,
        help="Spatial width for frame resizing.",
    )
    p.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel processes.",
    )
    p.add_argument(
        "--label_mode", choices=["3class", "full"], default="3class",
        help="Label mapping mode: '3class' maps to Positive/Negative/Surprise; "
             "'full' uses raw MPI expression names.",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Scan and generate CSV only; skip tensor computation.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Limit the number of sequences processed.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Select emotion map based on label mode
    if args.label_mode == "3class":
        # Import from ablation_config (deferred)
        try:
            sys.path.insert(0, str(_THIS_DIR))
            from ablation_config import MPI_EMOTION_MAP_3CLASS
            emotion_map = MPI_EMOTION_MAP_3CLASS
        except ImportError:
            logging.warning(
                "MPI_EMOTION_MAP_3CLASS not found in ablation_config; "
                "using raw expression names."
            )
            emotion_map = {}
    else:
        emotion_map = {}

    pipeline = MPIDataPipeline(
        mpi_root=args.mpi_root,
        output_dir=args.output_dir,
        csv_path=args.csv_path,
        target_length=args.target_length,
        spatial_size=(args.spatial_h, args.spatial_w),
        max_workers=args.workers,
        emotion_map=emotion_map,
    )
    pipeline.run(dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    main()
