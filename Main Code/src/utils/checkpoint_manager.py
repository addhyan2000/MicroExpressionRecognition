"""Atomic Checkpoint Manager for Fault-Tolerant Training.

Implements a "Temporary-File-then-Rename" persistence pattern using
``os.replace()`` to guarantee atomic writes.  This prevents checkpoint
corruption during sudden power failures or SIGKILL events — the file
is either fully written or not present at all.

Author  : Addhyan Pant
Project : Master's Thesis — Micro-Expression Recognition
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)

__all__ = ["CheckpointManager"]


class CheckpointManager:
    """Production-grade checkpoint manager with atomic persistence.

    Uses the Temporary-File-then-Rename pattern to guarantee that every
    saved file is either complete and valid, or does not exist at all.
    This eliminates the risk of loading a half-written checkpoint after
    an unexpected interruption (e.g., CUDA OOM, power failure, SIGKILL).

    Parameters
    ----------
    save_dir : str or Path
        Root directory for all checkpoint artifacts.
    """

    def __init__(self, save_dir: str | Path) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"CheckpointManager initialized. Root: {self.save_dir}")

    # ------------------------------------------------------------------
    # Core Atomic Write
    # ------------------------------------------------------------------

    def _atomic_save(self, obj: Any, target_path: Path, is_torch: bool = True) -> None:
        """Write ``obj`` to ``target_path`` atomically.

        Writes to a temporary file in the same directory first, then
        performs an ``os.replace()`` which is guaranteed to be atomic
        on POSIX and Windows (NTFS) file-systems.

        Parameters
        ----------
        obj : Any
            The object to persist (torch state-dict or JSON-serialisable).
        target_path : Path
            Final destination path.
        is_torch : bool
            If True, use ``torch.save``; otherwise, ``json.dump``.
        """
        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Create temp file in the SAME directory to guarantee same filesystem
        fd, tmp_path = tempfile.mkstemp(
            dir=target_path.parent,
            prefix=f".{target_path.stem}_tmp_",
            suffix=target_path.suffix,
        )
        try:
            if is_torch:
                torch.save(obj, tmp_path)
            else:
                with os.fdopen(fd, "w") as f:
                    json.dump(obj, f, indent=4)
                fd = None  # fd is now owned by the context manager

            # Atomic rename — either the full file appears or the old one remains
            os.replace(tmp_path, target_path)
            logger.debug(f"Atomic save completed: {target_path}")
        except Exception:
            # Clean up the temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        finally:
            # Close the fd if it was never passed to os.fdopen
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Model Checkpoints
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        filename: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Atomically save a model/optimizer state-dict.

        Parameters
        ----------
        state_dict : dict
            The ``model.state_dict()`` or combined checkpoint dict.
        filename : str
            Target filename (e.g., ``"best_model_fold_3.pth"``).
        metadata : dict, optional
            Extra metadata (epoch, metrics) saved alongside the state.

        Returns
        -------
        Path
            Absolute path to the written checkpoint.
        """
        payload = {"state_dict": state_dict}
        if metadata is not None:
            payload["metadata"] = metadata

        target = self.save_dir / filename
        self._atomic_save(payload, target, is_torch=True)
        logger.info(f"Checkpoint saved atomically: {target}")
        return target

    def load_checkpoint(
        self, filename: str, map_location: Any = "cpu"
    ) -> Optional[Dict[str, Any]]:
        """Load a previously saved checkpoint.

        Returns
        -------
        dict or None
            The checkpoint dict, or ``None`` if the file does not exist.
        """
        target = self.save_dir / filename
        if not target.exists():
            logger.warning(f"Checkpoint not found: {target}")
            return None
        data = torch.load(target, map_location=map_location)
        logger.info(f"Checkpoint loaded: {target}")
        return data

    # ------------------------------------------------------------------
    # Binary Prediction Persistence (Fold Resumption)
    # ------------------------------------------------------------------

    def save_predictions(
        self,
        preds: Any,
        labels: Any,
        subjects: Any,
        variant: str,
        fold_idx: int,
    ) -> Path:
        """Atomically save raw predictions for fold-level resumption.

        Parameters
        ----------
        preds, labels : array-like
            Predictions and ground-truth, converted to numpy for storage.
        subjects : list
            Subject identifiers aligned with predictions.
        variant : str
            Ablation variant name.
        fold_idx : int
            LOSO fold index.

        Returns
        -------
        Path
            Path to the saved predictions file.
        """
        payload = {
            "preds": preds.cpu().numpy() if torch.is_tensor(preds) else preds,
            "labels": labels.cpu().numpy() if torch.is_tensor(labels) else labels,
            "subjects": subjects,
        }
        filename = f"{variant}_fold_{fold_idx}_raw_preds.pth"
        target = self.save_dir / filename
        self._atomic_save(payload, target, is_torch=True)
        logger.info(f"Predictions saved atomically: {target}")
        return target

    def load_predictions(
        self, variant: str, fold_idx: int, map_location: Any = "cpu"
    ) -> Optional[Dict[str, Any]]:
        """Load saved predictions for fold resumption.

        Returns
        -------
        dict or None
            Dict with keys ``preds``, ``labels``, ``subjects`` or ``None``.
        """
        filename = f"{variant}_fold_{fold_idx}_raw_preds.pth"
        target = self.save_dir / filename
        if not target.exists():
            return None
        data = torch.load(target, map_location=map_location)
        logger.info(f"Predictions loaded for resumption: {target}")
        return data

    def predictions_exist(self, variant: str, fold_idx: int) -> bool:
        """Check whether saved predictions exist for a given fold."""
        filename = f"{variant}_fold_{fold_idx}_raw_preds.pth"
        return (self.save_dir / filename).exists()

    # ------------------------------------------------------------------
    # JSON Progress Tracking
    # ------------------------------------------------------------------

    def save_json(self, data: Dict[str, Any], filename: str) -> Path:
        """Atomically save a JSON document (e.g., pipeline progress)."""
        target = self.save_dir / filename
        self._atomic_save(data, target, is_torch=False)
        return target

    def load_json(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load a JSON document."""
        target = self.save_dir / filename
        if not target.exists():
            return None
        with open(target, "r") as f:
            return json.load(f)
