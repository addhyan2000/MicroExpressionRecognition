# -*- coding: utf-8 -*-
"""Atomic checkpoint manager for the preprocessing pipeline.

The ``CheckpointManager`` persists which video IDs have been successfully
processed to a JSON file using **atomic writes** (write → temp file, then
``os.replace``).  This guarantees that a crash or kill-signal can never
leave the checkpoint in a half-written / corrupt state.

All public methods are guarded by a ``threading.Lock`` so concurrent
workers can safely call ``is_processed`` / ``mark_processed`` without
races.

Usage::

    mgr = CheckpointManager(Path("processed_state.json"))
    if not mgr.is_processed("sub01__EP02_01f"):
        # ... process video ...
        mgr.mark_processed("sub01__EP02_01f")
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Set, Union


logger = logging.getLogger("mer_preprocessing")


class CheckpointManager:
    """Thread-safe, crash-proof checkpoint backed by a JSON file.

    The internal state is a dictionary with at least a ``"processed"`` key
    whose value is a **set** of video IDs (strings) that completed
    successfully.  Additional metadata (e.g. timestamps) can be attached
    in the future without breaking backwards-compatibility.

    Attributes:
        path: Absolute path to the checkpoint JSON file.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        """Initialise the checkpoint manager.

        If *path* already exists it is loaded into memory; otherwise an
        empty state is created (the file itself is written lazily on the
        first ``mark_processed`` call).

        Args:
            path: Absolute path to the ``processed_state.json`` file.
                  Accepts both ``str`` and ``pathlib.Path``.
        """
        self.path: Path = Path(path)
        self._lock: threading.Lock = threading.Lock()
        self._state: Dict[str, Any] = self._load()
        logger.debug(
            "CheckpointManager initialised — %d videos already processed.",
            len(self._state.get("processed", [])),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_processed(self, video_id: str) -> bool:
        """Check whether *video_id* has already been processed.

        Args:
            video_id: Unique identifier for the video (e.g.
                ``"sub01__EP02_01f"``).

        Returns:
            ``True`` if the video has been processed, ``False`` otherwise.
        """
        with self._lock:
            return video_id in self._processed_set

    def mark_processed(self, video_id: str) -> None:
        """Record *video_id* as successfully processed and flush to disk.

        The write is **atomic**: data is first written to a temporary file
        in the same directory, then ``os.replace`` atomically swaps it
        into the target path.  This prevents corruption from partial
        writes.

        The entire read-modify-write cycle (set insertion + disk flush)
        is performed under a single lock acquisition.

        Args:
            video_id: Unique identifier for the video.
        """
        with self._lock:
            self._processed_set.add(video_id)
            self._flush()
            logger.debug(
                "Checkpoint updated — marked '%s' as processed.", video_id
            )

    def processed_count(self) -> int:
        """Return the number of videos that have been processed.

        Returns:
            Integer count of processed video IDs.
        """
        with self._lock:
            return len(self._processed_set)

    def reset(self) -> None:
        """Clear all checkpoint state and delete the backing file.

        This is a destructive operation intended only for development /
        debugging.
        """
        with self._lock:
            self._state = {"processed": [], "_processed_set": set()}
            if self.path.exists():
                self.path.unlink()
            logger.warning("Checkpoint state has been RESET.")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @property
    def _processed_set(self) -> Set[str]:
        """Return the mutable set of processed IDs."""
        return self._state.setdefault("_processed_set", set())

    def _load(self) -> Dict[str, Any]:
        """Deserialise the checkpoint file from disk.

        If the file exists but is corrupt **and** non-empty, it is
        renamed to ``*.json.corrupted`` so the raw bytes are preserved
        for post-mortem analysis.  An empty scaffold is returned so the
        pipeline can continue from scratch.

        Returns:
            Parsed state dictionary, or an empty scaffold if the file
            does not exist or is corrupt.
        """
        with self._lock:
            if not self.path.exists():
                return {"processed": [], "_processed_set": set()}

            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    data: Dict[str, Any] = json.load(fh)
                processed_list = data.get("processed", [])
                data["_processed_set"] = set(processed_list)
                return data
            except (json.JSONDecodeError, IOError) as exc:
                # ── CRITICAL: back up before discarding ──────────────
                if self.path.exists() and self.path.stat().st_size > 0:
                    backup_path = self.path.with_name(
                        f"{self.path.stem}_{int(time.time())}.json.corrupted"
                    )
                    os.replace(self.path, backup_path)
                    logger.warning(
                        "!!! CORRUPTED CHECKPOINT DETECTED !!!  "
                        "Corrupted checkpoint file was moved to '%s' "
                        "and a fresh state was initialized.  "
                        "Original error: %s",
                        backup_path,
                        exc,
                    )
                else:
                    logger.error(
                        "Checkpoint file '%s' is corrupt or empty — "
                        "starting fresh.  Error: %s",
                        self.path,
                        exc,
                    )
                return {"processed": [], "_processed_set": set()}

    def _flush(self) -> None:
        """Atomically persist the current state to disk.

        Strategy:
        1. Serialise to a temporary file in the **same directory**
           as the target (required for ``os.replace`` to be atomic on all
           platforms).
        2. ``os.replace`` the temp file onto the target path.

        .. note:: This method must be called while ``self._lock`` is held.
        """
        # Prepare a JSON-serialisable snapshot (sets → sorted lists).
        serialisable: Dict[str, Any] = {
            "processed": sorted(self._processed_set),
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Write to a temp file in the same dir, then atomic-swap.
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.path.parent),
            prefix=".ckpt_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_fh:
                json.dump(serialisable, tmp_fh, indent=2)
            os.chmod(tmp_path, 0o644)
            os.replace(tmp_path, str(self.path))
        except BaseException:
            # Clean up the temp file if the swap failed.
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
