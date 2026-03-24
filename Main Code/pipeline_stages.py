"""Pipeline Stage Orchestrator with Fault-Tolerant Resumption.

Tracks progress across four sequential stages (Preprocess → Pre-train →
Evaluate → Ablate) using a ``pipeline_progress.json`` file.  On restart,
completed stages are automatically skipped.

Author  : Addhyan Pant
Project : Master's Thesis — Micro-Expression Recognition
"""

import logging
from pathlib import Path
from typing import Callable, Dict, Optional

from src.utils.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)

__all__ = ["PipelineOrchestrator"]

# Canonical stage ordering
PIPELINE_STAGES = ("preprocess", "pretrain", "evaluate", "ablate")


class PipelineOrchestrator:
    """Four-stage pipeline controller with JSON-based resumption.

    Persists completion status in ``pipeline_progress.json`` inside the
    output directory.  Each stage transitions through three states:
    ``pending`` → ``running`` → ``completed``.

    Parameters
    ----------
    output_dir : str or Path
        Root output directory (also checkpoint root).
    """

    PROGRESS_FILE = "pipeline_progress.json"

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.ckpt = CheckpointManager(self.output_dir)
        self._progress = self._load_progress()

    # ------------------------------------------------------------------
    # Progress persistence
    # ------------------------------------------------------------------

    def _load_progress(self) -> Dict[str, str]:
        """Load or initialise the progress dictionary."""
        data = self.ckpt.load_json(self.PROGRESS_FILE)
        if data is not None:
            logger.info(f"Pipeline progress loaded: {data}")
            return data
        # Bootstrap default progress
        return {stage: "pending" for stage in PIPELINE_STAGES}

    def _save_progress(self) -> None:
        """Atomically persist the current progress state."""
        self.ckpt.save_json(self._progress, self.PROGRESS_FILE)

    # ------------------------------------------------------------------
    # Stage execution
    # ------------------------------------------------------------------

    def is_stage_completed(self, stage: str) -> bool:
        """Check whether *stage* has already been completed."""
        return self._progress.get(stage) == "completed"

    def mark_stage_running(self, stage: str) -> None:
        """Mark *stage* as currently in progress."""
        self._progress[stage] = "running"
        self._save_progress()

    def mark_stage_completed(self, stage: str) -> None:
        """Mark *stage* as successfully completed."""
        self._progress[stage] = "completed"
        self._save_progress()
        logger.info(f"Pipeline stage '{stage}' marked completed.")

    def run_stage(
        self,
        stage: str,
        handler: Callable[[], None],
        *,
        force: bool = False,
    ) -> None:
        """Execute *handler* for *stage* if not already completed.

        Parameters
        ----------
        stage : str
            One of ``PIPELINE_STAGES``.
        handler : callable
            Zero-argument function implementing the stage logic.
        force : bool
            If True, re-run even if already completed.
        """
        if stage not in PIPELINE_STAGES:
            raise ValueError(f"Unknown stage '{stage}'. Must be one of {PIPELINE_STAGES}")

        if not force and self.is_stage_completed(stage):
            logger.info(f"Stage '{stage}' already completed — skipping.")
            return

        logger.info(f"Starting pipeline stage: {stage}")
        self.mark_stage_running(stage)

        try:
            handler()
            self.mark_stage_completed(stage)
        except Exception as e:
            logger.error(f"Stage '{stage}' failed: {e}")
            self._progress[stage] = "failed"
            self._save_progress()
            raise

    def run_all(
        self,
        handlers: Dict[str, Callable[[], None]],
        *,
        force: bool = False,
    ) -> None:
        """Execute all pipeline stages in canonical order.

        Parameters
        ----------
        handlers : dict
            Mapping of stage name → handler callable.
        force : bool
            If True, re-run even if completed.
        """
        for stage in PIPELINE_STAGES:
            if stage in handlers:
                self.run_stage(stage, handlers[stage], force=force)

    def get_status_summary(self) -> Dict[str, str]:
        """Return the current pipeline progress as a dict."""
        return dict(self._progress)
