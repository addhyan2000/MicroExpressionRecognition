"""
main_step1.py — Entry Point for Stage 1 / Step 1
=================================================

This script is the single entry point for the Metadata Unification
pipeline.  It performs the following:

    1.  Validates that all required raw dataset files exist.
    2.  Instantiates the ``PipelineManager``.
    3.  Calls ``PipelineManager.run()`` which handles:
        a.  CASME II metadata unification  (checkpoint-guarded)
        b.  CAS(ME)^2 metadata unification (checkpoint-guarded)
        c.  Merge & CSV export             (checkpoint-guarded)
    4.  Exits cleanly with a summary.

Usage::

    cd Stage1_DataPipeline
    python main_step1.py

    # To force a full re-run (reset all checkpoints):
    python main_step1.py --force

Author  : Addhyan
Stage   : 1 — Data Pipeline / Step 1 — Metadata Unification
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── Ensure the Stage1 package root is on sys.path ──────────────────
# This allows ``from config import …`` to work when the script is
# invoked as ``python main_step1.py`` from within Stage1_DataPipeline/.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from config import CASME_II_CFG, CASME2_SQ_CFG, OUTPUT_CFG    # noqa: E402
from pipeline_manager import PipelineManager                   # noqa: E402
from utils.logger import get_logger                             # noqa: E402


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1 / Step 1 — Unified Metadata Pipeline.  "
            "Reads CASME II and CAS(ME)^2 ground-truth labels, "
            "unifies emotion categories, calculates sequence lengths, "
            "and outputs master_thesis_labels.csv."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help=(
            "Delete existing checkpoints and intermediate files, "
            "forcing a full re-run from scratch."
        ),
    )
    return parser.parse_args()


def _validate_prerequisites(log) -> bool:
    """
    Verify that all required raw dataset files exist before we start.

    Returns
    -------
    bool
        ``True`` if all prerequisites are satisfied.
    """
    all_ok = True

    # ── CASME II Excel ──────────────────────────────────────────────
    if not CASME_II_CFG.excel_path.exists():
        log.error("MISSING: CASME II Excel at %s", CASME_II_CFG.excel_path)
        all_ok = False
    else:
        log.info("  ✓  CASME II Excel:     %s", CASME_II_CFG.excel_path)

    # ── CASME II frames root ────────────────────────────────────────
    if not CASME_II_CFG.frames_root.is_dir():
        log.error("MISSING: CASME II frames at %s", CASME_II_CFG.frames_root)
        all_ok = False
    else:
        log.info("  ✓  CASME II Frames:    %s", CASME_II_CFG.frames_root)

    # ── CAS(ME)^2 Excel ────────────────────────────────────────────
    if not CASME2_SQ_CFG.excel_path.exists():
        log.error("MISSING: CAS(ME)^2 Excel at %s", CASME2_SQ_CFG.excel_path)
        all_ok = False
    else:
        log.info("  ✓  CAS(ME)^2 Excel:    %s", CASME2_SQ_CFG.excel_path)

    # ── CAS(ME)^2 frames root ──────────────────────────────────────
    if not CASME2_SQ_CFG.frames_root.is_dir():
        log.error("MISSING: CAS(ME)^2 frames at %s", CASME2_SQ_CFG.frames_root)
        all_ok = False
    else:
        log.info("  ✓  CAS(ME)^2 Frames:   %s", CASME2_SQ_CFG.frames_root)

    return all_ok


def _handle_force_reset(log) -> None:
    """
    Delete checkpoint and intermediate files for a clean re-run.

    This is a destructive operation — we only delete *our own*
    generated artefacts, never the raw dataset files.
    """
    import shutil

    targets = [
        OUTPUT_CFG.checkpoint_path,
        OUTPUT_CFG.master_csv_path,
        OUTPUT_CFG.processed_root / "intermediates",
    ]

    for target in targets:
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
                log.info("  🗑  Deleted directory: %s", target)
            else:
                target.unlink()
                log.info("  🗑  Deleted file: %s", target)
        else:
            log.debug("  (not found, skipping): %s", target)

    log.info("Force reset complete.  All checkpoints cleared.")


def main() -> None:
    """Main entry point."""
    args = _parse_args()
    log = get_logger("main_step1")

    log.info("╔" + "═" * 68 + "╗")
    log.info("║  Micro-Expression Recognition — Thesis Data Pipeline            ║")
    log.info("║  Stage 1 / Step 1: Unified Metadata Generation                  ║")
    log.info("╚" + "═" * 68 + "╝")

    # ── Force reset if requested ────────────────────────────────────
    if args.force:
        log.info("--force flag detected.  Resetting all checkpoints …")
        _handle_force_reset(log)

    # ── Pre-flight checks ───────────────────────────────────────────
    log.info("Running pre-flight checks …")
    if not _validate_prerequisites(log):
        log.error(
            "Pre-flight checks FAILED.  Please ensure all raw dataset "
            "files are present before running the pipeline."
        )
        sys.exit(1)
    log.info("All pre-flight checks passed.  ✓\n")

    # ── Run the pipeline ────────────────────────────────────────────
    t_start = time.perf_counter()

    try:
        manager = PipelineManager()
        master_df = manager.run()
    except Exception:
        log.exception(
            "Pipeline terminated with an unrecoverable error.  "
            "Fix the issue and re-run — completed blocks will be skipped "
            "thanks to the checkpoint system."
        )
        sys.exit(2)

    elapsed = time.perf_counter() - t_start
    log.info("Total wall-clock time: %.2f seconds.", elapsed)

    # ── Final human-readable summary ────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("OUTPUT FILE:  %s", OUTPUT_CFG.master_csv_path)
    log.info("TOTAL ROWS:   %d", len(master_df))
    log.info("COLUMNS:      %s", master_df.columns.tolist())
    log.info("=" * 70)
    log.info("Step 1 complete.  Ready for Step 2 (Optical Flow extraction).")


if __name__ == "__main__":
    main()
