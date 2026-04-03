"""
main_step2.py — Entry Point for Stage 1 / Step 2
================================================

Orchestrates the conversion of cropped micro-expression frames into
homogenized multi-modal tensor cubes (shape: 3x32x224x224).
This script directly kicks off the Multiprocessing Execution pool
via `tensor_pipeline_manager`.

Usage:
    python main_step2.py

Author : Addhyan
Stage  : 1 — Data Pipeline / Step 2
"""

import sys

from step2_extraction.tensor_pipeline_manager import TensorPipelineManager
from utils.logger import get_logger

if __name__ == "__main__":
    log = get_logger("main_step2")
    log.info("=" * 60)
    log.info("STAGE 1 / STEP 2 : Tensor Interpolation & Modality Synthesis")
    log.info("=" * 60)

    try:
        # Multiprocessing required on host hardware. Max processes: 12 cores.
        pipeline_mgr = TensorPipelineManager(max_workers=12)
        pipeline_mgr.run()
        log.info("Pipeline Execution Reached End-Of-Life Sub-Routine Successfully.")
    except KeyboardInterrupt:
        log.warning("Received KeyboardInterrupt - Pipeline terminated by administrative user interjection.")
        sys.exit(130)
    except Exception as e:
        log.critical("Uncaught architectural level exception escaped to root: %s", e, exc_info=True)
        sys.exit(1)
