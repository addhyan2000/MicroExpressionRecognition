"""
tensor_pipeline_manager.py — Multiprocessing Orchestrator
=========================================================

Reads the master CSV and coordinates parallel extraction of
Farnebäck optical flow and optical strain tensors.

Design Constraints Satisfied:
    - Multiprocessing via ProcessPoolExecutor (max_workers=12).
    - CV2 threading disabled in the worker via FlowStrainExtractor.
    - Checkpoints correctly resume aborted runs.
    - Resulting (3, 32, 224, 224) tensors saved natively.

Author: Addhyan
Stage : 1 — Data Pipeline / Step 2
"""

from __future__ import annotations

import concurrent.futures
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from config import OUTPUT_CFG, SCHEMA
from step2_extraction.flow_strain_extractor import FlowStrainExtractor
from step2_extraction.temporal_interpolator import TemporalInterpolator
from utils.checkpoint_manager import CheckpointManager
from utils.logger import get_logger

# Pre-instantiate logger for process space (optional, we use local mostly)
_log = get_logger(__name__)


def process_video_worker(row: Dict[str, Any], tensors_dir_str: str) -> Dict[str, Any]:
    """
    Worker function executed in isolated child processes.

    Parameters
    ----------
    row : dict
        A row representing a video metadata entry from pandas.
    tensors_dir_str : str
        Absolute path to the outputs directory.

    Returns
    -------
    dict
        Execution status summary.
    """
    dataset = row[SCHEMA.dataset]
    video_id = row[SCHEMA.video_id]
    onset = int(row[SCHEMA.onset_frame])
    offset = int(row[SCHEMA.offset_frame])
    frames_dir = row[SCHEMA.frames_dir]

    # Instantiate locally to guarantee memory/lock safety inside standard Python spawn
    # The FlowStrainExtractor class disables OpenCV threading upon init.
    ti = TemporalInterpolator(target_length=33, spatial_size=(224, 224))
    fse = FlowStrainExtractor()

    try:
        # 1. Uniformly sample frames
        frames = ti.load_frames(
            frames_dir=frames_dir,
            onset=onset,
            offset=offset,
            dataset=dataset
        )
        if frames is None:
            return {"status": "error", "dataset": dataset, "video_id": video_id, 
                    "error": f"Failed to load frames spanning {onset} -> {offset} from {frames_dir}"}

        # 2. Extract modalities
        tensor = fse.extract(frames)
        if tensor is None:
            return {"status": "error", "dataset": dataset, "video_id": video_id, 
                    "error": "Extractor returned None"}

        # 3. Save as .npy
        out_filename = f"{dataset}_{video_id}.npy"
        out_path = Path(tensors_dir_str) / out_filename
        np.save(str(out_path), tensor)

        return {
            "status": "success",
            "dataset": dataset,
            "video_id": video_id,
            "save_path": str(out_path)
        }

    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {str(exc)}\n{traceback.format_exc()}"
        return {"status": "error", "dataset": dataset, "video_id": video_id, "error": err_msg}


class TensorPipelineManager:
    """
    Orchestrates the asynchronous processing of all valid Micro-Expression videos.

    Parameters
    ----------
    max_workers : int
        Maximum number of parallel child processes. Must be tuned exactly
        for Core i7-12650H architecture (12 threads requested by prompt constraint).
    """

    def __init__(self, max_workers: int = 12) -> None:
        self.max_workers = max_workers
        self.log = get_logger(self.__class__.__name__)

        self.master_csv_path = OUTPUT_CFG.master_csv_path

        # Tensors output directory inside Processed_Data
        self.tensors_dir = OUTPUT_CFG.processed_root / "tensors"
        self.tensors_dir.mkdir(parents=True, exist_ok=True)

        # Isolated step 2 checkpoint
        step2_ckpt_path = OUTPUT_CFG.checkpoint_dir / "step2_state.json"
        self.checkpoint = CheckpointManager(step2_ckpt_path)

    def run(self) -> None:
        """Main entry point. Identifies remaining videos and dispatches them uniformly."""
        self.log.info("Starting Temporal Interpolation & Optimal Flow Extraction.")
        
        if not self.master_csv_path.exists():
            self.log.error("Master CSV not found at %s. Please run Step 1 metadata unification.", 
                           self.master_csv_path)
            sys.exit(1)

        df = pd.read_csv(self.master_csv_path)

        # Subset exactly according to requirements: 
        # Frames_Exist == True and Sequence_Length > 2
        cond_exist = df[SCHEMA.frames_exist] == True
        cond_len = df[SCHEMA.sequence_length] > 2
        
        valid_df = df[cond_exist & cond_len]
        self.log.info("Validation complete. Found %d valid clips out of %d available in CSV.", 
                      len(valid_df), len(df))

        # Filter out completed chunks
        tasks_queue = []
        for _, row in valid_df.iterrows():
            dataset = row[SCHEMA.dataset]
            video_id = row[SCHEMA.video_id]
            block_key = f"{dataset}_{video_id}"

            if not self.checkpoint.is_completed(block_key):
                tasks_queue.append(row.to_dict())

        self.log.info("Clips queued for Optical Flow extractions (ignoring already successful runs): %d", len(tasks_queue))

        if not tasks_queue:
            self.log.info("No clips pending. Ensure the checkpoint file matches the truth or run the CSV sync step.")
            self._sync_csv_with_checkpoint()
            return

        # Start execution pool
        success_count = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_key = {}
            for row in tasks_queue:
                dataset = row[SCHEMA.dataset]
                video_id = row[SCHEMA.video_id]
                block_key = f"{dataset}_{video_id}"

                _fut = executor.submit(process_video_worker, row, str(self.tensors_dir))
                future_to_key[_fut] = block_key

            # Track progress synchronously inside the master event loop handling exceptions
            completed = 0
            total = len(tasks_queue)
            for future in concurrent.futures.as_completed(future_to_key):
                block_key = future_to_key[future]
                completed += 1

                try:
                    res = future.result()
                    if res["status"] == "success":
                        self.checkpoint.mark_completed(block_key)
                        success_count += 1
                        self.log.info("[%d/%d] ✔ SUCCESS - %s saved to %s", 
                                      completed, total, block_key, res["save_path"])
                    else:
                        self.checkpoint.mark_failed(block_key, reason=res["error"])
                        self.log.error("[%d/%d] ✖ FAILED - %s: %s", 
                                       completed, total, block_key, res["error"])

                except Exception as loop_exc:
                    self.checkpoint.mark_failed(block_key, reason=str(loop_exc))
                    self.log.error("[%d/%d] ✖ CRITICAL ERROR - %s generated an internal exception: %s", 
                                   completed, total, block_key, loop_exc)

        self.log.info("Multiprocessing phase completed. Synchronizing statuses strictly onto the core dataset CSV.")
        self._sync_csv_with_checkpoint()
        self.log.info("Batch extraction pipeline exited with %d active tensor transformations.", success_count)

    def _sync_csv_with_checkpoint(self) -> None:
        """Reads checkpoint state and globally stamps status on the master CSV."""
        df = pd.read_csv(self.master_csv_path)
        modified_count = 0

        for idx, row in df.iterrows():
            dataset = row[SCHEMA.dataset]
            video_id = row[SCHEMA.video_id]
            block_key = f"{dataset}_{video_id}"

            if self.checkpoint.is_completed(block_key):
                if not row[SCHEMA.of_processed]:
                    df.at[idx, SCHEMA.of_processed] = True
                    modified_count += 1
            else:
                if row[SCHEMA.of_processed]:
                    df.at[idx, SCHEMA.of_processed] = False
                    modified_count += 1

        if modified_count > 0:
            df.to_csv(self.master_csv_path, index=False)
            self.log.info("Reflected %d new tracking updates accurately across output data file: %s", 
                          modified_count, OUTPUT_CFG.master_csv_filename)
        else:
            self.log.info("Dataset sync complete. No divergent OF_Processed statuses flagged.")
