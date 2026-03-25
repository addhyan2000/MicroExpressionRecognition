# -*- coding: utf-8 -*-
import sys
from pathlib import Path

# --- PROJECT ROOT INITIALIZATION ---
_current_file = Path(__file__).resolve()
_project_root = _current_file.parents[2] 
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import os
import time
import traceback
import logging
import concurrent.futures
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

# Local Imports
from src.data_pipeline.config import PipelineConfig
from src.data_pipeline.logger import get_pipeline_logger
from src.data_pipeline.checkpoint import CheckpointManager
from src.data_pipeline.processors import (
    EVMProcessor,
    FlowExtractor,
    TemporalInterpolator,
)
from src.data_pipeline.metadata_parser import CASME2_MetadataParser

# ======================================================================
# Logic Core: Loading Engines
# ======================================================================

def load_frames_from_video(video_path: Path, config: PipelineConfig, onset: int, offset: int) -> NDArray[np.float32]:
    """
    Decodes specific frame ranges from an AVI file with safety fallbacks.
    Used primarily for CAS(ME)^2 rawvideo.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    # --- SAFETY CHECK: Range Validation ---
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # If Excel asks for frames beyond the video's actual length, take the whole video
    if onset > total_frames or offset > total_frames:
        onset = 1
        offset = total_frames

    frames = []
    # Seek to onset (Excel is 1-based, OpenCV is 0-based)
    cap.set(cv2.CAP_PROP_POS_FRAMES, onset - 1)
    
    num_to_read = max(1, offset - onset + 1)
    for _ in range(num_to_read):
        ret, frame = cap.read()
        if not ret:
            break
        
        img = cv2.resize(frame, (config.target_width, config.target_height))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img_rgb.astype(np.float32) / 255.0)
    
    cap.release()
    
    # FALLBACK: If specific range failed, try to load the entire video
    if len(frames) < 2:
        if onset == 1: # We already tried the whole video and it failed
            raise ValueError(f"Video {video_path.name} is corrupted or too short.")
        return load_frames_from_video(video_path, config, 1, total_frames)
        
    return np.stack(frames, axis=0).astype(np.float32)

def load_frames_from_directory(video_dir: Path, config: PipelineConfig, onset=None, offset=None) -> NDArray[np.float32]:
    """
    Loads frames from a folder of images. 
    Used for CASME II and pre-extracted datasets.
    """
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    frame_paths = []
    for ext in extensions: 
        frame_paths.extend(video_dir.glob(ext))
    
    if not frame_paths:
        raise ValueError(f"No images found in {video_dir}")
    
    frame_paths.sort()
    
    # Apply clipping if indices are provided (1-based to 0-based)
    if onset is not None and offset is not None:
        start_idx = max(0, onset - 1)
        end_idx = min(len(frame_paths), offset)
        frame_paths = frame_paths[start_idx : end_idx]
    
    if len(frame_paths) < 2:
        raise ValueError(f"Insufficient frames in {video_dir}")

    frames = []
    for p in frame_paths:
        img = cv2.imread(str(p))
        if img is None: continue
        img = cv2.resize(img, (config.target_width, config.target_height))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img_rgb.astype(np.float32) / 255.0)
        
    return np.stack(frames, axis=0).astype(np.float32)

# ======================================================================
# Logic Core: Processing & Discovery
# ======================================================================

def discover_videos(config: PipelineConfig) -> List[Dict[str, Any]]:
    videos = []
    
    if config.dataset_name == "CASME_II":
        for subject_dir in sorted(config.input_dir.iterdir()):
            if not subject_dir.is_dir(): continue
            for video_dir in sorted(subject_dir.iterdir()):
                if not video_dir.is_dir(): continue
                videos.append({
                    "video_id": f"{subject_dir.name}_{video_dir.name}",
                    "video_path": video_dir, 
                    "onset": None, "offset": None
                })
                
    elif config.dataset_name == "CAS_ME_2":
        excel_path = config.input_dir.parent / "CAS(ME)^2code_final.xlsx"
        parser = CASME2_MetadataParser(excel_path=excel_path, dataset_root=config.input_dir)
        clips = parser.get_micro_clips()
        
        for clip in clips:
            videos.append({
                "video_id": clip["video_id"], 
                "video_path": clip["path"], 
                "onset": clip["onset"], 
                "offset": clip["offset"]
            })
            
    return videos

def process_single_video(video_id: str, video_path: Path, config: PipelineConfig, onset=None, offset=None) -> bool:
    logger = logging.getLogger("mer_preprocessing")
    try:
        interpolator = TemporalInterpolator(target_frames=config.target_frames)
        evm = EVMProcessor(alpha=config.evm_alpha, low_omega=config.evm_low_omega, 
                           high_omega=config.evm_high_omega, pyramid_levels=config.evm_pyramid_levels)
        flow_extractor = FlowExtractor()

        # STEP 1: LOAD (Detect if File (.avi) or Folder)
        if video_path.is_file():
            raw = load_frames_from_video(video_path, config, onset, offset)
        else:
            raw = load_frames_from_directory(video_path, config, onset, offset)

        # STEP 2: INTERPOLATE (Standardize to 32 frames)
        interp = interpolator.process(raw)
        
        # STEP 3: MAGNIFY (EVM)
        magnified = evm.process(interp, fps=config.target_fps)
        
        # STEP 4: EXTRACT (Flow X, Flow Y, Strain)
        flow_data = flow_extractor.process(magnified)

        # STEP 5: SAVE AS TENSOR
        out_path = config.output_dir / f"{video_id}.npy"
        np.save(str(out_path), flow_data)
        return True
    except Exception:
        logger.error(f"[{video_id}] Failed: {traceback.format_exc()}")
        return False

# ======================================================================
# Orchestration
# ======================================================================

def run_dataset_pipeline(params: dict):
    config = PipelineConfig.from_dict({**params, "num_workers": os.cpu_count() or 4})
    config.setup_filesystem()
    logger = get_pipeline_logger(log_file=config.log_file)
    ckpt = CheckpointManager(config.checkpoint_file)

    video_metadata_list = discover_videos(config)
    to_process = [v for v in video_metadata_list if not ckpt.is_processed(v["video_id"])]
    
    logger.info(f"\n{'='*60}\nRUNNING: {config.dataset_name}\n{'='*60}")
    if not to_process:
        logger.info(f"No remaining videos for {config.dataset_name}. Skipping.")
        return

    pbar = tqdm(total=len(to_process), desc=f"{config.dataset_name}", unit="vid")
    with concurrent.futures.ProcessPoolExecutor(max_workers=config.num_workers) as executor:
        futures = {
            executor.submit(process_single_video, v["video_id"], v["video_path"], config, v["onset"], v["offset"]): v["video_id"] 
            for v in to_process
        }
        for future in concurrent.futures.as_completed(futures):
            if future.result(): 
                ckpt.mark_processed(futures[future])
            pbar.update(1)
    pbar.close()

def main():
    jobs = {
        "1": {
            "dataset_name": "CASME_II",
            "input_dir": r"C:/Users/Addhyan/Desktop/Thesis/Thesis2/THESIS WORK new/DATASETS/CASME II/CASME2-RAW",
            "output_dir": r"C:/Users/Addhyan/Desktop/Thesis/Thesis2/THESIS WORK new/NewCode/outputs/CASME_II",
            "target_fps": 30, "target_frames": 32,
        },
        "2": {
            "dataset_name": "CAS_ME_2",
            "input_dir": r"C:/Users/Addhyan/Desktop/Thesis/Thesis2/THESIS WORK new/DATASETS/CAS(ME)2/rawvideo", 
            "output_dir": r"C:/Users/Addhyan/Desktop/Thesis/Thesis2/THESIS WORK new/NewCode/outputs/CAS_ME_2",
            "target_fps": 30, "target_frames": 32, "evm_alpha": 15.0
        }
    }

    print("\n" + "="*50)
    print(" MER PREPROCESSING PIPELINE - DATASET SELECTOR ")
    print("="*50)
    print("[1] Process CASME II")
    print("[2] Process CAS(ME)^2")
    print("[3] Process BOTH")
    print("[q] Quit")
    
    choice = input("\nEnter your choice (1/2/3/q): ").strip().lower()
    if choice == 'q': return
    
    selected_jobs = []
    if choice == '1': selected_jobs = [jobs["1"]]
    elif choice == '2': selected_jobs = [jobs["2"]]
    elif choice == '3': selected_jobs = [jobs["1"], jobs["2"]]
    else: return

    for job in selected_jobs:
        run_dataset_pipeline(job)
    
    print("\nAll selected tasks completed.")

if __name__ == "__main__":
    main()