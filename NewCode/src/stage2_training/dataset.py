# -*- coding: utf-8 -*-
"""
MERDataset — Professional PyTorch Dataset for Micro-Expression Recognition.
Optimized for STSTNet (Liong et al., 2019).
Features: Z-Score Normalization, Temporal Padding (T=32), and Multi-source Labeling.
"""

from __future__ import annotations
import csv
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ──────────────────────────────────────────────────────────────────────
# Configuration & Constants
# ──────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# Expected input shape from Stage 1: (Frames, Height, Width, Channels)
EXPECTED_T, EXPECTED_H, EXPECTED_W, EXPECTED_C = 32, 224, 224, 3

# Emotion → Class-ID mapping (4-class problem)
EMOTION_KEYWORDS: Dict[int, List[str]] = {
    0: ["disgust", "anger", "negative", "repression", "sadness", "fear", "contempt"],
    1: ["happy", "positive", "happiness"],
    2: ["surprise"],
    3: ["others"],
}

class MERDataset(Dataset):
    """
    PyTorch Dataset that loads .npy micro-expression feature tensors.
    
    Permutes from (T, H, W, C) to (C, T, H, W) for 3D-CNN compatibility.
    """

    def __init__(
        self,
        data_dir: str | Path,
        label_csv: Optional[str | Path] = None,
        transform: Any = None,
        mmap: bool = True,
        preload: bool = False,
        strict_labeling: bool = False,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        self.data_dir = Path(data_dir).resolve()
        self.transform = transform
        self.mmap = mmap
        self.preload = preload
        self.strict_labeling = strict_labeling
        self.normalize = normalize

        # 1. Load external label map (if provided)
        self._csv_labels: Dict[str, str] = {}
        if label_csv is not None:
            self._csv_labels = self._load_label_csv(Path(label_csv))
            logger.info(f"Loaded {len(self._csv_labels)} labels from CSV.")

        # 2. Discover all .npy files and resolve their labels
        self.samples: List[Tuple[Path, int]] = self._discover_samples()

        # 3. Handle preloading into RAM
        self.cache: List[np.ndarray] = []
        if self.preload:
            self._preload_data()

        # 4. Compute global standardization stats
        self.mean = torch.zeros((3, 1, 1, 1))
        self.std = torch.ones((3, 1, 1, 1))
        if self.normalize:
            self.mean, self.std = self._compute_stats()

        logger.info(f"MERDataset initialized — {len(self.samples)} samples from '{self.data_dir.name}'")
        self._log_class_distribution()

    # ------------------------------------------------------------------ #
    #  Dataset Protocol
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        filepath, label = self.samples[idx]

        # 1. Load Data
        if self.preload:
            raw = self.cache[idx]
        else:
            # Load with mmap for lazy I/O, cast to float32 to avoid double-memory usage
            raw = np.load(str(filepath), mmap_mode='r' if self.mmap else None)
        
        # 2. Convert to Tensor & Permute: (T, H, W, C) → (C, T, H, W)
        # We use .copy() if mmap to allow safe manipulation in Torch
        tensor = torch.from_numpy(raw.copy() if self.mmap and not self.preload else raw).float()
        tensor = tensor.permute(3, 0, 1, 2)

        # 3. Temporal Resizing Logic (Standardize to 32 frames)
        c, t, h, w = tensor.shape
        if t < EXPECTED_T:
            padding_size = EXPECTED_T - t
            last_frame = tensor[:, -1:, :, :] 
            padding = last_frame.repeat(1, padding_size, 1, 1)
            tensor = torch.cat([tensor, padding], dim=1)
        elif t > EXPECTED_T:
            start = (t - EXPECTED_T) // 2
            tensor = tensor[:, start : start + EXPECTED_T, :, :]

        # 4. Standardization (Z-Score)
        if self.normalize:
            tensor = (tensor - self.mean) / (self.std + 1e-7)

        # 5. Transforms
        if self.transform:
            tensor = self.transform(tensor)

        return tensor, torch.tensor(label, dtype=torch.long)

    # ------------------------------------------------------------------ #
    #  Utility Methods
    # ------------------------------------------------------------------ #

    def get_class_weights(self) -> torch.Tensor:
        """Calculate inverse frequency weights for CrossEntropyLoss."""
        counts = np.zeros(4)
        for _, label in self.samples:
            counts[label] += 1
        
        counts[counts == 0] = 1 # Avoid division by zero
        weights = len(self.samples) / (4 * counts)
        return torch.from_numpy(weights).float()

    def dry_run(self, batch_size: int = 4) -> None:
        """Attempts to load a single batch as a sanity check."""
        logger.info(f"Initiating Dry Run (batch_size={batch_size})...")
        try:
            loader = DataLoader(self, batch_size=batch_size, shuffle=True)
            x, y = next(iter(loader))
            logger.info(f"✅ Dry Run Successful! Input shape: {x.shape}, Label shape: {y.shape}")
        except Exception as e:
            logger.error(f"❌ Dry Run Failed: {e}")
            raise e

    # ------------------------------------------------------------------ #
    #  Internal Helpers
    # ------------------------------------------------------------------ #

    def _compute_stats(self, max_samples: int = 200) -> Tuple[torch.Tensor, torch.Tensor]:
        num_samples = min(len(self.samples), max_samples)
        logger.info(f"Computing Z-Score stats using {num_samples} samples...")
        
        indices = np.random.choice(len(self.samples), size=num_samples, replace=False)
        channel_sum = np.zeros(EXPECTED_C, dtype=np.float64)
        channel_sq_sum = np.zeros(EXPECTED_C, dtype=np.float64)
        total_pixels = 0
        
        for idx in indices:
            fpath, _ = self.samples[idx]
            raw = np.load(fpath).astype(np.float32)
            pixels = raw.reshape(-1, EXPECTED_C)
            channel_sum += pixels.sum(axis=0)
            channel_sq_sum += (pixels ** 2).sum(axis=0)
            total_pixels += pixels.shape[0]
            
        mean = channel_sum / total_pixels
        std = np.sqrt(np.maximum((channel_sq_sum / total_pixels) - (mean ** 2), 1e-8))
        
        logger.info(f"Stats Ready -> Mean: {mean.tolist()}, Std: {std.tolist()}")
        return (torch.tensor(mean).float().view(3, 1, 1, 1), 
                torch.tensor(std).float().view(3, 1, 1, 1))

    def _preload_data(self) -> None:
        logger.info("Preloading dataset into RAM...")
        for fpath, _ in self.samples:
            raw = np.load(str(fpath)).astype(np.float32)
            self.cache.append(raw)

    def _discover_samples(self) -> List[Tuple[Path, int]]:
        samples = []
        npy_files = sorted(self.data_dir.rglob("*.npy"))
        for fpath in npy_files:
            label = self._csv_labels.get(fpath.name)
            if not label:
                label_idx = self._extract_label_from_filename(fpath.stem.lower())
            else:
                label_idx = self._emotion_to_label(label)

            if label_idx is not None:
                samples.append((fpath, label_idx))
            elif not self.strict_labeling:
                samples.append((fpath, 3)) # Fallback to Others
        return samples

    @staticmethod
    def _extract_label_from_filename(fname: str) -> Optional[int]:
        for cid, keywords in EMOTION_KEYWORDS.items():
            for kw in keywords:
                if re.search(rf"(?:^|[^a-z]){re.escape(kw)}(?:[^a-z]|$)", fname):
                    return cid
        return None

    @staticmethod
    def _emotion_to_label(emotion: str) -> int:
        emo = emotion.lower().strip()
        for cid, keywords in EMOTION_KEYWORDS.items():
            if emo in keywords: return cid
        return 3

    def _load_label_csv(self, path: Path) -> Dict[str, str]:
        if not path.exists(): return {}
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f, skipinitialspace=True)
            return {row[0]: row[1] for row in reader if row[0].lower() != "filename"}

    def _log_class_distribution(self) -> None:
        dist = {0:0, 1:0, 2:0, 3:0}
        for _, l in self.samples: dist[l] += 1
        logger.info(f"Class distribution: Neg:{dist[0]} | Pos:{dist[1]} | Sur:{dist[2]} | Oth:{dist[3]}")