"""
mer_dataset.py — PyTorch Dataset for Micro-Expression Recognition (Stage 3)
=============================================================================

Reads the Stage 1 ``master_thesis_labels.csv`` and the pre-computed
``[3, 32, 224, 224]`` NumPy tensors (horizontal flow, vertical flow,
optical strain) from ``Processed_Data/tensors/``.

Each sample returns a tuple::

    (tensor, emotion_label, subject_label)

Key Features:
    • Filters by ``Expression_Type`` (e.g. only "micro-expression" rows).
    • Maps unified emotion strings to contiguous integer labels.
    • Maps raw ``Subject_ID`` to a continuous ``[0, num_subjects - 1]`` range.
    • Only includes rows that have a corresponding ``.npy`` tensor on disk.
    • Extensive logging of dataset statistics and class distributions.

Author  : Addhyan
Stage   : 3 — Adversarial Domain Adaptation
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ── Resolve project root (Thesis3/) and wire up Stage 1 logger ──────
_STAGE3_DIR = Path(__file__).resolve().parent.parent        # Stage3_Training/
_PROJECT_ROOT = _STAGE3_DIR.parent                          # Thesis3/
sys.path.insert(0, str(_PROJECT_ROOT / "Stage1_DataPipeline"))

from utils.logger import get_logger  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Default label mappings
# ─────────────────────────────────────────────────────────────────────
DEFAULT_EMOTION_MAP: Dict[str, int] = {
    "Negative": 0,
    "Positive": 1,
    "Surprise": 2,
    "Others": 3,
}


class MERDataset(Dataset):
    """
    PyTorch Dataset for Micro-Expression Recognition with identity labels.

    This dataset is designed for adversarial training where we need both
    emotion labels (for the emotion head) and subject identity labels
    (for the identity head with GRL).

    Parameters
    ----------
    csv_path : Path
        Path to ``master_thesis_labels.csv``.
    tensor_dir : Path
        Path to the directory containing ``.npy`` tensor files.
    expression_filter : str or None
        If not ``None``, only rows where ``Expression_Type == expression_filter``
        are retained.  Common values: ``"micro-expression"``, ``"macro-expression"``.
    emotion_map : dict, optional
        Mapping from unified emotion string → integer label.
        Defaults to ``{"Negative": 0, "Positive": 1, "Surprise": 2, "Others": 3}``.
    transform : callable, optional
        Optional transform applied to the tensor after loading
        (e.g. data augmentation).
    log_dir : Path or None
        Directory for log files.  Defaults to ``Stage3_Training/logs/``.

    Attributes
    ----------
    samples : list of dict
        Each entry contains ``tensor_path``, ``emotion_label``, ``subject_label``,
        and metadata for debugging.
    num_classes : int
        Number of unique emotion classes present.
    num_subjects : int
        Number of unique subject identities present.
    emotion_map : dict
        String-to-int mapping used for emotions.
    subject_map : dict
        Raw Subject_ID → contiguous int mapping.
    """

    def __init__(
        self,
        csv_path: Path,
        tensor_dir: Path,
        expression_filter: Optional[str] = None,
        emotion_map: Optional[Dict[str, int]] = None,
        transform: bool = False,
        log_dir: Optional[Path] = None,
    ) -> None:
        super().__init__()

        # ── Logger ──────────────────────────────────────────────────
        if log_dir is None:
            log_dir = _STAGE3_DIR / "logs"
        self._log = get_logger(
            "MERDataset",
            log_dir=log_dir,
            log_filename="stage3_training.log",
        )

        self._csv_path = Path(csv_path)
        self._tensor_dir = Path(tensor_dir)
        self._expression_filter = expression_filter
        self.emotion_map = emotion_map or DEFAULT_EMOTION_MAP
        self._transform = transform

        self._log.info("=" * 70)
        self._log.info("  MERDataset Initialization")
        self._log.info("=" * 70)
        self._log.info("CSV path        : %s", self._csv_path)
        self._log.info("Tensor dir      : %s", self._tensor_dir)
        self._log.info("Expression filter: %s", self._expression_filter or "None (all)")
        self._log.info("Emotion mapping : %s", self.emotion_map)

        # ── Load and prepare data ───────────────────────────────────
        self.samples: List[dict] = []
        self.subject_map: Dict[int, int] = {}
        self._build_samples()

        # ── Derived attributes ──────────────────────────────────────
        self.num_classes = len(self.emotion_map)
        self.num_subjects = len(self.subject_map)

        self._log.info("─" * 70)
        self._log.info("  Dataset Ready: %d samples | %d classes | %d subjects",
                        len(self.samples), self.num_classes, self.num_subjects)
        self._log.info("─" * 70)

    # ─────────────────────────────────────────────────────────────────
    #  Dataset Construction
    # ─────────────────────────────────────────────────────────────────

    def _build_samples(self) -> None:
        """
        Read CSV, apply filters, map labels, and verify tensor existence.

        Steps:
            1. Load ``master_thesis_labels.csv``.
            2. Filter by ``Expression_Type`` if requested.
            3. Build a contiguous subject ID mapping.
            4. For each row, verify the ``.npy`` tensor exists on disk.
            5. Populate ``self.samples`` with validated entries.
        """
        # ── Step 1: Load CSV ────────────────────────────────────────
        if not self._csv_path.exists():
            raise FileNotFoundError(
                f"CSV file not found: {self._csv_path}"
            )

        df = pd.read_csv(self._csv_path, encoding="utf-8-sig")
        self._log.info("Loaded CSV: %d rows × %d columns", len(df), len(df.columns))

        # ── Step 2: Filter by expression type ───────────────────────
        if self._expression_filter is not None:
            before_count = len(df)
            df = df[df["Expression_Type"] == self._expression_filter].copy()
            df.reset_index(drop=True, inplace=True)
            self._log.info(
                "Expression filter '%s': %d → %d rows (dropped %d)",
                self._expression_filter, before_count, len(df),
                before_count - len(df),
            )

        if len(df) == 0:
            self._log.warning("No rows remain after filtering — empty dataset!")
            return

        # ── Step 3: Build contiguous subject mapping ────────────────
        unique_subjects = sorted(df["Subject_ID"].unique().tolist())
        self.subject_map = {
            raw_id: contiguous_idx
            for contiguous_idx, raw_id in enumerate(unique_subjects)
        }
        self._log.info(
            "Subject ID mapping: %d unique subjects → [0, %d]",
            len(unique_subjects), len(unique_subjects) - 1,
        )

        # ── Step 4 & 5: Verify tensors and build sample list ────────
        matched_count = 0
        missing_count = 0
        unknown_emotion_count = 0
        emotion_distribution: Dict[str, int] = {}

        for _, row in df.iterrows():
            dataset_tag = str(row["Dataset"])
            video_id = str(row["Video_ID"])
            unified_emotion = str(row["Unified_Emotion"])
            subject_id = int(row["Subject_ID"])

            # ── Construct expected tensor filename ──────────────────
            tensor_filename = f"{dataset_tag}_{video_id}.npy"
            tensor_path = self._tensor_dir / tensor_filename

            # ── Check tensor existence ──────────────────────────────
            if not tensor_path.exists():
                missing_count += 1
                continue

            # ── Validate emotion label ──────────────────────────────
            if unified_emotion not in self.emotion_map:
                self._log.debug(
                    "Unknown emotion '%s' for %s — skipping.",
                    unified_emotion, tensor_filename,
                )
                unknown_emotion_count += 1
                continue

            # ── Build sample entry ──────────────────────────────────
            emotion_label = self.emotion_map[unified_emotion]
            subject_label = self.subject_map[subject_id]

            self.samples.append({
                "tensor_path": tensor_path,
                "emotion_label": emotion_label,
                "subject_label": subject_label,
                "dataset": dataset_tag,
                "video_id": video_id,
                "unified_emotion": unified_emotion,
                "raw_subject_id": subject_id,
            })
            matched_count += 1

            # ── Accumulate distribution ─────────────────────────────
            emotion_distribution[unified_emotion] = (
                emotion_distribution.get(unified_emotion, 0) + 1
            )

        # ── Log statistics ──────────────────────────────────────────
        self._log.info("Tensor matching results:")
        self._log.info("  Matched  : %d", matched_count)
        self._log.info("  Missing  : %d (no .npy file on disk)", missing_count)
        self._log.info("  Unknown  : %d (emotion not in map)", unknown_emotion_count)

        self._log.info("Emotion class distribution:")
        for emotion, count in sorted(emotion_distribution.items()):
            label_int = self.emotion_map.get(emotion, "?")
            self._log.info("  %-12s (label=%s) : %d samples", emotion, label_int, count)

        # ── Log subject distribution ────────────────────────────────
        subject_counts: Dict[int, int] = {}
        for s in self.samples:
            sid = s["raw_subject_id"]
            subject_counts[sid] = subject_counts.get(sid, 0) + 1

        self._log.info("Subject distribution (top 10):")
        for sid, count in sorted(
            subject_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]:
            self._log.info(
                "  Subject %3d → mapped_id=%d : %d samples",
                sid, self.subject_map[sid], count,
            )

    # ─────────────────────────────────────────────────────────────────
    #  PyTorch Dataset Interface
    # ─────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Return the number of valid samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        Load and return a single sample.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        tuple of (torch.Tensor, int, int)
            ``(tensor, emotion_label, subject_label)`` where:
            - ``tensor`` has shape ``[3, 32, 224, 224]`` (float32)
            - ``emotion_label`` is an integer class index
            - ``subject_label`` is a contiguous integer subject index
        """
        sample = self.samples[idx]

        # ── Load tensor from disk ───────────────────────────────────
        tensor_np = np.load(str(sample["tensor_path"]))
        tensor = torch.from_numpy(tensor_np).float()

        # ── Apply augmentations ──────────────────────────────────────
        T = tensor.size(1)
        if T > 32:
            if self._transform:
                start_idx = torch.randint(0, T - 32 + 1, (1,)).item()
            else:
                start_idx = (T - 32) // 2
            tensor = tensor[:, start_idx:start_idx+32, :, :]

        if self._transform:
            import random
            if random.random() > 0.5:
                # Spatial flip (width is dim 3)
                tensor = torch.flip(tensor, dims=[3])
                # Negate u channel (horizontal flow) to align directions
                tensor[0, :, :, :] *= -1

        emotion_label = sample["emotion_label"]
        subject_label = sample["subject_label"]

        return tensor, emotion_label, subject_label

    # ─────────────────────────────────────────────────────────────────
    #  Utility Methods
    # ─────────────────────────────────────────────────────────────────

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for the emotion labels.

        Useful for weighted loss functions to handle class imbalance.

        Returns
        -------
        torch.Tensor
            Weight vector of shape ``[num_classes]``.
        """
        counts = torch.zeros(self.num_classes, dtype=torch.float32)
        for sample in self.samples:
            counts[sample["emotion_label"]] += 1.0

        # ── Inverse frequency with smoothing ────────────────────────
        total = counts.sum()
        weights = total / (self.num_classes * counts.clamp(min=1.0))

        self._log.info(
            "Class weights (inverse freq): %s",
            {k: f"{weights[v]:.4f}" for k, v in self.emotion_map.items()},
        )
        return weights

    def get_emotion_distribution(self) -> Dict[str, int]:
        """Return a dict of emotion_name → count."""
        dist: Dict[str, int] = {}
        for sample in self.samples:
            emotion = sample["unified_emotion"]
            dist[emotion] = dist.get(emotion, 0) + 1
        return dist

    def get_unique_subject_ids(self) -> List[int]:
        """
        Return the list of unique *raw* Subject_IDs present in this dataset.

        Returns
        -------
        list of int
            Sorted list of unique raw Subject_IDs.
        """
        return sorted(set(s["raw_subject_id"] for s in self.samples))

    def get_subject_for_sample(self, idx: int) -> int:
        """
        Return the raw Subject_ID for sample at index ``idx``.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        int
            The raw (unmapped) Subject_ID.
        """
        return self.samples[idx]["raw_subject_id"]

    def get_indices_for_subjects(self, subject_ids: List[int]) -> List[int]:
        """
        Return all sample indices belonging to the given raw Subject_IDs.

        Parameters
        ----------
        subject_ids : list of int
            Raw Subject_IDs to retrieve indices for.

        Returns
        -------
        list of int
            Sorted list of sample indices.
        """
        sid_set = set(subject_ids)
        return sorted(
            i for i, s in enumerate(self.samples)
            if s["raw_subject_id"] in sid_set
        )

    @staticmethod
    def subject_disjoint_split(
        dataset: "MERDataset",
        val_fraction: float = 0.2,
        seed: int = 42,
    ) -> Tuple[List[int], List[int]]:
        """
        Split the dataset into train/val by **Subject_ID** (not by sample).

        This avoids **identity leakage**: no subject appears in both
        the training and validation sets.  This is critical for
        adversarial domain adaptation because the GRL is designed to
        remove identity information — testing on a seen identity
        would invalidate the entire experimental design.

        Algorithm:
            1. Collect unique Subject_IDs.
            2. Randomly shuffle them (deterministically via seed).
            3. Assign ``ceil(val_fraction * num_subjects)`` subjects
               to validation, remainder to training.
            4. Return the sample indices for each group.

        Parameters
        ----------
        dataset : MERDataset
            The dataset to split.
        val_fraction : float
            Fraction of *subjects* to hold out for validation.
        seed : int
            Random seed for reproducible shuffling.

        Returns
        -------
        tuple of (list[int], list[int])
            ``(train_indices, val_indices)`` — disjoint sample indices.
        """
        import random

        unique_subjects = dataset.get_unique_subject_ids()
        num_subjects = len(unique_subjects)

        # ── Deterministic shuffle ───────────────────────────────────
        rng = random.Random(seed)
        shuffled = list(unique_subjects)
        rng.shuffle(shuffled)

        # ── Split subjects ──────────────────────────────────────────
        num_val_subjects = max(1, int(round(val_fraction * num_subjects)))
        val_subjects = shuffled[:num_val_subjects]
        train_subjects = shuffled[num_val_subjects:]

        # ── Convert to sample indices ───────────────────────────────
        train_indices = dataset.get_indices_for_subjects(train_subjects)
        val_indices = dataset.get_indices_for_subjects(val_subjects)

        # ── Log the split ───────────────────────────────────────────
        dataset._log.info("─" * 70)
        dataset._log.info("  Subject-Disjoint Split (seed=%d)", seed)
        dataset._log.info("─" * 70)
        dataset._log.info(
            "  Total subjects: %d → Train: %d subjects (%d samples) "
            "| Val: %d subjects (%d samples)",
            num_subjects,
            len(train_subjects), len(train_indices),
            len(val_subjects), len(val_indices),
        )
        dataset._log.info("  Val subjects  : %s", sorted(val_subjects))
        dataset._log.info("  Train subjects: %s", sorted(train_subjects))

        # ── Sanity check: no overlap ────────────────────────────────
        overlap = set(train_subjects) & set(val_subjects)
        assert len(overlap) == 0, (
            f"CRITICAL: Subject overlap detected between train/val: {overlap}"
        )
        assert len(train_indices) + len(val_indices) == len(dataset), (
            f"Index count mismatch: {len(train_indices)} + {len(val_indices)} "
            f"!= {len(dataset)}"
        )

        dataset._log.info("  ✓ Zero subject overlap verified.")
        dataset._log.info("─" * 70)

        return train_indices, val_indices


# ─────────────────────────────────────────────────────────────────────
# Standalone Verification
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  MERDataset — Standalone Verification")
    print("=" * 70)

    csv_path = _PROJECT_ROOT / "Processed_Data" / "master_thesis_labels.csv"
    tensor_dir = _PROJECT_ROOT / "Processed_Data" / "tensors"

    # ── Test with micro-expression filter ───────────────────────────
    dataset = MERDataset(
        csv_path=csv_path,
        tensor_dir=tensor_dir,
        expression_filter="micro-expression",
    )

    print(f"\nDataset size   : {len(dataset)}")
    print(f"Num classes    : {dataset.num_classes}")
    print(f"Num subjects   : {dataset.num_subjects}")
    print(f"Emotion map    : {dataset.emotion_map}")

    if len(dataset) > 0:
        tensor, emotion, subject = dataset[0]
        print(f"\nSample 0:")
        print(f"  Tensor shape : {tensor.shape}")
        print(f"  Tensor dtype : {tensor.dtype}")
        print(f"  Emotion label: {emotion}")
        print(f"  Subject label: {subject}")

        weights = dataset.get_class_weights()
        print(f"\nClass weights  : {weights}")

    print(f"\n{'=' * 70}")
    print(f"  ✓ MERDataset verification complete")
    print(f"{'=' * 70}")
