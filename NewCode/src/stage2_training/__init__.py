# -*- coding: utf-8 -*-
"""Stage 2 — STSTNet Model Training Pipeline.

This package implements the complete training infrastructure for the
Shallow Triple Stream Three-dimensional CNN (STSTNet) for Micro-Expression
Recognition (MER).

Modules:
    dataset  — MERDataset (PyTorch Dataset for .npy tensor loading)
    model    — STSTNet (3D-CNN triple-stream architecture)
    trainer  — METrainer (training loop, validation, checkpointing)
    main     — MERManager (orchestrator with argparse CLI)
"""

from src.stage2_training.dataset import MERDataset
from src.stage2_training.model import STSTNet
from src.stage2_training.trainer import METrainer

__all__ = ["MERDataset", "STSTNet", "METrainer"]
