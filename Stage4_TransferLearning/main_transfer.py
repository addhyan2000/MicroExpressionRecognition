"""
main_transfer.py — Stage 4: Macro-to-Micro Transfer Learning Orchestrator
===========================================================================

This is the top-level entry point for the two-phase transfer learning
pipeline designed to overcome the data starvation wall (~400 micro-expression
videos) that causes feature collapse when training from scratch.

Strategy:
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Phase 1: Macro-Expression Pre-training (Adversary OFF)            │
    │  ─────────────────────────────────────────────────────────          │
    │  Train on ~2,000 macro-expression videos with identity_loss_wt=0.0 │
    │  to teach the network general facial motion dynamics.              │
    │  GRL is disabled so the backbone can freely use identity cues      │
    │  to bootstrap feature learning before we constrain it.             │
    │                                                                     │
    │  Output: macro_pretrained_best.pth                                 │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Phase 2: Micro-Expression Fine-tuning (Adversary ON)              │
    │  ─────────────────────────────────────────────────────              │
    │  Load macro_pretrained_best.pth into a fresh wrapper.              │
    │  Fine-tune on ~400 micro-expression videos with identity_loss_wt>0 │
    │  so the GRL forces identity-invariance while preserving emotion    │
    │  features learned in Phase 1.                                      │
    │                                                                     │
    │  Output: micro_finetuned_best.pth                                  │
    └─────────────────────────────────────────────────────────────────────┘

Fault Tolerance (State Machine):
    • If `macro_pretrained_best.pth` already exists → skip directly to Phase 2.
    • If Phase 2 crashes mid-training → detects `latest_checkpoint.pth` in the
      Phase 2 checkpoint directory and resumes seamlessly.
    • All exceptions are caught, logged, and re-raised for visibility.

Usage::

    # Full two-phase run (default settings)
    python Stage4_TransferLearning/main_transfer.py

    # Custom hyperparameters
    python Stage4_TransferLearning/main_transfer.py \\
        --macro_epochs 50 --macro_lr 1e-4 \\
        --micro_epochs 150 --micro_lr 5e-5 \\
        --batch_size 2 --identity_loss_weight 0.4 --use_amp

    # Resume after Phase 2 crash (automatic — just re-run the same command)
    python Stage4_TransferLearning/main_transfer.py --identity_loss_weight 0.4 --use_amp

Author  : Addhyan
Stage   : 4 — Macro-to-Micro Transfer Learning
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# ─────────────────────────────────────────────────────────────────────
# Path Resolution
# ─────────────────────────────────────────────────────────────────────
# Stage4_TransferLearning/ sits alongside Stage1..3 inside Thesis3/.
_STAGE4_DIR = Path(__file__).resolve().parent              # Stage4_TransferLearning/
_PROJECT_ROOT = _STAGE4_DIR.parent                         # Thesis3/
_STAGE3_DIR = _PROJECT_ROOT / "Stage3_Training"

# ── Wire up import paths to reach all three prior stages ────────────
sys.path.insert(0, str(_PROJECT_ROOT / "Stage1_DataPipeline"))
sys.path.insert(0, str(_PROJECT_ROOT / "Stage2_Architecture"))
sys.path.insert(0, str(_STAGE3_DIR))

# ─────────────────────────────────────────────────────────────────────
# Stage 3 Imports  (ZERO code duplication — everything is imported)
# ─────────────────────────────────────────────────────────────────────
from utils.logger import get_logger                                # noqa: E402
from data.mer_dataset import MERDataset                            # noqa: E402
from modules.adversarial_wrapper import AdversarialMERWrapper      # noqa: E402
from losses.focal_loss import FocalLoss                            # noqa: E402
from losses.supcon_loss import SupConLoss                          # noqa: E402
from core.trainer import AdversarialTrainer                        # noqa: E402


# ═════════════════════════════════════════════════════════════════════
#  Transfer Learning Orchestrator
# ═════════════════════════════════════════════════════════════════════

class TransferLearningOrchestrator:
    """
    Two-phase transfer learning orchestrator for Micro-Expression Recognition.

    This class encapsulates the full Macro→Micro transfer pipeline as a
    crash-proof state machine.  It is designed to be re-entrant: re-running
    after a crash will detect completed phases (via checkpoint files) and
    resume from the correct point.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments containing all hyperparameters for
        both phases, plus shared configuration (paths, device, etc.).

    Attributes
    ----------
    device : torch.device
        CUDA or CPU device for training.
    log : logging.Logger
        Dedicated Stage 4 logger.
    macro_checkpoint_dir : Path
        Directory for Phase 1 checkpoints.
    micro_checkpoint_dir : Path
        Directory for Phase 2 checkpoints.
    macro_best_path : Path
        Path to the saved Phase 1 best model weights.
    micro_best_path : Path
        Path to the saved Phase 2 best model weights.
    """

    # ─────────────────────────────────────────────────────────────────
    #  Construction
    # ─────────────────────────────────────────────────────────────────

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

        # ── Logger ──────────────────────────────────────────────────
        self.log_dir = _STAGE4_DIR / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log = get_logger(
            "TransferOrchestrator",
            log_dir=self.log_dir,
            log_filename="stage4_transfer.log",
        )

        # ── Reproducibility ─────────────────────────────────────────
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        # ── Device ──────────────────────────────────────────────────
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ── Checkpoint Directories ──────────────────────────────────
        self.macro_checkpoint_dir = _STAGE4_DIR / "checkpoints" / "phase1_macro"
        self.micro_checkpoint_dir = _STAGE4_DIR / "checkpoints" / "phase2_micro"
        self.macro_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.micro_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # ── State Machine Paths ─────────────────────────────────────
        self.macro_best_path = self.macro_checkpoint_dir / "macro_pretrained_best.pth"
        self.micro_best_path = self.micro_checkpoint_dir / "micro_finetuned_best.pth"

    # ─────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Execute the full two-phase transfer learning pipeline.

        State machine logic:
            1. If ``macro_pretrained_best.pth`` exists → skip Phase 1.
            2. Otherwise → run Phase 1 (macro pre-training).
            3. Run Phase 2 (micro fine-tuning) — the trainer internally
               detects ``latest_checkpoint.pth`` for crash recovery.
        """
        self._print_banner()

        # ════════════════════════════════════════════════════════════
        #  Phase 1: Macro-Expression Pre-training
        # ════════════════════════════════════════════════════════════
        if self.macro_best_path.exists():
            self.log.info("═" * 70)
            self.log.info(
                "  PHASE 1 ALREADY COMPLETED — checkpoint found:"
            )
            self.log.info("    %s", self.macro_best_path)
            self.log.info("  Skipping directly to Phase 2.")
            self.log.info("═" * 70)
        else:
            try:
                self._run_macro_pretraining()
            except Exception as exc:
                self.log.error(
                    "PHASE 1 FAILED with %s: %s",
                    type(exc).__name__, exc,
                    exc_info=True,
                )
                raise

        # ════════════════════════════════════════════════════════════
        #  Phase 2: Micro-Expression Fine-tuning
        # ════════════════════════════════════════════════════════════
        try:
            self._run_micro_finetuning()
        except Exception as exc:
            self.log.error(
                "PHASE 2 FAILED with %s: %s",
                type(exc).__name__, exc,
                exc_info=True,
            )
            raise

        # ── Final Summary ───────────────────────────────────────────
        self._print_final_summary()

    # ─────────────────────────────────────────────────────────────────
    #  Phase 1: Macro-Expression Pre-training (Adversary OFF)
    # ─────────────────────────────────────────────────────────────────

    def _run_macro_pretraining(self) -> Dict[str, list]:
        """
        Pre-train the full pipeline on macro-expressions with the
        adversarial identity head disabled (identity_loss_weight = 0.0).

        This teaches the backbone general facial motion dynamics,
        providing a strong initialization for the micro-expression
        fine-tuning phase.

        Returns
        -------
        dict
            Training history from the AdversarialTrainer.
        """
        self.log.info("\n" + "═" * 70)
        self.log.info("  ╔═══════════════════════════════════════════════════╗")
        self.log.info("  ║  PHASE 1: MACRO-EXPRESSION PRE-TRAINING          ║")
        self.log.info("  ║  Adversary: OFF  │  identity_loss_weight = 0.0   ║")
        self.log.info("  ╚═══════════════════════════════════════════════════╝")
        self.log.info("═" * 70)

        phase_start = time.time()

        # ── 1.1 Dataset ─────────────────────────────────────────────
        self.log.info("\n[Phase1 · 1/5] Loading Macro-Expression Dataset...")
        try:
            macro_dataset = MERDataset(
                csv_path=Path(self.args.csv_path),
                tensor_dir=Path(self.args.tensor_dir),
                expression_filter="macro-expression",
                log_dir=self.log_dir,
            )
        except FileNotFoundError as exc:
            self.log.error(
                "Dataset CSV/tensor directory not found: %s", exc
            )
            raise
        except Exception as exc:
            self.log.error(
                "Unexpected error loading macro dataset: %s", exc,
                exc_info=True,
            )
            raise

        if len(macro_dataset) == 0:
            raise RuntimeError(
                "Macro-expression dataset is empty! Check CSV path, tensor "
                "directory, and ensure Expression_Type='macro-expression' "
                "rows exist in master_thesis_labels.csv."
            )

        self.log.info(
            "Macro dataset: %d samples, %d classes, %d subjects",
            len(macro_dataset),
            macro_dataset.num_classes,
            macro_dataset.num_subjects,
        )

        # ── 1.2 Subject-Disjoint Split ──────────────────────────────
        self.log.info("\n[Phase1 · 2/5] Subject-Disjoint Split (80/20)...")
        train_idx, val_idx = MERDataset.subject_disjoint_split(
            dataset=macro_dataset,
            val_fraction=self.args.val_split,
            seed=self.args.seed,
        )
        train_set = Subset(macro_dataset, train_idx)
        val_set = Subset(macro_dataset, val_idx)

        self.log.info("  Train: %d samples  │  Val: %d samples",
                       len(train_set), len(val_set))

        # ── Verify zero subject overlap ─────────────────────────────
        train_sids = set(
            macro_dataset.get_subject_for_sample(i) for i in train_idx
        )
        val_sids = set(
            macro_dataset.get_subject_for_sample(i) for i in val_idx
        )
        overlap = train_sids & val_sids
        assert len(overlap) == 0, (
            f"DATA LEAKAGE in Phase 1: {len(overlap)} subjects in both "
            f"train and val: {overlap}"
        )
        self.log.info("  ✓ Subject-disjoint split verified: 0 overlap.")

        # ── DataLoaders ─────────────────────────────────────────────
        train_loader = DataLoader(
            train_set,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
        )

        # ── 1.3 Model ──────────────────────────────────────────────
        self.log.info("\n[Phase1 · 3/5] Instantiating AdversarialMERWrapper...")
        model = AdversarialMERWrapper(
            num_emotions=self.args.num_emotions,
            num_subjects=macro_dataset.num_subjects,
            grl_lambda=0.0,           # Adversary OFF in Phase 1
            feature_dim=self.args.feature_dim,
            proj_dim=self.args.proj_dim,
            head_dropout=self.args.head_dropout,
        )
        self._log_model_params(model)

        # ── 1.4 Loss Functions ──────────────────────────────────────
        self.log.info("\n[Phase1 · 4/5] Configuring Loss Functions...")

        supcon_criterion = SupConLoss(
            temperature=self.args.supcon_temperature,
            memory_size=self.args.xbm_memory_size,
            proj_dim=self.args.proj_dim,
        )

        class_weights = macro_dataset.get_class_weights()
        emotion_criterion = FocalLoss(
            alpha=class_weights,
            gamma=self.args.focal_gamma,
            label_smoothing=self.args.label_smoothing,
        )
        identity_criterion = nn.CrossEntropyLoss()

        # CRITICAL: identity_loss_weight = 0.0 → Adversary OFF
        self.log.info(
            "  ★ ADVERSARY OFF: identity_loss_weight = 0.0 "
            "(macro pre-training learns facial physics freely)"
        )

        # ── Optimizer & Scheduler ───────────────────────────────────
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.args.macro_lr,
            weight_decay=self.args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.args.macro_epochs,
            eta_min=1e-7,
        )
        self.log.info(
            "  Optimizer: AdamW(lr=%.6f)  │  Scheduler: CosineAnnealing(T=%d)",
            self.args.macro_lr, self.args.macro_epochs,
        )

        # ── 1.5 Trainer ────────────────────────────────────────────
        self.log.info("\n[Phase1 · 5/5] Creating AdversarialTrainer...")

        clip_norm: Optional[float] = (
            self.args.gradient_clip_norm
            if self.args.gradient_clip_norm > 0 else None
        )

        trainer = AdversarialTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            emotion_criterion=emotion_criterion,
            identity_criterion=identity_criterion,
            supcon_criterion=supcon_criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            num_epochs=self.args.macro_epochs,
            identity_loss_weight=0.0,  # ← ADVERSARY OFF
            supcon_loss_weight=self.args.supcon_weight,
            grl_gamma=self.args.grl_gamma,
            checkpoint_dir=self.macro_checkpoint_dir,
            log_dir=self.log_dir,
            use_amp=self.args.use_amp,
            gradient_clip_norm=clip_norm,
        )

        # ── Launch Training ─────────────────────────────────────────
        self.log.info("\n  ▶ LAUNCHING PHASE 1 TRAINING (%d epochs)...",
                       self.args.macro_epochs)
        try:
            history = trainer.train()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                self.log.error(
                    "CUDA OOM during Phase 1! Try reducing --batch_size "
                    "or disabling AMP. Current batch_size=%d",
                    self.args.batch_size,
                )
            raise

        # ── Save best weights specifically as macro_pretrained_best ──
        # The trainer already saved best_model.pth inside the checkpoint
        # dir.  We copy it to the well-known name for Phase 2 to find.
        trainer_best_path = self.macro_checkpoint_dir / "best_model.pth"
        if trainer_best_path.exists():
            import shutil
            shutil.copy2(str(trainer_best_path), str(self.macro_best_path))
            self.log.info(
                "★ Phase 1 best model copied → %s", self.macro_best_path
            )
        else:
            # Fallback: save the current model state directly
            torch.save(
                {"model_state_dict": model.state_dict()},
                str(self.macro_best_path),
            )
            self.log.info(
                "★ Phase 1 model state saved → %s", self.macro_best_path
            )

        phase_time = time.time() - phase_start
        self.log.info(
            "\n  ✓ PHASE 1 COMPLETE in %.1f min  │  Best Val Acc: %.4f",
            phase_time / 60, trainer._best_val_accuracy,
        )

        return history

    # ─────────────────────────────────────────────────────────────────
    #  Phase 2: Micro-Expression Fine-tuning (Adversary ON)
    # ─────────────────────────────────────────────────────────────────

    def _run_micro_finetuning(self) -> Dict[str, list]:
        """
        Fine-tune on micro-expressions with the adversarial identity
        head enabled (identity_loss_weight > 0) to produce identity-
        invariant emotion features.

        Loads the macro pre-trained weights as initialization, then trains
        a fresh optimizer/scheduler on the micro-expression data.

        Returns
        -------
        dict
            Training history from the AdversarialTrainer.
        """
        self.log.info("\n" + "═" * 70)
        self.log.info("  ╔═══════════════════════════════════════════════════╗")
        self.log.info("  ║  PHASE 2: MICRO-EXPRESSION FINE-TUNING           ║")
        self.log.info("  ║  Adversary: ON   │  identity_loss_weight = %.2f  ║",
                       self.args.identity_loss_weight)
        self.log.info("  ╚═══════════════════════════════════════════════════╝")
        self.log.info("═" * 70)

        phase_start = time.time()

        # ── 2.1 Dataset ─────────────────────────────────────────────
        self.log.info("\n[Phase2 · 1/6] Loading Micro-Expression Dataset...")
        try:
            micro_dataset = MERDataset(
                csv_path=Path(self.args.csv_path),
                tensor_dir=Path(self.args.tensor_dir),
                expression_filter="micro-expression",
                log_dir=self.log_dir,
            )
        except FileNotFoundError as exc:
            self.log.error(
                "Dataset CSV/tensor directory not found: %s", exc
            )
            raise
        except Exception as exc:
            self.log.error(
                "Unexpected error loading micro dataset: %s", exc,
                exc_info=True,
            )
            raise

        if len(micro_dataset) == 0:
            raise RuntimeError(
                "Micro-expression dataset is empty! Check CSV path, tensor "
                "directory, and ensure Expression_Type='micro-expression' "
                "rows exist in master_thesis_labels.csv."
            )

        self.log.info(
            "Micro dataset: %d samples, %d classes, %d subjects",
            len(micro_dataset),
            micro_dataset.num_classes,
            micro_dataset.num_subjects,
        )

        # ── 2.2 Subject-Disjoint Split ──────────────────────────────
        self.log.info("\n[Phase2 · 2/6] Subject-Disjoint Split (80/20)...")
        train_idx, val_idx = MERDataset.subject_disjoint_split(
            dataset=micro_dataset,
            val_fraction=self.args.val_split,
            seed=self.args.seed,
        )
        train_set = Subset(micro_dataset, train_idx)
        val_set = Subset(micro_dataset, val_idx)

        self.log.info("  Train: %d samples  │  Val: %d samples",
                       len(train_set), len(val_set))

        # ── Verify zero subject overlap ─────────────────────────────
        train_sids = set(
            micro_dataset.get_subject_for_sample(i) for i in train_idx
        )
        val_sids = set(
            micro_dataset.get_subject_for_sample(i) for i in val_idx
        )
        overlap = train_sids & val_sids
        assert len(overlap) == 0, (
            f"DATA LEAKAGE in Phase 2: {len(overlap)} subjects in both "
            f"train and val: {overlap}"
        )
        self.log.info("  ✓ Subject-disjoint split verified: 0 overlap.")

        # ── DataLoaders ─────────────────────────────────────────────
        train_loader = DataLoader(
            train_set,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
        )

        # ── 2.3 Model (Fresh Wrapper) ──────────────────────────────
        self.log.info("\n[Phase2 · 3/6] Instantiating FRESH AdversarialMERWrapper...")

        model = AdversarialMERWrapper(
            num_emotions=self.args.num_emotions,
            num_subjects=micro_dataset.num_subjects,
            grl_lambda=self.args.grl_lambda,
            feature_dim=self.args.feature_dim,
            proj_dim=self.args.proj_dim,
            head_dropout=self.args.head_dropout,
        )

        # ── 2.4 Load Macro Pre-trained Weights ─────────────────────
        self.log.info("\n[Phase2 · 4/6] Loading Macro Pre-trained Weights...")

        if not self.macro_best_path.exists():
            raise FileNotFoundError(
                f"Phase 1 checkpoint not found at {self.macro_best_path}! "
                f"Cannot proceed with fine-tuning. Run Phase 1 first."
            )

        try:
            checkpoint = torch.load(
                str(self.macro_best_path),
                map_location=self.device,
                weights_only=False,
            )
            macro_state = checkpoint["model_state_dict"]
        except Exception as exc:
            self.log.error(
                "Failed to load macro checkpoint: %s", exc, exc_info=True,
            )
            raise

        # ────────────────────────────────────────────────────────────
        # CRITICAL: Partial weight loading strategy
        # ────────────────────────────────────────────────────────────
        # The macro model was trained with num_subjects = N_macro, but
        # the micro model has num_subjects = N_micro (different count).
        # The identity_head's final Linear layer will have mismatched
        # shapes.  We handle this gracefully by loading with strict=False
        # and re-initializing any mismatched layers.
        #
        # Similarly, the SupCon memory bank buffers may differ in
        # shape and should be excluded from the transfer.
        # ────────────────────────────────────────────────────────────
        micro_state = model.state_dict()
        transferred_keys = []
        skipped_keys = []

        for key, param in macro_state.items():
            if key in micro_state:
                if micro_state[key].shape == param.shape:
                    micro_state[key] = param
                    transferred_keys.append(key)
                else:
                    skipped_keys.append(
                        f"{key}: macro={list(param.shape)} → "
                        f"micro={list(micro_state[key].shape)}"
                    )
            else:
                skipped_keys.append(f"{key}: not found in micro model")

        model.load_state_dict(micro_state)

        self.log.info(
            "  ✓ Transferred %d/%d parameter tensors from macro checkpoint.",
            len(transferred_keys), len(macro_state),
        )
        if skipped_keys:
            self.log.info(
                "  ⚠ Skipped %d tensors (shape mismatch or missing):",
                len(skipped_keys),
            )
            for sk in skipped_keys:
                self.log.info("      %s", sk)

        self._log_model_params(model)

        # ── 2.5 Loss Functions ──────────────────────────────────────
        self.log.info("\n[Phase2 · 5/6] Configuring Loss Functions...")

        supcon_criterion = SupConLoss(
            temperature=self.args.supcon_temperature,
            memory_size=self.args.xbm_memory_size,
            proj_dim=self.args.proj_dim,
        )

        class_weights = micro_dataset.get_class_weights()
        emotion_criterion = FocalLoss(
            alpha=class_weights,
            gamma=self.args.focal_gamma,
            label_smoothing=self.args.label_smoothing,
        )
        identity_criterion = nn.CrossEntropyLoss()

        # CRITICAL: identity_loss_weight > 0 → Adversary ON
        self.log.info(
            "  ★ ADVERSARY ON: identity_loss_weight = %.2f "
            "(GRL forces identity-invariance on micro data)",
            self.args.identity_loss_weight,
        )

        # ── Optimizer & Scheduler (fresh for fine-tuning) ───────────
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.args.micro_lr,
            weight_decay=self.args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.args.micro_epochs,
            eta_min=1e-7,
        )
        self.log.info(
            "  Optimizer: AdamW(lr=%.6f)  │  Scheduler: CosineAnnealing(T=%d)",
            self.args.micro_lr, self.args.micro_epochs,
        )

        # ── 2.6 Trainer ────────────────────────────────────────────
        self.log.info("\n[Phase2 · 6/6] Creating AdversarialTrainer...")

        clip_norm: Optional[float] = (
            self.args.gradient_clip_norm
            if self.args.gradient_clip_norm > 0 else None
        )

        trainer = AdversarialTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            emotion_criterion=emotion_criterion,
            identity_criterion=identity_criterion,
            supcon_criterion=supcon_criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            num_epochs=self.args.micro_epochs,
            identity_loss_weight=self.args.identity_loss_weight,  # ← ADVERSARY ON
            supcon_loss_weight=self.args.supcon_weight,
            grl_gamma=self.args.grl_gamma,
            checkpoint_dir=self.micro_checkpoint_dir,
            log_dir=self.log_dir,
            use_amp=self.args.use_amp,
            gradient_clip_norm=clip_norm,
        )

        # ── Launch Training ─────────────────────────────────────────
        # NOTE: The AdversarialTrainer's _try_resume() will automatically
        # detect latest_checkpoint.pth in micro_checkpoint_dir if Phase 2
        # was previously interrupted.  This is the crash-recovery path.
        self.log.info(
            "\n  ▶ LAUNCHING PHASE 2 TRAINING (%d epochs)...",
            self.args.micro_epochs,
        )
        try:
            history = trainer.train()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                self.log.error(
                    "CUDA OOM during Phase 2! Try reducing --batch_size "
                    "or disabling AMP. Current batch_size=%d",
                    self.args.batch_size,
                )
            raise

        # ── Save best weights as micro_finetuned_best ───────────────
        trainer_best_path = self.micro_checkpoint_dir / "best_model.pth"
        if trainer_best_path.exists():
            import shutil
            shutil.copy2(str(trainer_best_path), str(self.micro_best_path))
            self.log.info(
                "★ Phase 2 best model copied → %s", self.micro_best_path
            )
        else:
            torch.save(
                {"model_state_dict": model.state_dict()},
                str(self.micro_best_path),
            )
            self.log.info(
                "★ Phase 2 model state saved → %s", self.micro_best_path
            )

        phase_time = time.time() - phase_start
        self.log.info(
            "\n  ✓ PHASE 2 COMPLETE in %.1f min  │  Best Val Acc: %.4f",
            phase_time / 60, trainer._best_val_accuracy,
        )

        return history

    # ─────────────────────────────────────────────────────────────────
    #  Private Utility Methods
    # ─────────────────────────────────────────────────────────────────

    def _log_model_params(self, model: AdversarialMERWrapper) -> None:
        """Log parameter counts by component for thesis documentation."""
        params = model.count_parameters()
        self.log.info("  Model parameters:")
        self.log.info("    Backbone (Stage 2):  %10s trainable",
                       f"{params['backbone_trainable']:,}")
        self.log.info("    Projection Head:     %10s trainable",
                       f"{params['projection_head_trainable']:,}")
        self.log.info("    Emotion Head:        %10s trainable",
                       f"{params['emotion_head_trainable']:,}")
        self.log.info("    Identity Head:       %10s trainable",
                       f"{params['identity_head_trainable']:,}")
        self.log.info("    TOTAL:               %10s trainable",
                       f"{params['model_trainable']:,}")

    def _print_banner(self) -> None:
        """Print the Stage 4 startup banner with all configuration."""
        self.log.info("\n" + "═" * 70)
        self.log.info("  ████████╗██████╗  █████╗ ███╗   ██╗███████╗███████╗███████╗██████╗ ")
        self.log.info("  ╚══██╔══╝██╔══██╗██╔══██╗████╗  ██║██╔════╝██╔════╝██╔════╝██╔══██╗")
        self.log.info("     ██║   ██████╔╝███████║██╔██╗ ██║███████╗█████╗  █████╗  ██████╔╝")
        self.log.info("     ██║   ██╔══██╗██╔══██║██║╚██╗██║╚════██║██╔══╝  ██╔══╝  ██╔══██╗")
        self.log.info("     ██║   ██║  ██║██║  ██║██║ ╚████║███████║██║     ███████╗██║  ██║")
        self.log.info("     ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝")
        self.log.info("═" * 70)
        self.log.info("  STAGE 4: Macro-to-Micro Transfer Learning")
        self.log.info("  Micro-Expression Recognition — Master's Thesis")
        self.log.info("═" * 70)

        self.log.info("\n  Configuration:")
        self.log.info("  ─────────────────────────────────────────────────")
        self.log.info("  Device         : %s", self.device)
        if self.device.type == "cuda":
            self.log.info("    GPU          : %s", torch.cuda.get_device_name(0))
            self.log.info("    VRAM         : %.1f GB",
                           torch.cuda.get_device_properties(0).total_memory / 1e9)
        self.log.info("  Seed           : %d", self.args.seed)
        self.log.info("  Batch size     : %d", self.args.batch_size)
        self.log.info("  AMP            : %s", self.args.use_amp)
        self.log.info("")
        self.log.info("  Phase 1 (Macro):")
        self.log.info("    Epochs       : %d", self.args.macro_epochs)
        self.log.info("    LR           : %.6f", self.args.macro_lr)
        self.log.info("    Adversary    : OFF (identity_loss_weight=0.0)")
        self.log.info("")
        self.log.info("  Phase 2 (Micro):")
        self.log.info("    Epochs       : %d", self.args.micro_epochs)
        self.log.info("    LR           : %.6f", self.args.micro_lr)
        self.log.info("    Adversary    : ON  (identity_loss_weight=%.2f)",
                       self.args.identity_loss_weight)
        self.log.info("  ─────────────────────────────────────────────────")

    def _print_final_summary(self) -> None:
        """Print the final post-training summary with checkpoint locations."""
        self.log.info("\n" + "═" * 70)
        self.log.info("  ╔═══════════════════════════════════════════════════╗")
        self.log.info("  ║           STAGE 4 TRANSFER LEARNING COMPLETE     ║")
        self.log.info("  ╚═══════════════════════════════════════════════════╝")
        self.log.info("═" * 70)
        self.log.info("")
        self.log.info("  Artifacts:")
        self.log.info("    Phase 1 (Macro): %s", self.macro_best_path)
        self.log.info("    Phase 2 (Micro): %s", self.micro_best_path)
        self.log.info("")
        self.log.info("  To evaluate the final model, load:")
        self.log.info("    checkpoint = torch.load('%s')", self.micro_best_path)
        self.log.info("    model.load_state_dict(checkpoint['model_state_dict'])")
        self.log.info("")
        self.log.info("═" * 70)


# ═════════════════════════════════════════════════════════════════════
#  Argument Parser
# ═════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the transfer learning pipeline."""
    parser = argparse.ArgumentParser(
        description="Stage 4: Macro→Micro Transfer Learning Orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Dataset arguments ───────────────────────────────────────────
    parser.add_argument(
        "--csv_path", type=str,
        default=str(_PROJECT_ROOT / "Processed_Data" / "master_thesis_labels.csv"),
        help="Path to master_thesis_labels.csv",
    )
    parser.add_argument(
        "--tensor_dir", type=str,
        default=str(_PROJECT_ROOT / "Processed_Data" / "tensors"),
        help="Path to directory of .npy tensor files",
    )
    parser.add_argument(
        "--val_split", type=float, default=0.2,
        help="Fraction of subjects held out for validation",
    )

    # ── Phase 1 (Macro Pre-training) ────────────────────────────────
    phase1 = parser.add_argument_group("Phase 1 — Macro Pre-training")
    phase1.add_argument(
        "--macro_epochs", type=int, default=50,
        help="Number of training epochs for macro pre-training",
    )
    phase1.add_argument(
        "--macro_lr", type=float, default=1e-4,
        help="Learning rate for macro pre-training (AdamW)",
    )

    # ── Phase 2 (Micro Fine-tuning) ─────────────────────────────────
    phase2 = parser.add_argument_group("Phase 2 — Micro Fine-tuning")
    phase2.add_argument(
        "--micro_epochs", type=int, default=150,
        help="Number of training epochs for micro fine-tuning",
    )
    phase2.add_argument(
        "--micro_lr", type=float, default=5e-5,
        help="Learning rate for micro fine-tuning (AdamW)",
    )

    # ── Shared Training arguments ───────────────────────────────────
    shared = parser.add_argument_group("Shared Hyperparameters")
    shared.add_argument(
        "--batch_size", type=int, default=4,
        help="Training batch size (both phases)",
    )
    shared.add_argument(
        "--identity_loss_weight", type=float, default=0.4,
        help="Scalar multiplier α for identity loss in Phase 2 "
             "(Phase 1 always uses 0.0)",
    )
    shared.add_argument(
        "--supcon_temperature", type=float, default=0.1,
        help="Temperature τ for SupCon loss",
    )
    shared.add_argument(
        "--supcon_weight", type=float, default=1.0,
        help="Scalar multiplier β for SupCon loss",
    )
    shared.add_argument(
        "--xbm_memory_size", type=int, default=64,
        help="Cross-Batch Memory bank size for SupCon",
    )

    # ── Model arguments ─────────────────────────────────────────────
    model = parser.add_argument_group("Model Architecture")
    model.add_argument(
        "--num_emotions", type=int, default=4,
        help="Number of emotion classes",
    )
    model.add_argument(
        "--feature_dim", type=int, default=96,
        help="Feature dimension from Transformer (d_model)",
    )
    model.add_argument(
        "--proj_dim", type=int, default=64,
        help="Projection head output dimension for SupCon",
    )
    model.add_argument(
        "--head_dropout", type=float, default=0.3,
        help="Dropout rate for classification heads",
    )
    model.add_argument(
        "--grl_lambda", type=float, default=0.0,
        help="Initial GRL lambda (annealed via sigmoid schedule)",
    )
    model.add_argument(
        "--grl_gamma", type=float, default=10.0,
        help="Gamma parameter for GRL sigmoid annealing",
    )

    # ── Loss arguments ──────────────────────────────────────────────
    loss = parser.add_argument_group("Loss Configuration")
    loss.add_argument(
        "--focal_gamma", type=float, default=2.0,
        help="Gamma parameter for Focal Loss",
    )
    loss.add_argument(
        "--label_smoothing", type=float, default=0.05,
        help="Label smoothing for Focal Loss",
    )

    # ── Training arguments ──────────────────────────────────────────
    training = parser.add_argument_group("Training Configuration")
    training.add_argument(
        "--weight_decay", type=float, default=1e-4,
        help="Weight decay (L2 regularization) for AdamW",
    )
    training.add_argument(
        "--gradient_clip_norm", type=float, default=1.0,
        help="Max gradient norm for clipping (0 = no clipping)",
    )
    training.add_argument(
        "--use_amp", action="store_true",
        help="Enable automatic mixed precision (AMP) training",
    )
    training.add_argument(
        "--num_workers", type=int, default=0,
        help="Number of DataLoader workers (0 = main thread)",
    )
    training.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


# ═════════════════════════════════════════════════════════════════════
#  Entry Point
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()
    orchestrator = TransferLearningOrchestrator(args)
    orchestrator.run()
