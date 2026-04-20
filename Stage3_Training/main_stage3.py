"""
main_stage3.py — Stage 3 Orchestrator: Adversarial Domain Adaptation + SupCon
===============================================================================

This is the top-level entry point that ties together all Stage 3
components:

    1. Parse command-line arguments
    2. Initialize MERDataset with expression filtering
    3. Subject-Disjoint Split (no identity leakage between train/val)
    4. Create DataLoaders
    5. Instantiate AdversarialMERWrapper (with Projection Head)
    6. Instantiate SupConLoss + FocalLoss + CrossEntropyLoss
    7. Configure optimizer (AdamW) and scheduler (CosineAnnealing)
    8. Create AdversarialTrainer and call trainer.train()

Loss Function:
    Total Loss = (β × SupCon Loss) + Emotion Loss + (α × Identity Loss)

    where SupCon anchors the latent space against GRL-induced feature
    collapse, Focal Loss handles class imbalance, and CE through GRL
    forces identity-invariance.

Usage::

    # Basic usage (micro-expression only, 100 epochs)
    python main_stage3.py

    # Custom configuration with SupCon tuning
    python main_stage3.py --epochs 200 --supcon_weight 1.0 --supcon_temperature 0.1

    # Resume from checkpoint
    python main_stage3.py  # (automatic — detects latest_checkpoint.pth)

Author  : Addhyan
Stage   : 3 — Adversarial Domain Adaptation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# ── Resolve paths ───────────────────────────────────────────────────
_STAGE3_DIR = Path(__file__).resolve().parent               # Stage3_Training/
_PROJECT_ROOT = _STAGE3_DIR.parent                          # Thesis3/

# ── Wire up import paths ───────────────────────────────────────────
sys.path.insert(0, str(_PROJECT_ROOT / "Stage1_DataPipeline"))
sys.path.insert(0, str(_PROJECT_ROOT / "Stage2_Architecture"))
sys.path.insert(0, str(_STAGE3_DIR))

from utils.logger import get_logger                         # noqa: E402
from data.mer_dataset import MERDataset                     # noqa: E402
from modules.adversarial_wrapper import AdversarialMERWrapper  # noqa: E402
from losses.focal_loss import FocalLoss                     # noqa: E402
from losses.supcon_loss import SupConLoss                   # noqa: E402
from core.trainer import AdversarialTrainer                 # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Stage 3 training."""
    parser = argparse.ArgumentParser(
        description="Stage 3: Adversarial Domain Adaptation + SupCon Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Dataset arguments ───────────────────────────────────────────
    parser.add_argument(
        "--csv_path",
        type=str,
        default=str(_PROJECT_ROOT / "Processed_Data" / "master_thesis_labels.csv"),
        help="Path to master_thesis_labels.csv",
    )
    parser.add_argument(
        "--tensor_dir",
        type=str,
        default=str(_PROJECT_ROOT / "Processed_Data" / "tensors"),
        help="Path to directory of .npy tensor files",
    )
    parser.add_argument(
        "--expression_filter",
        type=str,
        default=None,
        choices=["micro-expression", "macro-expression", None],
        help="Filter by Expression_Type (None = use all)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Fraction of data to use for validation",
    )

    # ── Model arguments ─────────────────────────────────────────────
    parser.add_argument(
        "--num_emotions",
        type=int,
        default=4,
        help="Number of emotion classes",
    )
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=96,
        help="Feature dimension from Transformer (d_model)",
    )
    parser.add_argument(
        "--proj_dim",
        type=int,
        default=64,
        help="Projection head output dimension for SupCon",
    )
    parser.add_argument(
        "--head_dropout",
        type=float,
        default=0.3,
        help="Dropout rate for classification heads",
    )
    parser.add_argument(
        "--grl_lambda",
        type=float,
        default=0.0,
        help="Initial GRL lambda (annealed via sigmoid schedule)",
    )
    parser.add_argument(
        "--grl_gamma",
        type=float,
        default=10.0,
        help="Gamma parameter for GRL sigmoid annealing schedule",
    )

    # ── SupCon arguments ────────────────────────────────────────────
    parser.add_argument(
        "--supcon_temperature",
        type=float,
        default=0.1,
        help="Temperature τ for SupCon loss (lower = tighter clusters)",
    )
    parser.add_argument(
        "--supcon_weight",
        type=float,
        default=1.0,
        help="Scalar multiplier β for the SupCon loss term",
    )
    parser.add_argument(
        "--xbm_memory_size",
        type=int,
        default=64,
        help="Cross-Batch Memory bank size for SupCon (FIFO queue of past embeddings)",
    )

    # ── Training arguments ──────────────────────────────────────────
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Total number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for AdamW optimizer",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 regularization)",
    )
    parser.add_argument(
        "--identity_loss_weight",
        type=float,
        default=0.5,
        help="Scalar multiplier α for the identity loss term",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Gamma parameter for Focal Loss",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.05,
        help="Label smoothing for Focal Loss (0 = none)",
    )
    parser.add_argument(
        "--gradient_clip_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping (0 = no clipping)",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Enable automatic mixed precision (AMP) training",
    )

    # ── Output arguments ────────────────────────────────────────────
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(_STAGE3_DIR / "checkpoints"),
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers (0 = main thread)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for Stage 3 training."""

    args = parse_args()

    # ── Logger ──────────────────────────────────────────────────────
    log_dir = _STAGE3_DIR / "logs"
    log = get_logger(
        "main_stage3",
        log_dir=log_dir,
        log_filename="stage3_training.log",
    )

    log.info("=" * 70)
    log.info("  STAGE 3: Adversarial Domain Adaptation + SupCon Training")
    log.info("  Micro-Expression Recognition — Master's Thesis")
    log.info("=" * 70)

    # ── Reproducibility ─────────────────────────────────────────────
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    log.info("Random seed: %d", args.seed)

    # ── Device ──────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("  GPU: %s", torch.cuda.get_device_name(0))
        log.info("  VRAM: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)

    # ─────────────────────────────────────────────────────────────────
    # 1. Dataset
    # ─────────────────────────────────────────────────────────────────
    log.info("\n[1/7] Initializing Dataset...")

    full_dataset = MERDataset(
        csv_path=Path(args.csv_path),
        tensor_dir=Path(args.tensor_dir),
        expression_filter=args.expression_filter,
        log_dir=log_dir,
    )

    if len(full_dataset) == 0:
        log.error("Dataset is empty! Check CSV path and tensor directory.")
        sys.exit(1)

    log.info("Full dataset: %d samples, %d classes, %d subjects",
             len(full_dataset), full_dataset.num_classes, full_dataset.num_subjects)

    # ─────────────────────────────────────────────────────────────────
    # 2. Subject-Disjoint Train/Val Split
    # ─────────────────────────────────────────────────────────────────
    #  CRITICAL: We split by Subject_ID, NOT by individual samples.
    #  random_split would leak identities across train/val, which
    #  invalidates the GRL-based adversarial training since we are
    #  literally training the network to be identity-invariant.
    #
    #  Subject-disjoint split guarantees zero identity overlap.
    # ─────────────────────────────────────────────────────────────────
    log.info("\n[2/7] Subject-Disjoint Split (%.0f%% val subjects)...",
             args.val_split * 100)

    train_indices, val_indices = MERDataset.subject_disjoint_split(
        dataset=full_dataset,
        val_fraction=args.val_split,
        seed=args.seed,
    )

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    log.info("Train split: %d samples", len(train_dataset))
    log.info("Val split  : %d samples", len(val_dataset))

    # ── Verify zero subject overlap ─────────────────────────────────
    train_sids = set(full_dataset.get_subject_for_sample(i) for i in train_indices)
    val_sids = set(full_dataset.get_subject_for_sample(i) for i in val_indices)
    overlap = train_sids & val_sids
    assert len(overlap) == 0, (
        f"DATA LEAKAGE: {len(overlap)} subjects appear in both train and val: {overlap}"
    )
    log.info("✓ Subject-disjoint split verified: 0 subject overlap.")
    log.info("  Train subjects (%d): %s", len(train_sids), sorted(train_sids))
    log.info("  Val subjects   (%d): %s", len(val_sids), sorted(val_sids))

    # ── DataLoaders ─────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    log.info("Train batches: %d (batch_size=%d)", len(train_loader), args.batch_size)
    log.info("Val batches  : %d", len(val_loader))

    # ─────────────────────────────────────────────────────────────────
    # 3. Model
    # ─────────────────────────────────────────────────────────────────
    log.info("\n[3/7] Instantiating AdversarialMERWrapper (with Projection Head)...")

    model = AdversarialMERWrapper(
        num_emotions=args.num_emotions,
        num_subjects=full_dataset.num_subjects,
        grl_lambda=args.grl_lambda,
        feature_dim=args.feature_dim,
        proj_dim=args.proj_dim,
        head_dropout=args.head_dropout,
    )

    params = model.count_parameters()
    log.info("Model parameters:")
    log.info("  Backbone (Stage 2):  %10s trainable", f"{params['backbone_trainable']:,}")
    log.info("  Projection Head:     %10s trainable", f"{params['projection_head_trainable']:,}")
    log.info("  Emotion Head:        %10s trainable", f"{params['emotion_head_trainable']:,}")
    log.info("  Identity Head:       %10s trainable", f"{params['identity_head_trainable']:,}")
    log.info("  TOTAL:               %10s trainable", f"{params['model_trainable']:,}")

    # ─────────────────────────────────────────────────────────────────
    # 4. Loss Functions
    # ─────────────────────────────────────────────────────────────────
    log.info("\n[4/7] Configuring Loss Functions...")

    # ── Supervised Contrastive Loss (SupCon) ────────────────────────
    supcon_criterion = SupConLoss(
        temperature=args.supcon_temperature,
        memory_size=args.xbm_memory_size,
        proj_dim=args.proj_dim,
    )
    log.info("SupCon Loss: SupConLoss(τ=%.2f, weight=%.2f, XBM=%d, proj_dim=%d)",
             args.supcon_temperature, args.supcon_weight,
             args.xbm_memory_size, args.proj_dim)

    # ── Focal Loss for emotions (with class weights) ────────────────
    class_weights = full_dataset.get_class_weights()
    emotion_criterion = FocalLoss(
        alpha=class_weights,
        gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )
    log.info("Emotion Loss: FocalLoss(γ=%.1f, α=%s, smoothing=%.2f)",
             args.focal_gamma,
             [f"{w:.3f}" for w in class_weights.tolist()],
             args.label_smoothing)

    # ── Cross-Entropy for identity ──────────────────────────────────
    identity_criterion = nn.CrossEntropyLoss()
    log.info("Identity Loss: CrossEntropyLoss (unweighted)")

    log.info("Total Loss = (%.2f × SupCon) + Emotion + (%.2f × Identity)",
             args.supcon_weight, args.identity_loss_weight)

    # ─────────────────────────────────────────────────────────────────
    # 5. Optimizer & Scheduler
    # ─────────────────────────────────────────────────────────────────
    log.info("\n[5/7] Configuring Optimizer & Scheduler...")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    log.info("Optimizer: AdamW(lr=%.6f, weight_decay=%.6f)", args.lr, args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-7,
    )
    log.info("Scheduler: CosineAnnealingLR(T_max=%d, eta_min=1e-7)", args.epochs)

    # ─────────────────────────────────────────────────────────────────
    # 6. Trainer
    # ─────────────────────────────────────────────────────────────────
    log.info("\n[6/7] Creating AdversarialTrainer...")

    clip_norm = args.gradient_clip_norm if args.gradient_clip_norm > 0 else None

    trainer = AdversarialTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        emotion_criterion=emotion_criterion,
        identity_criterion=identity_criterion,
        supcon_criterion=supcon_criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        identity_loss_weight=args.identity_loss_weight,
        supcon_loss_weight=args.supcon_weight,
        grl_gamma=args.grl_gamma,
        checkpoint_dir=Path(args.checkpoint_dir),
        log_dir=log_dir,
        use_amp=args.use_amp,
        gradient_clip_norm=clip_norm,
    )

    # ─────────────────────────────────────────────────────────────────
    # 7. Launch Training
    # ─────────────────────────────────────────────────────────────────
    log.info("\n[7/7] Launching Training...")
    log.info("\n" + "═" * 70)
    log.info("  LAUNCHING ADVERSARIAL + SUPCON TRAINING")
    log.info("═" * 70)

    history = trainer.train()

    # ─────────────────────────────────────────────────────────────────
    # Post-Training Summary
    # ─────────────────────────────────────────────────────────────────
    log.info("\n" + "═" * 70)
    log.info("  POST-TRAINING SUMMARY")
    log.info("═" * 70)

    if history["train_emotion_acc"]:
        log.info("Final Train Emotion Accuracy : %.4f", history["train_emotion_acc"][-1])
        log.info("Final Train Identity Accuracy: %.4f", history["train_identity_acc"][-1])
        log.info("Final Train SupCon Loss      : %.4f", history["train_supcon_loss"][-1])
    if history.get("val_emotion_acc"):
        log.info("Final Val Emotion Accuracy   : %.4f", history["val_emotion_acc"][-1])
        log.info("Final Val Identity Accuracy  : %.4f", history["val_identity_acc"][-1])
    log.info("Best Val Accuracy            : %.4f", trainer._best_val_accuracy)
    log.info("Final GRL Lambda             : %.4f", history["grl_lambda"][-1] if history["grl_lambda"] else 0)

    log.info("\nCheckpoints saved to: %s", args.checkpoint_dir)
    log.info("  • latest_checkpoint.pth  — for crash recovery")
    log.info("  • best_model.pth         — optimal weights")

    log.info("\n" + "═" * 70)
    log.info("  STAGE 3 COMPLETE")
    log.info("═" * 70)


if __name__ == "__main__":
    main()
