"""
trainer.py — Adversarial Training Loop with SupCon + Checkpointing (Stage 3)
==============================================================================

Implements ``AdversarialTrainer``, a production-grade, fault-tolerant
training loop that jointly optimizes three objectives:

    Total Loss = (SupCon Loss × β) + Emotion Loss + (Identity Loss × α)

where:
    - **SupCon Loss** anchors the latent space with emotion-aware clustering
      (Khosla et al., 2020), preventing feature collapse at high GRL λ.
    - **Emotion Loss** (Focal Loss) classifies micro-expressions.
    - **Identity Loss** (Cross-Entropy through GRL) forces identity-invariance.

Key Features:
    • **Triple-loss optimization:** SupCon + Focal + CE via GRL
    • **Gradient Accumulation:** Accumulates gradients over multiple
      micro-batches before stepping, enabling effective larger batch sizes
      on limited VRAM.
    • **GRL lambda scheduling:** Sigmoid annealing from 0 → 1 over training
    • **Fault-tolerant checkpointing:**
        - ``latest_checkpoint.pth`` — full state for crash recovery
        - ``best_model.pth``       — optimal weights by validation accuracy
    • **Resume capability:** Automatic detection and loading of checkpoints
    • **Comprehensive logging:** Per-epoch metrics, including separate
      SupCon / Emotion / Identity loss curves
    • **Mixed-precision training:** Optional torch.cuda.amp support

Author  : Addhyan
Stage   : 3 — Adversarial Domain Adaptation
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Wire up Stage 1 logger ─────────────────────────────────────────
_STAGE3_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _STAGE3_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "Stage1_DataPipeline"))

from utils.logger import get_logger  # noqa: E402


class AdversarialTrainer:
    """
    Adversarial training loop for identity-invariant MER with SupCon.

    This class encapsulates the full training lifecycle:
        1. Epoch iteration with GRL lambda annealing
        2. Forward pass through AdversarialMERWrapper (3 outputs)
        3. Triple-loss computation (SupCon + Focal + CE)
        4. Gradient accumulation and optimizer step
        5. Validation evaluation
        6. Checkpoint save/resume logic

    Parameters
    ----------
    model : nn.Module
        The ``AdversarialMERWrapper`` model.
    train_loader : DataLoader
        Training data loader yielding ``(tensor, emotion_label, subject_label)``.
    val_loader : DataLoader or None
        Validation data loader. If ``None``, validation is skipped.
    emotion_criterion : nn.Module
        Loss function for emotion classification (e.g., FocalLoss).
    identity_criterion : nn.Module
        Loss function for identity classification (e.g., CrossEntropyLoss).
    supcon_criterion : nn.Module
        Supervised Contrastive Loss for feature clustering (SupConLoss).
    optimizer : torch.optim.Optimizer
        The optimizer (e.g., AdamW).
    scheduler : torch.optim.lr_scheduler or None
        Optional learning rate scheduler.
    device : torch.device
        Device to run training on.
    num_epochs : int
        Total number of training epochs.
    identity_loss_weight : float
        Scalar multiplier α for the identity loss.
    supcon_loss_weight : float
        Scalar multiplier β for the SupCon loss.
        ``total_loss = β * supcon + emotion + α * identity``
    grl_gamma : float
        Gamma parameter for the GRL sigmoid schedule.
    accumulation_steps : int
        Number of micro-batches to accumulate before an optimizer step.
    checkpoint_dir : Path or None
        Directory for saving checkpoints.
    log_dir : Path or None
        Directory for log files.
    use_amp : bool
        Whether to use automatic mixed precision (AMP).
    gradient_clip_norm : float or None
        If not ``None``, clip gradient norms to this value.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        emotion_criterion: nn.Module,
        identity_criterion: nn.Module,
        supcon_criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: torch.device = torch.device("cpu"),
        num_epochs: int = 100,
        identity_loss_weight: float = 1.0,
        supcon_loss_weight: float = 1.0,
        grl_gamma: float = 10.0,
        accumulation_steps: int = 4,
        checkpoint_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None,
        use_amp: bool = False,
        gradient_clip_norm: Optional[float] = 1.0,
    ) -> None:

        # ── Logger ──────────────────────────────────────────────────
        if log_dir is None:
            log_dir = _STAGE3_DIR / "logs"
        self._log = get_logger(
            "AdversarialTrainer",
            log_dir=log_dir,
            log_filename="stage3_training.log",
        )

        # ── Core components ─────────────────────────────────────────
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.emotion_criterion = emotion_criterion.to(device)
        self.identity_criterion = identity_criterion.to(device)
        self.supcon_criterion = supcon_criterion.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        # ── Training hyperparameters ────────────────────────────────
        self.num_epochs = num_epochs
        self.identity_loss_weight = identity_loss_weight
        self.supcon_loss_weight = supcon_loss_weight
        self.grl_gamma = grl_gamma
        self.accumulation_steps = accumulation_steps
        self.use_amp = use_amp
        self.gradient_clip_norm = gradient_clip_norm

        # ── Checkpointing ───────────────────────────────────────────
        if checkpoint_dir is None:
            checkpoint_dir = _STAGE3_DIR / "checkpoints"
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._latest_ckpt_path = self.checkpoint_dir / "latest_checkpoint.pth"
        self._best_ckpt_path = self.checkpoint_dir / "best_model.pth"

        # ── State tracking ──────────────────────────────────────────
        self._start_epoch: int = 0
        self._best_val_accuracy: float = 0.0
        self._history: Dict[str, list] = {
            "train_loss": [],
            "train_supcon_loss": [],
            "train_emotion_loss": [],
            "train_identity_loss": [],
            "train_emotion_acc": [],
            "train_identity_acc": [],
            "val_loss": [],
            "val_supcon_loss": [],
            "val_emotion_acc": [],
            "val_identity_acc": [],
            "grl_lambda": [],
            "lr": [],
        }

        # ── AMP scaler ─────────────────────────────────────────────
        self._scaler = torch.amp.GradScaler(
            device=device.type, enabled=self.use_amp,
        )

        # ── Log configuration ──────────────────────────────────────
        self._log.info("=" * 70)
        self._log.info("  AdversarialTrainer Configuration (SupCon)")
        self._log.info("=" * 70)
        self._log.info("Device              : %s", self.device)
        self._log.info("Epochs              : %d", self.num_epochs)
        self._log.info("Identity loss weight: %.4f", self.identity_loss_weight)
        self._log.info("SupCon loss weight  : %.4f", self.supcon_loss_weight)
        self._log.info("GRL gamma           : %.2f", self.grl_gamma)
        self._log.info("Accumulation steps  : %d", self.accumulation_steps)
        self._log.info("AMP enabled         : %s", self.use_amp)
        self._log.info("Gradient clip norm  : %s", self.gradient_clip_norm)
        self._log.info("Checkpoint dir      : %s", self.checkpoint_dir)
        self._log.info("Train batches       : %d", len(self.train_loader))
        self._log.info("Val batches         : %d",
                        len(self.val_loader) if self.val_loader else 0)

    # ─────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────

    def train(self) -> Dict[str, list]:
        """
        Execute the full training loop.

        If a ``latest_checkpoint.pth`` exists, training resumes from
        the saved epoch.

        Returns
        -------
        dict
            Training history containing per-epoch metrics.
        """
        # ── Attempt to resume from checkpoint ──────────────────────
        self._try_resume()

        self._log.info("=" * 70)
        self._log.info("  Training Start — Epoch %d/%d",
                        self._start_epoch + 1, self.num_epochs)
        self._log.info("=" * 70)

        for epoch in range(self._start_epoch, self.num_epochs):
            epoch_start = time.time()

            # ── Update GRL lambda (sigmoid schedule) ────────────────
            progress = epoch / max(self.num_epochs - 1, 1)
            grl_lambda = self.model.schedule_grl_lambda(progress, self.grl_gamma)

            # ── Train one epoch ─────────────────────────────────────
            train_metrics = self._train_epoch(epoch)

            # ── Validate ────────────────────────────────────────────
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self._validate_epoch(epoch)

            # ── Step scheduler ──────────────────────────────────────
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                self.scheduler.step()

            # ── Record history ──────────────────────────────────────
            self._record_history(train_metrics, val_metrics, grl_lambda, current_lr)

            # ── Epoch summary log ───────────────────────────────────
            epoch_time = time.time() - epoch_start
            self._log_epoch_summary(epoch, train_metrics, val_metrics,
                                     grl_lambda, current_lr, epoch_time)

            # ── Checkpointing ───────────────────────────────────────
            self._save_latest_checkpoint(epoch)

            # ── Save best model (based on val emotion accuracy) ─────
            val_acc = val_metrics.get("emotion_acc", train_metrics["emotion_acc"])
            if val_acc > self._best_val_accuracy:
                self._best_val_accuracy = val_acc
                self._save_best_model(epoch, val_acc)

        self._log.info("=" * 70)
        self._log.info("  Training Complete — Best Val Accuracy: %.4f",
                        self._best_val_accuracy)
        self._log.info("=" * 70)

        return self._history

    # ─────────────────────────────────────────────────────────────────
    #  Training / Validation Epoch Logic
    # ─────────────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch with gradient accumulation.

        The forward pass unpacks three outputs from the model:
            ``(projected_features, emotion_logits, identity_logits)``

        The total loss is computed as::

            total = (β × SupCon) + Emotion + (α × Identity)

        and is divided by ``accumulation_steps`` before ``.backward()``
        to average gradients across micro-batches.

        Parameters
        ----------
        epoch : int
            Current epoch number (0-indexed).

        Returns
        -------
        dict
            Metrics: total_loss, supcon_loss, emotion_loss, identity_loss,
            emotion_acc, identity_acc.
        """
        self.model.train()

        total_loss_sum = 0.0
        supcon_loss_sum = 0.0
        emotion_loss_sum = 0.0
        identity_loss_sum = 0.0
        emotion_correct = 0
        identity_correct = 0
        total_samples = 0

        self.optimizer.zero_grad()  # Clear gradients before epoch begins

        for batch_idx, (tensors, emotion_labels, subject_labels) in enumerate(
            self.train_loader
        ):
            # ── Move to device ──────────────────────────────────────
            tensors = tensors.to(self.device)
            emotion_labels = emotion_labels.to(self.device).long()
            subject_labels = subject_labels.to(self.device).long()

            batch_size = tensors.size(0)

            # ── Forward pass (with optional AMP) ────────────────────
            with torch.amp.autocast(
                device_type=self.device.type, enabled=self.use_amp,
            ):
                # Unpack all three outputs from the wrapper
                projected_features, emotion_logits, identity_logits = (
                    self.model(tensors)
                )

                # ── Compute individual losses ───────────────────────
                supcon_loss, emotion_loss, identity_loss = (
                    self._compute_losses(
                        projected_features,
                        emotion_logits,
                        identity_logits,
                        emotion_labels,
                        subject_labels,
                    )
                )

                # ── Combined loss (Scaled for Accumulation) ─────────
                # Total = (β × SupCon) + Emotion + (α × Identity)
                # Then divide by accumulation_steps for gradient averaging
                total_loss = (
                    self.supcon_loss_weight * supcon_loss
                    + emotion_loss
                    + self.identity_loss_weight * identity_loss
                ) / self.accumulation_steps

            # ── Backward pass ───────────────────────────────────────
            self._scaler.scale(total_loss).backward()

            # ── Optimizer Step (Accumulated) ────────────────────────
            self._maybe_step_optimizer(batch_idx)

            # ── Accumulate metrics ──────────────────────────────────
            # Multiply total_loss back by accumulation_steps so
            # the logged metric reflects the true unscaled loss.
            total_loss_sum += (total_loss.item() * self.accumulation_steps) * batch_size
            supcon_loss_sum += supcon_loss.item() * batch_size
            emotion_loss_sum += emotion_loss.item() * batch_size
            identity_loss_sum += identity_loss.item() * batch_size

            emotion_preds = emotion_logits.argmax(dim=1)
            identity_preds = identity_logits.argmax(dim=1)

            emotion_correct += (emotion_preds == emotion_labels).sum().item()
            identity_correct += (identity_preds == subject_labels).sum().item()
            total_samples += batch_size

            # ── Periodic batch logging ──────────────────────────────
            if (batch_idx + 1) % max(1, len(self.train_loader) // 5) == 0:
                self._log.debug(
                    "  Epoch %d | Batch %d/%d | Loss: %.4f "
                    "(SC: %.4f, Emo: %.4f, ID: %.4f)",
                    epoch + 1, batch_idx + 1, len(self.train_loader),
                    total_loss.item() * self.accumulation_steps,
                    supcon_loss.item(), emotion_loss.item(),
                    identity_loss.item(),
                )

        # ── Compute epoch averages ──────────────────────────────────
        n = max(total_samples, 1)
        metrics = {
            "total_loss": total_loss_sum / n,
            "supcon_loss": supcon_loss_sum / n,
            "emotion_loss": emotion_loss_sum / n,
            "identity_loss": identity_loss_sum / n,
            "emotion_acc": emotion_correct / n,
            "identity_acc": identity_correct / n,
        }

        return metrics

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch (no gradient computation).

        Returns
        -------
        dict
            Metrics: total_loss, supcon_loss, emotion_loss, identity_loss,
            emotion_acc, identity_acc.
        """
        self.model.eval()

        total_loss_sum = 0.0
        supcon_loss_sum = 0.0
        emotion_loss_sum = 0.0
        identity_loss_sum = 0.0
        emotion_correct = 0
        identity_correct = 0
        total_samples = 0

        for tensors, emotion_labels, subject_labels in self.val_loader:
            tensors = tensors.to(self.device)
            emotion_labels = emotion_labels.to(self.device).long()
            subject_labels = subject_labels.to(self.device).long()

            batch_size = tensors.size(0)

            with torch.amp.autocast(
                device_type=self.device.type, enabled=self.use_amp,
            ):
                projected_features, emotion_logits, identity_logits = (
                    self.model(tensors)
                )

                supcon_loss, emotion_loss, identity_loss = (
                    self._compute_losses(
                        projected_features,
                        emotion_logits,
                        identity_logits,
                        emotion_labels,
                        subject_labels,
                    )
                )

                total_loss = (
                    self.supcon_loss_weight * supcon_loss
                    + emotion_loss
                    + self.identity_loss_weight * identity_loss
                )

            total_loss_sum += total_loss.item() * batch_size
            supcon_loss_sum += supcon_loss.item() * batch_size
            emotion_loss_sum += emotion_loss.item() * batch_size
            identity_loss_sum += identity_loss.item() * batch_size

            emotion_preds = emotion_logits.argmax(dim=1)
            identity_preds = identity_logits.argmax(dim=1)

            emotion_correct += (emotion_preds == emotion_labels).sum().item()
            identity_correct += (identity_preds == subject_labels).sum().item()
            total_samples += batch_size

        n = max(total_samples, 1)
        metrics = {
            "total_loss": total_loss_sum / n,
            "supcon_loss": supcon_loss_sum / n,
            "emotion_loss": emotion_loss_sum / n,
            "identity_loss": identity_loss_sum / n,
            "emotion_acc": emotion_correct / n,
            "identity_acc": identity_correct / n,
        }

        return metrics

    # ─────────────────────────────────────────────────────────────────
    #  Private Helper Methods
    # ─────────────────────────────────────────────────────────────────

    def _compute_losses(
        self,
        projected_features: torch.Tensor,
        emotion_logits: torch.Tensor,
        identity_logits: torch.Tensor,
        emotion_labels: torch.Tensor,
        subject_labels: torch.Tensor,
    ) -> tuple:
        """
        Compute the three individual loss terms.

        Parameters
        ----------
        projected_features : torch.Tensor
            L2-normalized projections of shape ``[B, proj_dim]``.
        emotion_logits : torch.Tensor
            Emotion predictions of shape ``[B, num_emotions]``.
        identity_logits : torch.Tensor
            Identity predictions of shape ``[B, num_subjects]``.
        emotion_labels : torch.Tensor
            Ground-truth emotion labels of shape ``[B]``.
        subject_labels : torch.Tensor
            Ground-truth subject labels of shape ``[B]``.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor, torch.Tensor)
            ``(supcon_loss, emotion_loss, identity_loss)``
        """
        supcon_loss = self.supcon_criterion(
            projected_features, emotion_labels,
        )
        emotion_loss = self.emotion_criterion(
            emotion_logits, emotion_labels,
        )
        identity_loss = self.identity_criterion(
            identity_logits, subject_labels,
        )
        return supcon_loss, emotion_loss, identity_loss

    def _maybe_step_optimizer(self, batch_idx: int) -> None:
        """
        Perform an optimizer step if we've accumulated enough gradients.

        Steps when:
            - ``(batch_idx + 1) % accumulation_steps == 0``, or
            - We've reached the last batch of the epoch.

        Also handles gradient clipping and AMP scaler updates.

        Parameters
        ----------
        batch_idx : int
            Current batch index (0-indexed).
        """
        is_accumulation_boundary = (
            (batch_idx + 1) % self.accumulation_steps == 0
        )
        is_last_batch = (batch_idx + 1) == len(self.train_loader)

        if is_accumulation_boundary or is_last_batch:
            # ── Gradient clipping ───────────────────────────────
            if self.gradient_clip_norm is not None:
                self._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_norm,
                )

            self._scaler.step(self.optimizer)
            self._scaler.update()
            self.optimizer.zero_grad()  # Reset gradients ONLY after stepping

    def _record_history(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        grl_lambda: float,
        current_lr: float,
    ) -> None:
        """
        Append epoch metrics to the history dictionary.

        Parameters
        ----------
        train_metrics : dict
            Training metrics for the epoch.
        val_metrics : dict
            Validation metrics (may be empty if no val_loader).
        grl_lambda : float
            Current GRL lambda value.
        current_lr : float
            Current learning rate.
        """
        self._history["train_loss"].append(train_metrics["total_loss"])
        self._history["train_supcon_loss"].append(train_metrics["supcon_loss"])
        self._history["train_emotion_loss"].append(train_metrics["emotion_loss"])
        self._history["train_identity_loss"].append(train_metrics["identity_loss"])
        self._history["train_emotion_acc"].append(train_metrics["emotion_acc"])
        self._history["train_identity_acc"].append(train_metrics["identity_acc"])
        self._history["grl_lambda"].append(grl_lambda)
        self._history["lr"].append(current_lr)

        if val_metrics:
            self._history["val_loss"].append(val_metrics.get("total_loss", 0))
            self._history["val_supcon_loss"].append(val_metrics.get("supcon_loss", 0))
            self._history["val_emotion_acc"].append(val_metrics.get("emotion_acc", 0))
            self._history["val_identity_acc"].append(val_metrics.get("identity_acc", 0))

    # ─────────────────────────────────────────────────────────────────
    #  Checkpointing: Save & Resume
    # ─────────────────────────────────────────────────────────────────

    def _save_latest_checkpoint(self, epoch: int) -> None:
        """
        Save a comprehensive checkpoint for crash recovery.

        The checkpoint includes:
            - Model state dict
            - Optimizer state dict
            - Scheduler state dict (if exists)
            - AMP scaler state dict
            - Current epoch
            - Best validation accuracy
            - Full training history

        Atomicity: write to .tmp then os.replace().
        """
        checkpoint = {
            "epoch": epoch + 1,  # Next epoch to run
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self._scaler.state_dict(),
            "best_val_accuracy": self._best_val_accuracy,
            "history": self._history,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # ── Atomic save ─────────────────────────────────────────────
        tmp_path = self._latest_ckpt_path.with_suffix(".tmp")
        torch.save(checkpoint, str(tmp_path))
        os.replace(str(tmp_path), str(self._latest_ckpt_path))

        self._log.debug(
            "Checkpoint saved: epoch=%d → %s", epoch + 1, self._latest_ckpt_path
        )

    def _save_best_model(self, epoch: int, val_acc: float) -> None:
        """
        Save the best model (by validation accuracy).

        Parameters
        ----------
        epoch : int
            Epoch number.
        val_acc : float
            Validation accuracy that triggered the save.
        """
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "best_val_accuracy": val_acc,
        }

        tmp_path = self._best_ckpt_path.with_suffix(".tmp")
        torch.save(checkpoint, str(tmp_path))
        os.replace(str(tmp_path), str(self._best_ckpt_path))

        self._log.info(
            "★ New best model saved: epoch=%d, val_acc=%.4f → %s",
            epoch + 1, val_acc, self._best_ckpt_path,
        )

    def _try_resume(self) -> None:
        """
        Attempt to resume training from ``latest_checkpoint.pth``.

        If the checkpoint file exists, loads all state and sets
        ``self._start_epoch`` so training continues from the correct
        epoch.
        """
        if not self._latest_ckpt_path.exists():
            self._log.info("No checkpoint found — starting fresh training.")
            return

        self._log.info("=" * 70)
        self._log.info("  RESUMING from checkpoint: %s", self._latest_ckpt_path)
        self._log.info("=" * 70)

        checkpoint = torch.load(
            str(self._latest_ckpt_path),
            map_location=self.device,
            weights_only=False,
        )

        # ── Restore model ──────────────────────────────────────────
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self._log.info("  Model state restored.")

        # ── Restore optimizer ──────────────────────────────────────
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._log.info("  Optimizer state restored.")

        # ── Restore scheduler ──────────────────────────────────────
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self._log.info("  Scheduler state restored.")

        # ── Restore AMP scaler ─────────────────────────────────────
        if "scaler_state_dict" in checkpoint:
            self._scaler.load_state_dict(checkpoint["scaler_state_dict"])
            self._log.info("  AMP scaler state restored.")

        # ── Restore training state ─────────────────────────────────
        self._start_epoch = checkpoint["epoch"]
        self._best_val_accuracy = checkpoint.get("best_val_accuracy", 0.0)
        self._history = checkpoint.get("history", self._history)

        self._log.info(
            "  Resuming from epoch %d (best_val_acc=%.4f)",
            self._start_epoch + 1, self._best_val_accuracy,
        )

    # ─────────────────────────────────────────────────────────────────
    #  Logging Utilities
    # ─────────────────────────────────────────────────────────────────

    def _log_epoch_summary(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        grl_lambda: float,
        lr: float,
        epoch_time: float,
    ) -> None:
        """
        Log a formatted summary of one epoch's results.

        Explicitly prints the SupCon loss alongside Emotion and Identity
        losses so the clustering behavior can be monitored during training.

        Parameters
        ----------
        epoch : int
            Current epoch (0-indexed).
        train_metrics : dict
            Training metrics for the epoch.
        val_metrics : dict
            Validation metrics (may be empty).
        grl_lambda : float
            Current GRL lambda value.
        lr : float
            Current learning rate.
        epoch_time : float
            Wall-clock time for this epoch in seconds.
        """
        self._log.info("─" * 70)
        self._log.info(
            "Epoch %d/%d  [%.1fs]  │  LR=%.6f  │  λ_GRL=%.4f",
            epoch + 1, self.num_epochs, epoch_time, lr, grl_lambda,
        )
        self._log.info(
            "  Train │ Loss: %.4f  "
            "(SC: %.4f × %.1f + Emo: %.4f + %.2f × ID: %.4f)",
            train_metrics["total_loss"],
            train_metrics["supcon_loss"],
            self.supcon_loss_weight,
            train_metrics["emotion_loss"],
            self.identity_loss_weight,
            train_metrics["identity_loss"],
        )
        self._log.info(
            "  Train │ EmoAcc: %.4f  │ IDAcc: %.4f",
            train_metrics["emotion_acc"],
            train_metrics["identity_acc"],
        )
        if val_metrics:
            self._log.info(
                "  Val   │ Loss: %.4f  (SC: %.4f)  "
                "│ EmoAcc: %.4f  │ IDAcc: %.4f",
                val_metrics["total_loss"],
                val_metrics.get("supcon_loss", 0.0),
                val_metrics["emotion_acc"],
                val_metrics["identity_acc"],
            )
        self._log.info("─" * 70)
