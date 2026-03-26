# -*- coding: utf-8 -*-
"""METrainer — Training loop, validation, and checkpoint management.

Updated to robustly handle any model architecture and provide 7-class metrics.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class METrainer:
    """Encapsulates training, validation, and checkpointing for MER."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        device: torch.device = torch.device("cpu"),
        checkpoint_dir: str | Path = "checkpoints",
        grad_clip_norm: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
        patience: int = 0,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.grad_clip_norm = grad_clip_norm
        self.patience = patience

        self.model.to(self.device)

        # Loss function with optional class weights for imbalance (e.g., Fear)
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.best_val_acc: float = 0.0
        self.epochs_no_improve: int = 0
        
        self.history: Dict[str, list[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
        }

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # --- ROBUST CLASS DETECTION ---
        # Automatically find the number of classes from the model's last Linear layer
        self.num_classes = 0
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                self.num_classes = module.out_features
        logger.info(f"Trainer initialized for {self.num_classes} classes.")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        running_loss, num_correct, num_samples = 0.0, 0, 0

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            loss.backward()
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            predictions = torch.argmax(outputs, dim=1)
            num_correct += (predictions == targets).sum().item()
            num_samples += batch_size

        return {"train_loss": running_loss/max(num_samples, 1), "train_acc": num_correct/max(num_samples, 1)}

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        running_loss, num_correct, num_samples = 0.0, 0, 0
        all_targets, all_preds = [], []

        for inputs, targets in self.val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            predictions = torch.argmax(logits, dim=1)
            num_correct += (predictions == targets).sum().item()
            num_samples += batch_size

            all_targets.extend(targets.cpu().tolist())
            all_preds.extend(predictions.cpu().tolist())

        epoch_loss = running_loss / max(num_samples, 1)
        epoch_acc = num_correct / max(num_samples, 1)

        logger.info(f"[Epoch {epoch}] VAL — loss: {epoch_loss:.4f} | acc: {epoch_acc:.4f}")

        if all_targets:
            # Per-class logging
            t_counts = Counter(all_targets)
            c_counts = Counter([t for t, p in zip(all_targets, all_preds) if t == p])
            per_class = [f"Class {c}: {c_counts[c]}/{t_counts[c]} ({c_counts[c]/t_counts[c]*100:.1f}%)" 
                         for c in sorted(t_counts.keys())]
            logger.info(f"VAL per-class → {' | '.join(per_class)}")

            # --- Confusion Matrix (The Fix) ---
            # Use the num_classes detected in __init__
            n = self.num_classes
            cm = torch.zeros(n, n, dtype=torch.int32)
            for t, p in zip(all_targets, all_preds):
                if t < n and p < n: # Safety check for indices
                    cm[t, p] += 1
            
            logger.info("--- Confusion Matrix (T=True, P=Pred) ---")
            header = "      " + "".join([f"P{i:<2} " for i in range(n)])
            logger.info(header)
            for i in range(n):
                row = f"T{i:<2} | " + "".join([f"{cm[i, j]:<2}  " for j in range(n)])
                logger.info(row)

        return {"val_loss": epoch_loss, "val_acc": epoch_acc}

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        state = {
            "epoch": epoch, "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc, "history": self.history,
        }
        if self.scheduler: state["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(state, self.checkpoint_dir / "last_checkpoint.pth")
        if is_best:
            torch.save(state, self.checkpoint_dir / "best_model.pth")
            logger.info(f"★ Best model updated (val_acc={self.best_val_acc:.4f})")

    def load_checkpoint(self, path: str | Path) -> int:
        ckpt = torch.load(str(path), map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.best_val_acc = ckpt.get("best_val_acc", 0.0)
        return ckpt["epoch"] + 1

    def train(self, num_epochs: int, start_epoch: int = 1) -> None:
        logger.info("=" * 60)
        logger.info(f"Training Start: Epoch {start_epoch} to {num_epochs} on {self.device}")
        logger.info("=" * 60)

        for epoch in range(start_epoch, num_epochs + 1):
            epoch_start = time.time()
            train_m = self.train_epoch(epoch)
            val_m = self.validate(epoch)

            self.history["train_loss"].append(train_m["train_loss"])
            self.history["train_acc"].append(train_m["train_acc"])
            self.history["val_loss"].append(val_m["val_loss"])
            self.history["val_acc"].append(val_m["val_acc"])

            if self.scheduler: self.scheduler.step()

            is_best = val_m["val_acc"] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_m["val_acc"]
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            self.save_checkpoint(epoch, is_best)
            if self.patience > 0 and self.epochs_no_improve >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            logger.info(f"Epoch {epoch} finished in {time.time() - epoch_start:.1f}s")