# -*- coding: utf-8 -*-
"""MERManager — Orchestrator for STSTNet training and Cross-Dataset Evaluation.

Supports standard training on CASME II and 'Evaluation Only' mode for 
validating robustness on CAS(ME)^2.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Fix pathing so 'src' can be found
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import argparse
import logging
import random
from logging.handlers import RotatingFileHandler
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src.stage2_training.dataset import MERDataset
from src.stage2_training.model import STSTNet
from src.stage2_training.trainer import METrainer

# --- Constants ---
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_LOG_DATE_FMT = "%Y-%m-%d %H:%M:%S"
_LOG_MAX_BYTES = 10 * 1024 * 1024
_LOG_BACKUP_COUNT = 5


class MERManager:
    """Orchestrator for training and cross-dataset validation."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.data_dir = Path(args.data_dir).resolve()
        
        run_name = getattr(args, "run_name", "default_run")
        self.checkpoint_dir = Path(args.checkpoint_dir).resolve() / run_name
        self.log_dir = Path(args.log_dir).resolve() / run_name
        
        self.label_csv: Optional[Path] = (
            Path(args.label_csv).resolve() if args.label_csv else None
        )

        self.logger: logging.Logger = logging.getLogger("MER")
        self.device: torch.device = torch.device("cpu")
        self.model: Optional[STSTNet] = None
        self.trainer: Optional[METrainer] = None

    def setup_logging(self) -> None:
        """Dual-stream logging setup."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.log_dir / "training.log"
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.handlers.clear()

        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FMT))
        root_logger.addHandler(console)

        try:
            file_h = RotatingFileHandler(str(log_file), maxBytes=_LOG_MAX_BYTES, 
                                        backupCount=_LOG_BACKUP_COUNT, encoding="utf-8")
            file_h.setLevel(logging.DEBUG)
            file_h.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FMT))
            root_logger.addHandler(file_h)
            self.logger.info("Logging to file: %s", log_file)
        except OSError as e:
            self.logger.warning("Log file creation failed: %s", e)

    def _select_device(self) -> torch.device:
        if torch.cuda.is_available():
            self.logger.info("CUDA available — using GPU: %s", torch.cuda.get_device_name(0))
            return torch.device("cuda")
        return torch.device("cpu")

    def _compute_train_stats(self, train_dataset: torch.utils.data.Subset) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean/std from train split."""
        self.logger.info("Computing Normalization Stats from train_dataset...")
        loader = DataLoader(train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=False)
        channel_sum, channel_sq_sum, total_pixels = torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64), 0
        
        for x, _ in loader:
            b, c, t, h, w = x.shape
            x_d = x.double()
            channel_sum += x_d.sum(dim=[0, 2, 3, 4])
            channel_sq_sum += (x_d ** 2).sum(dim=[0, 2, 3, 4])
            total_pixels += b * t * h * w
            
        mean = channel_sum / total_pixels
        std = torch.sqrt(torch.clamp((channel_sq_sum / total_pixels) - (mean ** 2), min=1e-8))
        return mean.float().view(3, 1, 1, 1), std.float().view(3, 1, 1, 1)

    def prepare_data(self) -> Tuple[DataLoader, DataLoader, MERDataset]:
        """Dataset prep. If eval_only, splits are ignored and full_ds is used for val."""
        full_ds = MERDataset(data_dir=self.data_dir, label_csv=self.label_csv, normalize=False)
        if len(full_ds) == 0: raise RuntimeError(f"No samples found in {self.data_dir}")

        gen = torch.Generator().manual_seed(self.args.seed)
        
        if self.args.eval_only:
            # In Evaluation mode, we treat the entire target dataset as a validation set
            self.logger.info("Evaluation Mode: Using entire dataset for validation.")
            # Dummy split to keep variables consistent
            train_ds, val_ds = full_ds, full_ds
            # Use default stats (usually 0 mean, 1 std) or compute from data
            t_mean, t_std = self._compute_train_stats(random_split(full_ds, [len(full_ds), 0], generator=gen)[0])
        else:
            train_sz = int(0.8 * len(full_ds))
            val_sz = len(full_ds) - train_sz
            train_ds, val_ds = random_split(full_ds, [train_sz, val_sz], generator=gen)
            t_mean, t_std = self._compute_train_stats(train_ds)

        for ds in [train_ds, val_ds]:
            ds.dataset.mean, ds.dataset.std, ds.dataset.normalize = t_mean, t_std, True

        tr_ldr = DataLoader(train_ds, batch_size=self.args.batch_size, shuffle=True, 
                            num_workers=self.args.num_workers, pin_memory=True, generator=gen)
        vl_ldr = DataLoader(val_ds, batch_size=self.args.batch_size, shuffle=False, 
                            num_workers=self.args.num_workers, pin_memory=True)
        return tr_ldr, vl_ldr, full_ds

    def _set_global_seed(self, seed: int) -> None:
        import os
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = True, False
        torch.use_deterministic_algorithms(False) 
        self.logger.info("Global seed set to %d (Performance Mode: ON)", seed)

    def run(self) -> None:
        self.setup_logging()
        self.logger.info("=" * 60)
        self.logger.info(f"  MER Pipeline Stage 2 — {'EVALUATION' if self.args.eval_only else 'TRAINING'}")
        self.logger.info("=" * 60)

        self._set_global_seed(self.args.seed)
        self.device = self._select_device()
        
        tr_ldr, vl_ldr, full_ds = self.prepare_data()
        cw = full_ds.get_class_weights().to(self.device)
        self._log_config(cw)

        self.model = STSTNet(num_classes=self.args.num_classes, dropout_p=self.args.dropout).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        sch = torch.optim.lr_scheduler.StepLR(opt, self.args.sch_step, self.args.sch_gamma) if self.args.use_scheduler else None

        self.trainer = METrainer(
            model=self.model, train_loader=tr_ldr, val_loader=vl_ldr,
            optimizer=opt, scheduler=sch, device=self.device,
            checkpoint_dir=self.checkpoint_dir, grad_clip_norm=self.args.grad_clip,
            class_weights=cw, patience=self.args.patience,
        )

        # Load weights
        start_epoch = 1
        ckpt_path = self.checkpoint_dir / "best_model.pth" if self.args.eval_only else self.checkpoint_dir / "last_checkpoint.pth"
        
        if ckpt_path.exists() and self.args.resume:
            start_epoch = self.trainer.load_checkpoint(ckpt_path)
        elif self.args.eval_only:
            raise FileNotFoundError(f"Evaluation requires a pre-trained model at {ckpt_path}")

        # --- 7. EVALUATION MODE ---
        if self.args.eval_only:
            self.logger.info("Starting Cross-Dataset Evaluation on target data...")
            metrics = self.trainer.validate(epoch=start_epoch)
            self.logger.info(f"Final Results on {self.data_dir.name} -> Acc: {metrics['acc']:.4f}, Loss: {metrics['loss']:.4f}")
            self.logger.info("Evaluation complete. Exiting.")
            return

        # --- 8. TRAINING MODE ---
        self.logger.info("Performing pipeline guardrail dry-run...")
        full_ds.dry_run(batch_size=self.args.batch_size)

        try:
            self.trainer.train(num_epochs=self.args.epochs, start_epoch=start_epoch)
            self.logger.info("Pipeline finished successfully.")
        except KeyboardInterrupt:
            self.logger.warning("Training interrupted.")

    def _log_config(self, cw: torch.Tensor) -> None:
        self.logger.info("  run_name       : %s", getattr(self.args, "run_name", "default_run"))
        self.logger.info("  epochs         : %d", self.args.epochs)
        self.logger.info("  batch_size     : %d", self.args.batch_size)
        self.logger.info("  class_weights  : %s", np.array2string(cw.cpu().numpy(), precision=4, separator=', '))
        self.logger.info("=" * 60)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="STSTNet Training & Eval", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir", type=str, required=True, help="Path to features (.npy)")
    p.add_argument("--label_csv", type=str, default=None, help="Path to labels (.csv)")
    p.add_argument("--run_name", type=str, default="default_run", help="Exp name to save/load from")
    p.add_argument("--eval_only", action="store_true", help="Skip training, just evaluate on data_dir")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--log_dir", type=str, default="logs")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--use_scheduler", action="store_true", default=False)
    p.add_argument("--sch_step", type=int, default=50)
    p.add_argument("--sch_gamma", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=20)
    return p.parse_args()


if __name__ == "__main__":
    manager = MERManager(_parse_args())
    manager.run()