"""Master Execution Controller validating the comprehensive MTMNet architecture.

Handles GRL state updates, Transfer Priorities over Teacher logic, dynamic
parameter optimisation, AMP integration for RTX 4080 Tensor Cores, VRAM OOM
hardening, and structured global UAR/UF1 extractions.

Return-Value Synchronisation
----------------------------
Returns a dictionary containing ``fold_preds``, ``fold_labels``, and
``fold_subjects`` tensors to guarantee 100% data alignment with the
``MERMetricTracker`` consumed by the ablation controller.

Author  : Addhyan Pant
Project : Master's Thesis — Micro-Expression Recognition
"""

import copy
import gc
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch.optim import AdamW, SGD
from torch.cuda.amp import autocast, GradScaler

from src.evaluation.metrics import MERMetricTracker
from src.evaluation.loso_handler import LOSOCrossValidator
from src.models.mtm_net import MTMNet
from src.losses.transfer_loss import MTMTransferLoss

LOG_FMT = "%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger("mer.loso")


def run_loso_experiment(
    student_model: MTMNet,
    teacher_model_path: str,
    transfer_loss_fn: MTMTransferLoss,
    loso_validator: LOSOCrossValidator,
    num_classes: int,
    epochs: int,
    device: torch.device,
    save_dir: str = "./checkpoints",
    batch_size: int = 16,
    num_workers: int = 4,
    max_folds: int = None,
    strict_load: bool = True,
    use_amp: bool = True,
    use_compile: bool = False,
) -> Dict[str, Any]:
    """Master routine traversing independent LOSO validation folds.

    Parameters
    ----------
    student_model : MTMNet
        Uniformly randomised initialisation architecture.
    teacher_model_path : str
        File system path to the MacroNet Teacher checkpoint.
    transfer_loss_fn : MTMTransferLoss
        Inequality Regularisation handler integrating Joint Focal-Center logic.
    loso_validator : LOSOCrossValidator
        Generator yielding disjoint subset DataLoaders.
    num_classes : int
        Number of emotion classes.
    epochs : int
        Training iterations per fold.
    device : torch.device
        Target computation device.
    save_dir : str
        Checkpoint storage directory.
    batch_size : int
        DataLoader batch size.
    num_workers : int
        DataLoader worker processes (0 for Docker safety).
    max_folds : int, optional
        Limit the number of folds for debugging.
    strict_load : bool
        Whether to use strict state_dict loading for the teacher.
    use_amp : bool
        Enable Automatic Mixed Precision (RTX 4080 4th-Gen Tensor Cores).
    use_compile : bool
        Apply torch.compile() for Ada Lovelace kernel optimisation.

    Returns
    -------
    dict
        Contains:
        - ``fold_preds``   : torch.Tensor — all predictions concatenated.
        - ``fold_labels``  : torch.Tensor — all ground-truth labels concatenated.
        - ``fold_subjects``: list         — per-sample subject identifiers.
        - ``Global_UF1``   : float
        - ``Global_UAR``   : float
        - ``UAR_std``      : float
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # === 1. Establish the Teacher Prior ===
    logger.info(f"Establishing MacroNet prior from {teacher_model_path}")
    teacher_model = copy.deepcopy(student_model)
    if Path(teacher_model_path).exists():
        teacher_model.load_state_dict(
            torch.load(teacher_model_path, map_location=device), strict=strict_load
        )
    else:
        logger.warning(f"Teacher checkpoint not found: {teacher_model_path}. Using random init.")
    teacher_model.to(device)
    teacher_model.freeze_as_teacher()
    teacher_model.eval()

    # === 2. Secure Initialisation Checkpointing ===
    student_model.to(device)
    transfer_loss_fn.to(device)

    # Optional torch.compile for Ada Lovelace
    if use_compile and hasattr(torch, "compile"):
        logger.info("Applying torch.compile() to student model...")
        student_model = torch.compile(student_model)

    logger.info("Deep-copying initial zero-state to prevent inter-fold identity leakage.")
    initial_state = copy.deepcopy(student_model.state_dict())
    initial_loss_state = copy.deepcopy(transfer_loss_fn.state_dict())

    # AMP infrastructure
    scaler = GradScaler(enabled=use_amp)
    logger.info(f"AMP {'ENABLED' if use_amp else 'DISABLED'} | "
                f"torch.compile {'ENABLED' if use_compile else 'DISABLED'}")

    # Global evaluation state
    metric_tracker = MERMetricTracker(num_classes=num_classes)
    best_record_uar = 0.0

    # Collect all predictions for return-value synchronisation
    all_fold_preds = []
    all_fold_labels = []
    all_fold_subjects = []

    # === 3. Primary Fold Traversal ===
    for fold_idx, test_subject, train_loader, test_loader in loso_validator.get_folds(batch_size, num_workers):
        if max_folds is not None and fold_idx >= max_folds:
            logger.info(f"Fast mode: halting after {max_folds} folds.")
            break

        logger.info(f"\n--- FOLD {fold_idx + 1}: Leaving out Subject '{test_subject}' ---")

        try:
            # Reset to clean zero-state
            student_model.load_state_dict(initial_state)
            transfer_loss_fn.load_state_dict(initial_loss_state)

            # Dual-Optimiser Construction
            optimizer_model = AdamW(student_model.parameters(), lr=1e-4, weight_decay=1e-4)
            optimizer_centers = SGD(transfer_loss_fn.joint_loss.parameters(), lr=0.1)

            # --- Training Loop ---
            for epoch in range(epochs):
                student_model.train()
                running_loss = 0.0

                for videos, labels, *extra in train_loader:
                    videos, labels = videos.to(device), labels.to(device)
                    lengths = torch.full(
                        (videos.size(0),), videos.size(1), dtype=torch.long, device=device
                    )

                    optimizer_model.zero_grad(set_to_none=True)
                    optimizer_centers.zero_grad(set_to_none=True)

                    with autocast(enabled=use_amp):
                        out_micro = student_model(videos, lengths=lengths)
                        with torch.no_grad():
                            out_macro = teacher_model(videos, lengths=lengths)

                        loss_dict = transfer_loss_fn(
                            micro_logits=out_micro["class_logits"],
                            micro_features_raw=out_micro["features_raw"],
                            micro_features_proj=out_micro["features_proj"],
                            micro_labels=labels,
                            micro_domain_logits=out_micro["domain_logits"],
                            macro_logits=out_macro["class_logits"],
                            macro_features_raw=out_macro["features_raw"],
                            macro_features_proj=out_macro["features_proj"],
                            macro_labels=labels,
                            macro_domain_logits=out_macro["domain_logits"],
                        )
                        loss = loss_dict["Total_Loss"]

                    scaler.scale(loss).backward()

                    # Unscale before gradient manipulation
                    scaler.unscale_(optimizer_model)
                    scaler.unscale_(optimizer_centers)

                    # Center Loss gradient scaling
                    for param in transfer_loss_fn.joint_loss.parameters():
                        if param.grad is not None:
                            param.grad.data *= 1.0 / max(transfer_loss_fn.lir_alpha, 1e-8)

                    scaler.step(optimizer_model)
                    scaler.step(optimizer_centers)
                    scaler.update()

                    running_loss += loss.item()

                avg_loss = running_loss / max(len(train_loader), 1)
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    logger.info(f"  -> Epoch [{epoch+1}/{epochs}] - Avg Transfer Loss: {avg_loss:.4f}")

            # === 4. Mathematical Validation & Accumulation ===
            student_model.eval()
            fold_preds_list = []
            fold_labels_list = []
            fold_subjects_list = []

            with torch.no_grad():
                for batch_data in test_loader:
                    # Handle variable-length batch unpacking
                    if isinstance(batch_data, (list, tuple)):
                        videos = batch_data[0]
                        labels = batch_data[1]
                        subjects_batch = batch_data[2] if len(batch_data) >= 3 else [test_subject] * labels.size(0)
                    else:
                        videos = batch_data
                        labels = None
                        subjects_batch = [test_subject]

                    videos = videos.to(device)
                    if labels is not None:
                        labels = labels.to(device)
                    lengths = torch.full(
                        (videos.size(0),), videos.size(1), dtype=torch.long, device=device
                    )

                    with autocast(enabled=use_amp):
                        out = student_model(videos, lengths=lengths)
                    preds = torch.argmax(out["class_logits"], dim=1)

                    fold_preds_list.append(preds.cpu())
                    if labels is not None:
                        fold_labels_list.append(labels.cpu())
                        metric_tracker.update(labels, preds)

                    # Subject tracking
                    if isinstance(subjects_batch, torch.Tensor):
                        subjects_batch = subjects_batch.tolist()
                    elif isinstance(subjects_batch, str):
                        subjects_batch = [subjects_batch] * preds.size(0)
                    fold_subjects_list.extend(subjects_batch)

            # Concatenate fold results
            fold_preds_tensor = torch.cat(fold_preds_list) if fold_preds_list else torch.tensor([])
            fold_labels_tensor = torch.cat(fold_labels_list) if fold_labels_list else torch.tensor([])

            all_fold_preds.append(fold_preds_tensor)
            all_fold_labels.append(fold_labels_tensor)
            all_fold_subjects.extend(fold_subjects_list)

            # Fold metrics
            fold_uf1, fold_uar = metric_tracker.compute_fold_metrics()
            logger.info(f">> Fold {fold_idx + 1} [{test_subject}] — UAR: {fold_uar:.4f} | UF1: {fold_uf1:.4f}")

            # === 5. Intelligent Checkpointing ===
            if fold_uar > best_record_uar:
                best_record_uar = fold_uar
                ckpt_path = save_path / "best_localized_mtmnet.pth"
                torch.save(student_model.state_dict(), ckpt_path)
                logger.info(f">> Best model saved to '{ckpt_path}'")

        except RuntimeError as e:
            # ──────────────────────────────────────────────────
            # VRAM OOM Hardening: Catch CUDA OOM gracefully
            # ──────────────────────────────────────────────────
            if "out of memory" in str(e).lower():
                logger.error(
                    f"CUDA OOM on Fold {fold_idx + 1} (Subject '{test_subject}'). "
                    f"Suggestion: reduce --batch_size below {batch_size}. "
                    f"Clearing VRAM and continuing to next fold."
                )
                # Move everything to CPU to free VRAM
                try:
                    student_model.cpu()
                    transfer_loss_fn.cpu()
                    teacher_model.cpu()
                except Exception:
                    pass
                torch.cuda.empty_cache()
                gc.collect()
                # Restore to device for next fold
                student_model.to(device)
                transfer_loss_fn.to(device)
                teacher_model.to(device)
                continue
            else:
                raise

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # === 6. Global Metrics ===
    global_metrics = metric_tracker.compute_global_metrics()
    logger.info("=======================================================")
    logger.info("       FINAL COMPOSITE DATABASE EVALUATION SUMMARY     ")
    logger.info("=======================================================")
    logger.info(f" Global Aggregate UF1       : {global_metrics['Global_UF1']:.4f}")
    logger.info(f" Global Aggregate UAR       : {global_metrics['Global_UAR']:.4f}")
    logger.info(f" Topological Variance (σ)   : {global_metrics['UAR_std']:.4f}")
    logger.info("=======================================================\n")

    # === Return-Value Synchronisation ===
    combined_preds = torch.cat(all_fold_preds) if all_fold_preds else torch.tensor([])
    combined_labels = torch.cat(all_fold_labels) if all_fold_labels else torch.tensor([])

    return {
        "fold_preds": combined_preds,
        "fold_labels": combined_labels,
        "fold_subjects": all_fold_subjects,
        **global_metrics,
    }
