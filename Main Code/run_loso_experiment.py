"""Master Execution Controller validating the comprehensive MTMNet architecture.

Handles GRL state updates, Transfer Priorities over Teacher logic, dynamic 
parameter optimization, and structured global UAR/UF1 extractions seamlessly.
"""
import copy
import logging
import os
import torch
from torch.optim import AdamW, SGD

# Note: Adjust relative namespace imports pointing equivalently based on layout.
from src.evaluation.metrics import MERMetricTracker
from src.evaluation.loso_handler import LOSOCrossValidator
from src.models.mtm_net import MTMNet
from src.losses.transfer_loss import MTMTransferLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


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
    num_workers: int = 4
) -> dict:
    """Master routine traversing independent LOSO validation folds.

    Parameters
    ----------
    student_model : MTMNet
        Uniformly randomized initialization architecture tracking the Micro mapping domain.
    teacher_model_path : str
        Global file system path indicating the MacroNet mapping (Teacher).
    transfer_loss_fn : MTMTransferLoss
        Inequality Regularization handler integrating Joint Focal-Center logic.
    loso_validator : LOSOCrossValidator
        Generator yielding systematically disjoint subset dataloaders.
    num_classes : int
        Underlying topological distribution classes.
    epochs : int
        Iterations designated per dynamically bounded fold subset.
    device : torch.device
        Target operational computation layer.
    save_dir : str
        Write checkpoint storage destination.
    batch_size : int
        Sequence packaging size.
    num_workers : int
        CPU core multiprocessing extraction threads.
    """
    os.makedirs(save_dir, exist_ok=True)

    # === 1. Establish the Teacher Prior ===
    logger.info(f"Establishing formal MacroNet prior from {teacher_model_path}")
    teacher_model = copy.deepcopy(student_model)
    teacher_model.load_state_dict(torch.load(teacher_model_path, map_location=device))
    teacher_model.to(device)
    teacher_model.freeze_as_teacher()
    teacher_model.eval()

    # === 2. Secure Initialization Checkpointing (CRITICAL) ===
    student_model.to(device)
    transfer_loss_fn.to(device)
    logger.info("Deep-copying Initial zero-state. Preventing inter-fold identity leakage.")
    # Statically clone memory weights capturing structural pristine properties
    initial_state = copy.deepcopy(student_model.state_dict())
    initial_loss_state = copy.deepcopy(transfer_loss_fn.state_dict())

    # Formulate global evaluation variables
    metric_tracker = MERMetricTracker(num_classes=num_classes)
    best_record_uar = 0.0

    # === 3. Primary Fold Traversal ===
    for fold_idx, test_subject, train_loader, test_loader in loso_validator.get_folds(batch_size, num_workers):
        logger.info(f"\n--- FOLD {fold_idx + 1}/{len(loso_validator.subjects)}: Segregating and leaving out Subject '{test_subject}' ---")

        # CRITICAL REINFORCEMENT: Synchronize completely back to clean zero states
        student_model.load_state_dict(initial_state)
        transfer_loss_fn.load_state_dict(initial_loss_state)

        # Dual-Optimizer Construction allocating base gradients + isolated Center gradients
        optimizer_model = AdamW(student_model.parameters(), lr=1e-4, weight_decay=1e-4)
        
        # Isolate optimizers explicitly updating internal center coordinate weights (momentum based)
        optimizer_centers = SGD(transfer_loss_fn.joint_loss.parameters(), lr=0.1)

        # Begin designated spatial fold iterations
        for epoch in range(epochs):
            student_model.train()
            running_loss = 0.0
            
            for videos, labels in train_loader:
                videos, labels = videos.to(device), labels.to(device)
                lengths = torch.full((videos.size(0),), videos.size(1), dtype=torch.long, device=device)

                optimizer_model.zero_grad()
                optimizer_centers.zero_grad()

                # Independent Domain Feature Traversals
                out_micro = student_model(videos, lengths=lengths)
                
                with torch.no_grad():
                    out_macro = teacher_model(videos, lengths=lengths)

                # Abstracted inequality and spatial metric alignment extraction
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
                    macro_domain_logits=out_macro["domain_logits"]
                )

                loss = loss_dict["Total_Loss"]
                loss.backward()

                # Step discrete variable weights corresponding to structural components
                optimizer_model.step()
                
                # Correct isolated gradients mapping specifically adjusting scaling
                for param in transfer_loss_fn.joint_loss.parameters():
                    if param.grad is not None:
                        param.grad.data *= (1.0 / transfer_loss_fn.lir_alpha)
                optimizer_centers.step()
                
                running_loss += loss.item()

            logger.info(f"  -> Epoch [{epoch+1}/{epochs}] - Avg Transfer Loss: {running_loss/len(train_loader):.4f}")

        # === 4. Mathematical Validation & Accumulation ===
        student_model.eval()
        with torch.no_grad():
            for videos, labels in test_loader:
                videos, labels = videos.to(device), labels.to(device)
                lengths = torch.full((videos.size(0),), videos.size(1), dtype=torch.long, device=device)
                
                out = student_model(videos, lengths=lengths)
                preds = torch.argmax(out["class_logits"], dim=1)
                
                metric_tracker.update(labels, preds)

        # Extrapolate internal structural variables yielding local evaluation
        fold_uf1, fold_uar = metric_tracker.compute_fold_metrics()
        logger.info(f">> Fold {fold_idx + 1}/{len(loso_validator.subjects)} [{test_subject}] completed. Localized UAR: {fold_uar:.4f} | Local UF1: {fold_uf1:.4f}")

        # === 5. Intelligent Checkpointing Verification ===
        if fold_uar > best_record_uar:
            best_record_uar = fold_uar
            save_path = f"{save_dir}/best_localized_mtmnet.pth"
            torch.save(student_model.state_dict(), save_path)
            logger.info(f">> Validation milestone reached! Memory saved mathematically securely to '{save_path}'")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # === 6. Terminal Final Logging Output Extraction ===
    global_metrics = metric_tracker.compute_global_metrics()
    logger.info("\n=======================================================")
    logger.info("       FINAL COMPOSITE DATABASE EVALUATION SUMMARY     ")
    logger.info("=======================================================")
    logger.info(f" Global Aggregate UF1       : {global_metrics['Global_UF1']:.4f}")
    logger.info(f" Global Aggregate UAR       : {global_metrics['Global_UAR']:.4f}")
    logger.info(f" Topological Variance (\u03c3)    : {global_metrics['UAR_std']:.4f} \u03c3 (std deviation from {len(loso_validator.subjects)} independent topologies)")
    logger.info("=======================================================\n")
    return global_metrics
