"""Ablation Study Master Controller.

Iteratively mutates the MTMNet structure, replacing explicit modules to
derive relative performance gaps.  Safely manages multi-epoch memory via
rigorous torch cache flushing and VRAM OOM hardening.

Ablation Variants
-----------------
1. **Baseline** — Full MTMNet
2. **-SimAM** — 3D Streams without SimAM attention
3. **-Transformer** — Pure CNN-LSTM (Global Average Pooling baseline)
4. **-JointLoss** — CE replaces Focal+Center; retains transfer guidance
5. **Vanilla-MTM** — Pure baseline with 0.0 transfer guidance weights

Scientific Parity
-----------------
- ``create_ablated_model()`` dynamically inherits ``embed_dim``,
  ``hidden_dim``, and ``patch_grid`` from the pristine student.
- ``AblatedSLSTTLSTM`` uses **Global Average Pooling (Mean)** to
  establish a fair, non-attentive baseline.
- ``reset_all_weights()`` respects Orthogonal Init for LSTMs.

Author  : Addhyan Pant
Project : Master's Thesis — Micro-Expression Recognition
"""

import csv
import datetime
import gc
import json
import logging
import os
import time
import copy
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm

from src.models.mtm_net import MTMNet
from src.evaluation.loso_handler import LOSOCrossValidator
from src.evaluation.metrics import MERMetricTracker
from src.losses.transfer_loss import MTMTransferLoss
from src.utils.checkpoint_manager import CheckpointManager

from run_loso_experiment import run_loso_experiment

LOG_FMT = "%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger("mer.ablation")


# ==========================================================================
# Reproducibility
# ==========================================================================

def seed_everything(seed: int = 42) -> None:
    """Variant-level reproducibility bounding deterministic execution."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Global seed locked to {seed}")


# ==========================================================================
# Ablated Modules
# ==========================================================================

class CrossEntropyLossWrapper(nn.Module):
    """Wraps standard CrossEntropyLoss to match the JointLoss API signature."""

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        # Dummy parameter to prevent ZeroDivision in SGD bindings
        self.dummy_center = nn.Parameter(torch.tensor([0.0]), requires_grad=True)

    def forward(
        self, logits: torch.Tensor, features: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        return self.ce(logits, labels)


class AblatedSpatialEncoder(nn.Module):
    """Pure Multi-stream 3D spatial encoder WITHOUT SimAM.

    Dynamically inherits ``embed_dim`` and ``patch_grid`` from the
    pristine encoder for scientific parity.
    """

    def __init__(
        self,
        in_channels: int = 3,
        stream_configs: tuple = ((3, (3, 3, 3)), (5, (3, 3, 3)), (8, (1, 3, 3))),
        patch_grid: tuple = (10, 10),
        embed_dim: int = 16,
    ) -> None:
        super().__init__()
        self.streams = nn.ModuleList()
        for out_c, kernel in stream_configs:
            pad_t = kernel[0] // 2
            self.streams.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_c, kernel_size=kernel, padding=(pad_t, 1, 1), bias=False),
                    nn.InstanceNorm3d(out_c, affine=True),
                    nn.ReLU(inplace=True),
                    # [SimAM Ablated — no attention module here]
                    nn.AdaptiveMaxPool3d((None, patch_grid[0], patch_grid[1])),
                )
            )
        self._out_channels = embed_dim
        sum_filters = sum(config[0] for config in stream_configs)
        self.feature_proj = nn.Conv3d(sum_filters, embed_dim, kernel_size=1, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm3d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @autocast(enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = x.float().permute(0, 2, 1, 3, 4).contiguous()
        out = torch.cat([stream(x) for stream in self.streams], dim=1)
        out = self.feature_proj(out).contiguous()
        b, c_out, t, h_out, w_out = out.shape
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        out = out.view(b, t, h_out * w_out, c_out)
        return out


class AblatedSLSTTLSTM(nn.Module):
    """Pure Temporal Modeler WITHOUT the Spatio-Temporal Transformer.

    Uses **Global Average Pooling (Mean)** across the spatial dimension
    to establish a fair, non-attentive baseline against the SLSTT block.

    Dynamically inherits ``embed_dim``, ``num_patches``, and
    ``lstm_hidden`` from the pristine aggregator.
    """

    def __init__(
        self,
        num_patches: int = 100,
        embed_dim: int = 16,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.lstm_hidden = lstm_hidden
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.0,
            bidirectional=True,
        )
        self.terminal_drop = nn.Dropout(p=dropout)
        self._init_specific_weights()

    def _init_specific_weights(self) -> None:
        """Apply orthogonal initialisation to recurrent weights (gate-wise)."""
        for name, param in self.lstm.named_parameters():
            if "weight_hh" in name:
                hidden_size = self.lstm_hidden
                for i in range(0, 4 * hidden_size, hidden_size):
                    nn.init.orthogonal_(param.data[i : i + hidden_size, :])
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                if "bias_ih" in name:
                    hidden_size = self.lstm_hidden
                    param.data[hidden_size : 2 * hidden_size].fill_(1.0)
        logger.info("Orthogonal init applied to AblatedSLSTTLSTM recurrent weights.")

    @property
    def out_features(self) -> int:
        return self.lstm_hidden * 2

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        # ── Global Average Pooling (Mean) across spatial patches ──
        # This is the scientifically fair non-attentive baseline:
        # mean over patches instead of Transformer CLS extraction.
        x_pool = x.mean(dim=2)  # (B, T, P, D) → (B, T, D)

        if lengths is not None:
            lengths_cpu = torch.as_tensor(lengths, dtype=torch.int64, device="cpu")
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                x_pool, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            _, (hn, _) = self.lstm(packed)
        else:
            _, (hn, _) = self.lstm(x_pool)

        forward_state = hn[-2, :, :]
        backward_state = hn[-1, :, :]
        out = torch.cat([forward_state, backward_state], dim=-1)
        return self.terminal_drop(out)


# ==========================================================================
# Delegation & Validation Helpers
# ==========================================================================

class CleanDelegationValidator:
    """Captures labels and subjects during the orchestrator's test pass."""

    def __init__(self, loader, labels_ref: list, subjects_ref: list, test_subject: str):
        self.loader = loader
        self.labels_ref = labels_ref
        self.subjects_ref = subjects_ref
        self.test_subject = test_subject

    def __iter__(self):
        self.labels_ref.clear()
        self.subjects_ref.clear()
        for batch in self.loader:
            labels = batch[1]
            self.labels_ref.extend(labels.cpu().tolist())
            if len(batch) >= 3:
                self.subjects_ref.extend(batch[2])
            else:
                self.subjects_ref.extend([self.test_subject] * labels.size(0))
            yield batch

    def __len__(self):
        return len(self.loader)


class SingleFoldOrchestrator:
    """Bounds the orchestrator to a specific LOSO fold."""

    def __init__(self, base_validator: LOSOCrossValidator, target_fold: int, labels_ref: list, subjects_ref: list):
        self.base_validator = base_validator
        self.target_fold = target_fold
        self.labels_ref = labels_ref
        self.subjects_ref = subjects_ref

    def get_folds(self, batch_size, num_workers):
        for f, s, tr, te in self.base_validator.get_folds(batch_size, num_workers):
            if f == self.target_fold:
                te_wrapped = CleanDelegationValidator(te, self.labels_ref, self.subjects_ref, s)
                yield f, s, tr, te_wrapped


# ==========================================================================
# Model Construction with Hyperparameter Inheritance
# ==========================================================================

def _extract_student_hyperparams(pristine_student: MTMNet) -> dict:
    """Dynamically extract architectural hyperparameters from the pristine student.

    Ensures scientific parity: ablated modules inherit the same
    ``embed_dim``, ``hidden_dim``, ``patch_grid``, etc.
    """
    encoder = pristine_student.spatial_encoder
    aggregator = pristine_student.temporal_aggregator

    # Extract embed_dim from encoder
    embed_dim = getattr(encoder, "_out_channels", 16)

    # Extract patch_grid from the AdaptivePool inside the first stream
    patch_grid = (10, 10)  # default
    if hasattr(encoder, "streams") and len(encoder.streams) > 0:
        for module in encoder.streams[0].modules():
            if isinstance(module, nn.AdaptiveMaxPool3d):
                output_size = module.output_size
                if output_size is not None and len(output_size) >= 3:
                    patch_grid = (output_size[1], output_size[2])
                break

    # Extract LSTM hidden from aggregator
    lstm_hidden = getattr(aggregator, "lstm_hidden", 64)

    # Extract num_patches
    num_patches = getattr(aggregator, "num_patches", patch_grid[0] * patch_grid[1])

    return {
        "embed_dim": embed_dim,
        "patch_grid": patch_grid,
        "lstm_hidden": lstm_hidden,
        "num_patches": num_patches,
    }


def create_ablated_model(pristine_student: MTMNet, variant: str, device: torch.device) -> MTMNet:
    """Instantiate an ablated model with dynamic hyperparameter inheritance."""
    student = copy.deepcopy(pristine_student)
    hp = _extract_student_hyperparams(pristine_student)

    if variant == "Baseline":
        return student.to(device)

    elif variant == "-SimAM":
        student.spatial_encoder = AblatedSpatialEncoder(
            in_channels=3,
            embed_dim=hp["embed_dim"],
            patch_grid=hp["patch_grid"],
        )
        logger.info(f"[-SimAM] Replaced SpatialEncoder (embed_dim={hp['embed_dim']}, patch_grid={hp['patch_grid']})")

    elif variant == "-Transformer":
        student.temporal_aggregator = AblatedSLSTTLSTM(
            num_patches=hp["num_patches"],
            embed_dim=hp["embed_dim"],
            lstm_hidden=hp["lstm_hidden"],
        )
        logger.info(
            f"[-Transformer] Replaced Aggregator (GAP baseline, "
            f"embed_dim={hp['embed_dim']}, lstm_hidden={hp['lstm_hidden']})"
        )

    elif variant in ["-JointLoss", "Vanilla-MTM"]:
        # Architecture remains pristine; loss ablation handles the objective change
        pass

    return student.to(device)


def create_ablated_teacher_weights(
    teacher_model_path: str, variant: str, save_dir: str, student_model: nn.Module
) -> str:
    """Shape-aware backbone-weight filtering for ablated architectures."""
    if variant in ["Baseline", "-JointLoss", "Vanilla-MTM"]:
        return teacher_model_path

    teacher_path = Path(teacher_model_path)
    if not teacher_path.exists():
        logger.warning(f"Teacher checkpoint not found: {teacher_path}. Initialising from scratch.")
        return teacher_model_path

    try:
        state_dict = torch.load(teacher_path, map_location="cpu")
    except Exception as e:
        logger.warning(f"Failed loading teacher state dict: {e}")
        return teacher_model_path

    student_state = student_model.state_dict()
    filtered_dict = {}
    deleted_keys = []

    for k, v in state_dict.items():
        if k in student_state and v.shape == student_state[k].shape:
            filtered_dict[k] = v
        else:
            deleted_keys.append(k)

    if deleted_keys:
        dropped = sorted([(k, state_dict[k].numel()) for k in deleted_keys], key=lambda x: x[1], reverse=True)
        top_3 = [f"{k} ({n:,})" for k, n in dropped[:3]]
        logger.info(f"[{variant}] Dropped keys (top 3): {', '.join(top_3)}")

    ablated_path = Path(save_dir) / f"ablated_teacher_{variant}.pth"
    torch.save(filtered_dict, ablated_path)
    return str(ablated_path)


# ==========================================================================
# Weight Reinitialisation
# ==========================================================================

def reset_all_weights(module: nn.Module) -> None:
    """Rigorous weight reinitialisation respecting module-specific inits."""
    for m in module.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
        elif hasattr(m, "_init_weights"):
            m._init_weights()

    # Second pass: restore specific inits (e.g., Orthogonal for LSTMs)
    for m in module.modules():
        if hasattr(m, "_init_specific_weights"):
            m._init_specific_weights()

    # Deep state leakage prevention for centroid parameters
    for name, param in module.named_parameters():
        if "center" in name.lower() and param.requires_grad:
            nn.init.normal_(param, std=0.01)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==========================================================================
# Fold Resumption
# ==========================================================================

def check_resume_fold(
    csv_path: str, variant: str, fold_idx: int, save_dir: str, current_config: dict = None
) -> bool:
    """Verify if [Variant, Fold] was previously evaluated with matching config."""
    if current_config is not None:
        config_path = Path(save_dir) / "study_config.json"
        if config_path.exists():
            try:
                saved_config = json.loads(config_path.read_text())
                mismatches = [
                    f"{k} (saved: {saved_config[k]}, current: {v})"
                    for k, v in current_config.items()
                    if k in saved_config and saved_config[k] != v
                ]
                if mismatches:
                    raise ValueError(
                        f"Resumption aborted: config mismatch would corrupt "
                        f"Confusion Matrix. Mismatches: {', '.join(mismatches)}"
                    )
            except json.JSONDecodeError:
                pass

    preds_path = Path(save_dir) / f"{variant}_fold_{fold_idx}_raw_preds.pth"
    csv_file = Path(csv_path)
    if not csv_file.exists() or not preds_path.exists():
        return False

    try:
        with open(csv_file, mode="r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Variant", "") == variant and row.get("Fold", "") == str(fold_idx):
                    return True
    except Exception:
        pass
    return False


# ==========================================================================
# LaTeX Export (Thesis-Grade)
# ==========================================================================

def _export_latex_table(
    delta_metrics: dict,
    param_counts: dict,
    baseline_param_count: int,
    ablation_variants: list,
    save_dir: str,
) -> None:
    """Export thesis-grade LaTeX table with booktabs and tabularx."""
    tex_path = Path(save_dir) / "ablation_results.tex"

    def escape_latex(val: str) -> str:
        return str(val).replace("_", "\\_").replace("%", "\\%")

    with open(tex_path, "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\resizebox{\\textwidth}{!}{\n")
        f.write("\\begin{tabularx}{\\textwidth}{X c c r r}\n")
        f.write("\\toprule\n")
        f.write(
            "\\textbf{Ablation Variant} & "
            "$\\mathbf{UF1}$ & "
            "$\\mathbf{UAR}$ & "
            "\\textbf{Parameters} & "
            "\\textbf{Saved Parameters} \\\\\n"
        )
        f.write("\\midrule\n")

        for variant in ablation_variants:
            if variant not in delta_metrics:
                continue
            metrics = delta_metrics.get(variant, {})
            uf1 = metrics.get("Global_UF1", 0.0)
            uar = metrics.get("Global_UAR", 0.0)
            p_cnt = param_counts.get(variant, 0)
            delta_p = baseline_param_count - p_cnt if baseline_param_count is not None else 0

            f.write(
                f"{escape_latex(variant)} & "
                f"{uf1:.4f} & "
                f"{uar:.4f} & "
                f"{p_cnt:,} & "
                f"{max(0, delta_p):,} \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabularx}\n")
        f.write("}\n")  # End resizebox
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(
            f"\\caption{{Ablation Study Results: Structural performance impact "
            f"on MTMNet. Generated: {timestamp}.}}\n"
        )
        f.write("\\label{tab:ablation_results}\n")
        f.write("\\end{table}\n")

    logger.info(f"LaTeX table exported to {tex_path}")


# ==========================================================================
# Main Ablation Suite
# ==========================================================================

def run_ablation_suite(
    pristine_student: MTMNet,
    teacher_model_path: str,
    base_transfer_loss_fn: MTMTransferLoss,
    loso_validator: LOSOCrossValidator,
    num_classes: int,
    epochs: int,
    device: torch.device,
    save_dir: str,
    alpha: float,
    gamma: float,
    lambda_c: float,
    lr: float = 1e-4,
    max_folds: int = None,
    num_workers: int = 0,
    use_amp: bool = True,
) -> None:
    """Execute Ablation Study with Total Orchestration Delegation.

    CRITICAL: For Docker containers with num_workers > 0, pass
    ``--shm-size=2g`` to prevent SIGBUS crashes during 5D tensor processing.
    """
    ablation_variants = ["Baseline", "-SimAM", "-Transformer", "-JointLoss", "Vanilla-MTM"]
    global_start_time = time.time()

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    csv_path = save_path / "ablation_results.csv"

    # Atomic checkpoint manager
    ckpt_mgr = CheckpointManager(save_path)

    # Save study config for resumption safety
    current_config = {"alpha": alpha, "gamma": gamma, "lambda_c": lambda_c, "epochs": epochs, "lr": lr}
    config_file = save_path / "study_config.json"
    if not config_file.exists():
        ckpt_mgr.save_json(current_config, "study_config.json")

    header = ["Timestamp", "Variant", "Fold", "Global_UF1", "Global_UAR", "Params", "Alpha", "Gamma", "Lambda_C"]
    if not csv_path.exists():
        with open(csv_path, mode="w", newline="") as f:
            csv.writer(f).writerow(header)

    delta_metrics = defaultdict(dict)
    param_counts = {}
    baseline_param_count = None

    var_pbar = tqdm(ablation_variants, desc="Ablation Variants", position=0)
    for variant in var_pbar:
        seed_everything(42)
        var_pbar.set_description(f"Variant: {variant}")

        student_variant = None
        loss_variant = None

        try:
            student_variant = create_ablated_model(pristine_student, variant, device)
            loss_variant = copy.deepcopy(base_transfer_loss_fn)
            reset_all_weights(student_variant)
            reset_all_weights(loss_variant)

            # Telemetry honesty logic
            current_alpha = alpha
            current_gamma = gamma
            current_lambda_c = lambda_c

            if variant == "-JointLoss":
                loss_variant.joint_loss = CrossEntropyLossWrapper()
                current_alpha, current_gamma, current_lambda_c = 0.0, 0.0, 0.0
            elif variant == "Vanilla-MTM":
                loss_variant.joint_loss = CrossEntropyLossWrapper()
                loss_variant.lir_alpha = 0.0
                loss_variant.triplet_beta = 0.0
                loss_variant.adv_gamma = 0.0
                current_alpha, current_gamma, current_lambda_c = 0.0, 0.0, 0.0

            safe_teacher_path = create_ablated_teacher_weights(
                teacher_model_path, variant, str(save_path), student_variant
            )
            param_count = count_parameters(student_variant)
            param_counts[variant] = param_count
            if variant == "Baseline":
                baseline_param_count = param_count

            metric_tracker = MERMetricTracker(num_classes=num_classes)
            subjects_list = loso_validator.subjects
            final_fold_idx = len(subjects_list) if max_folds is None else min(len(subjects_list), max_folds)

            fold_pbar = tqdm(range(final_fold_idx), desc="LOSO Folds", position=1, leave=False)
            for fold_idx in fold_pbar:
                fold_pbar.set_description(f"Fold {fold_idx}")

                # --- Hardened Resumption Logic ---
                if check_resume_fold(str(csv_path), variant, fold_idx, str(save_path), current_config):
                    saved = ckpt_mgr.load_predictions(variant, fold_idx, map_location=device)
                    if saved is not None:
                        logger.info(f"[{variant}] Fold {fold_idx} resumed from binary predictions.")
                        metric_tracker.update(
                            torch.tensor(saved["labels"], device=device),
                            torch.tensor(saved["preds"], device=device),
                            subjects=saved.get("subjects"),
                        )
                        continue

                # --- VRAM OOM Hardened Fold Execution ---
                try:
                    labels_ref, subjects_ref = [], []
                    orchestrator = SingleFoldOrchestrator(loso_validator, fold_idx, labels_ref, subjects_ref)

                    fold_results = run_loso_experiment(
                        student_model=student_variant,
                        teacher_model_path=safe_teacher_path,
                        transfer_loss_fn=loss_variant,
                        loso_validator=orchestrator,
                        num_classes=num_classes,
                        epochs=epochs,
                        device=device,
                        save_dir=str(save_path),
                        batch_size=16,
                        num_workers=num_workers,
                        strict_load=(variant in ["Baseline", "-JointLoss", "Vanilla-MTM"]),
                        use_amp=use_amp,
                    )

                    # Return-value synchronisation
                    all_preds = fold_results["fold_preds"]
                    all_labels = fold_results["fold_labels"]
                    subjects_final = fold_results.get("fold_subjects", subjects_ref)

                    metric_tracker.update(all_labels, all_preds, subjects=subjects_final)

                    # Atomic prediction save
                    ckpt_mgr.save_predictions(all_preds, all_labels, subjects_final, variant, fold_idx)

                    # CSV telemetry record
                    with open(csv_path, mode="a", newline="") as f:
                        csv.writer(f).writerow([
                            datetime.datetime.now().isoformat(),
                            variant,
                            fold_idx,
                            0.0,
                            0.0,
                            param_count,
                            current_alpha,
                            current_gamma,
                            current_lambda_c,
                        ])

                except RuntimeError as oom_err:
                    if "out of memory" in str(oom_err).lower():
                        logger.error(
                            f"CUDA OOM on [{variant}] Fold {fold_idx}. "
                            f"Reduce batch_size. Clearing VRAM, skipping fold."
                        )
                        if student_variant is not None:
                            student_variant.cpu()
                        if loss_variant is not None:
                            loss_variant.cpu()
                        torch.cuda.empty_cache()
                        gc.collect()
                        student_variant.to(device)
                        loss_variant.to(device)
                        continue
                    else:
                        raise

            global_metrics = metric_tracker.compute_global_metrics()
            delta_metrics[variant] = global_metrics
            _export_latex_table(delta_metrics, param_counts, baseline_param_count, ablation_variants, str(save_path))

        except Exception as e:
            logger.error(f"Failed variant [{variant}]: {e}")
            raise e
        finally:
            # VRAM Hardening: deterministic cleanup
            if student_variant is not None:
                student_variant.cpu()
                del student_variant
            if loss_variant is not None:
                loss_variant.cpu()
                del loss_variant
            torch.cuda.empty_cache()
            gc.collect()

    # Final summary
    print("\n" + "=" * 90)
    print(f"{'Ablation Variant':<20} | {'UF1':<8} | {'UAR':<8} | {'Params':<12} | {'Saved Params'}")
    print("-" * 90)
    for v in ablation_variants:
        m = delta_metrics.get(v, {})
        p = param_counts.get(v, 0)
        dw = (baseline_param_count - p) if baseline_param_count else 0
        print(f"{v:<20} | {m.get('Global_UF1', 0):.4f} | {m.get('Global_UAR', 0):.4f} | {p:<12,} | {max(0, dw):,}")
    print("=" * 90 + "\n")
    print(f"LaTeX exported to: {save_path / 'ablation_results.tex'}")

    total_time = time.time() - global_start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info(f"Total Execution Time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")


if __name__ == "__main__":
    logger.info("Ablation entrypoint detected. Import directly into orchestrators.")
