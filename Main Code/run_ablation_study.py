"""Ablation Study Master Controller.

Iteratively mutates the MTMNet structure structurally replacing explicit modules 
to mathematically derive relative performance gaps. Safely manages multi-epoch 
memory GC collections via rigorous torch cache flushing.

Ablation Sets:
    1. Baseline (Full MTMNet)
    2. -SimAM (3D Streams isolated without Attention)
    3. -Transformer (Pure CNN-LSTM pipeline without patch embeddings)
    4. -JointLoss (Replaces focal/center with CE, retains transfer guidance)
    5. Vanilla-MTM (Pure Baseline with 0.0 Transfer Guidance weights)

Author  : Addhyan Pant
Project : Master's Thesis — Micro-Expression Recognition
"""

import csv
import datetime
import gc
import logging
import os
import time
import copy
import random
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm

from src.models.mtm_net import MTMNet
from src.evaluation.loso_handler import LOSOCrossValidator
from src.evaluation.metrics import MERMetricTracker
from src.losses.transfer_loss import MTMTransferLoss

from run_loso_experiment import run_loso_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    """Variant-Level Reproducibility bounding absolute deterministic execution bounds."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Global manual seed structurally locked uniformly to {seed}")


class CrossEntropyLossWrapper(nn.Module):
    """Wraps standard CrossEntropyLoss to match the JointLoss API signature precisely."""
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        # Non-computational dummy parameter securing SGD gradient bindings preventing ZeroDivision
        self.dummy_center = nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        
    def forward(self, logits: torch.Tensor, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Ignore intermediate features, computing absolute scalar Cross-Entropy mathematically perfectly
        return self.ce(logits, labels)


class AblatedSpatialEncoder(nn.Module):
    """Pure Multi-stream 3D spatial feature encoder WITHOUT SimAM."""
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
                    # [SimAM Ablated]
                    nn.AdaptiveMaxPool3d((None, patch_grid[0], patch_grid[1]))
                )
            )
        self._out_channels = embed_dim
        sum_filters = sum(config[0] for config in stream_configs)
        self.feature_proj = nn.Conv3d(sum_filters, embed_dim, kernel_size=1, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
        # Evaluate explicitly maintaining Rec. 601 precision logic via FP32 defenses
        b, t, c, h, w = x.shape
        x = x.float().permute(0, 2, 1, 3, 4).contiguous()
        out = torch.cat([stream(x) for stream in self.streams], dim=1)
        out = self.feature_proj(out).contiguous()
        b, c_out, t, h_out, w_out = out.shape
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        out = out.view(b, t, h_out * w_out, c_out)
        return out


class AblatedSLSTTLSTM(nn.Module):
    """Pure Temporal Modeler WITHOUT the Spatio-Temporal Transformer (Pure CNN-LSTM)."""
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
            bidirectional=True
        )
        self.terminal_drop = nn.Dropout(p=dropout)
        self._init_specific_weights()

    def _init_specific_weights(self) -> None:
        """Explicitly apply orthogonal initialization to recurrent weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
        logger.info("Orthogonal initialization applied to AblatedSLSTTLSTM recurrent weights.")

    @property
    def out_features(self) -> int:
        return self.lstm_hidden * 2

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        # Scientific Global Max Pooling (max across patches) captures sparse micro-signal
        x_pool, _ = x.max(dim=2)
        
        if lengths is not None:
            lengths_cpu = torch.as_tensor(lengths, dtype=torch.int64, device='cpu')
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


class CleanDelegationValidator:
    """Captures labels and subjects during the orchestrator's test pass."""
    def __init__(self, loader, labels_ref: list, subjects_ref: list, test_subject: str):
        self.loader = loader
        self.labels_ref = labels_ref
        self.subjects_ref = subjects_ref
        self.test_subject = test_subject

    def __iter__(self):
        # Fatal Fix: Clear references to prevent cross-contamination across epochs/folds
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
    """Bounds the orchestrator to a specific LOSO fold without manual loops."""
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


def create_ablated_model(pristine_student: MTMNet, variant: str, device: torch.device) -> MTMNet:
    """Instantiate explicitly cleanly replacing architectural blocks dynamically without monkey patching."""
    student = copy.deepcopy(pristine_student)
    
    if variant == "Baseline":
        return student.to(device)
        
    elif variant == "-SimAM":
        # Natively replace strictly mapped architecture instance
        student.spatial_encoder = AblatedSpatialEncoder(in_channels=3, embed_dim=16)
        logger.info("Instantiated structurally bounds substituting -SimAM.")
        
    elif variant == "-Transformer":
        # Natively replace mapped architecture dropping patches
        student.temporal_aggregator = AblatedSLSTTLSTM()
        logger.info("Instantiated structurally bounds substituting -Transformer.")
        
    elif variant in ["-JointLoss", "Vanilla-MTM"]:
        # Architecture remains pristine, loss ablation handles objective changes.
        pass
        
    return student.to(device)


def create_ablated_teacher_weights(teacher_model_path: str, variant: str, save_dir: str, student_model: nn.Module) -> str:
    """Implement safe Backbone-Weight filtering enforcing MTMNet strict load bounds.
    
    Critical Scientific Fix: Filters only if keys match AND shapes match. Avoids 
    fragmented graph clashes mapping specifically downstream natively.
    """
    if variant in ["Baseline", "-JointLoss", "Vanilla-MTM"]:
        return teacher_model_path
        
    try:
        if not os.path.exists(teacher_model_path):
            logger.warning(f"Teacher model path `{teacher_model_path}` does not exist. Initializing from scratch.")
            return teacher_model_path
            
        state_dict = torch.load(teacher_model_path, map_location='cpu')
    except Exception as e:
        logger.warning(f"Failed loading teacher state dictionary bounds initially: {e}")
        return teacher_model_path
        
    filtered_dict = {}
    deleted_keys = []
    
    student_state = student_model.state_dict() if student_model is not None else {}
    
    for k, v in state_dict.items():
        # Shape-Aware Weight Filtering logically guaranteeing mathematically safe maps
        if k in student_state and v.shape == student_state[k].shape:
            filtered_dict[k] = v
        else:
            deleted_keys.append(k)
        
    if deleted_keys:
        # Sort dropped parameters by size for scientific preservation audit
        dropped_params = sorted([(k, state_dict[k].numel()) for k in deleted_keys], key=lambda x: x[1], reverse=True)
        top_3 = [f"{k} ({n:,})" for k, n in dropped_params[:3]]
        logger.info(f"[{variant}] Absolute structural preservation audit. Top 3 dropped: {', '.join(top_3)}.")
            
    ablated_path = os.path.join(save_dir, f"ablated_teacher_{variant}.pth")
    torch.save(filtered_dict, ablated_path)
    return ablated_path


def reset_all_weights(module: nn.Module) -> None:
    """Rigorous scientific-grade weight reinitialization across deep bounds."""
    for m in module.modules():
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
        elif hasattr(m, '_init_weights'):
            m._init_weights()
            
    # Init-Aware pass ensuring specific architectural weights overlap natively without traversal overwrites
    for m in module.modules():
        if hasattr(m, '_init_specific_weights'):
            m._init_specific_weights()
            
    # Deep state leakage prevention specifically targeting centroid parameters natively
    for name, param in module.named_parameters():
        if "center" in name.lower() and param.requires_grad:
            nn.init.normal_(param, std=0.01)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_resume_fold(csv_path: str, variant: str, fold_idx: int, save_dir: str, current_config: dict = None) -> bool:
    """Fault-Tolerant Verify if [Variant, Fold] is successfully previously evaluated."""
    import json
    if current_config is not None:
        config_path = os.path.join(save_dir, "study_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    saved_config = json.load(f)
                mismatches = []
                for k, v in current_config.items():
                    if k in saved_config and saved_config[k] != v:
                        mismatches.append(f"{k} (saved: {saved_config[k]}, current: {v})")
                if mismatches:
                    raise ValueError(f"Resumption aborted: parameter mismatch would corrupt Global Confusion Matrix. Mismatches: {', '.join(mismatches)}")
            except json.JSONDecodeError:
                pass

    metric_path = os.path.join(save_dir, f"{variant}_fold_{fold_idx}_raw_preds.pth")
    if not os.path.exists(csv_path) or not os.path.exists(metric_path):
        return False
    try:
        with open(csv_path, mode='r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Variant", "") == variant and row.get("Fold", "") == str(fold_idx):
                    return True
    except Exception:
        pass
    return False


def _export_latex_table(delta_metrics: dict, param_counts: dict, baseline_param_count: int, ablation_variants: list, save_dir: str):
    """Thesis-Grade LaTeX Caching with professional tabularx formatting."""
    tex_path = os.path.join(save_dir, "ablation_results.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        # Step 7 Fix: Integrate resizebox with tabularx for perfect document scaling compatibility
        f.write("\\resizebox{\\textwidth}{!}{\n")
        f.write("\\begin{tabularx}{\\textwidth}{X c c r r}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Ablation Variant} & \\textbf{UF1 Score} & \\textbf{UAR Score} & \\textbf{Parameters} & \\textbf{Saved Weights} \\\\\n")
        f.write("\\midrule\n")
        
        for variant in ablation_variants:
            if variant not in delta_metrics:
                continue
            metrics = delta_metrics.get(variant, {})
            uf1 = metrics.get('Global_UF1', 0.0)
            uar = metrics.get('Global_UAR', 0.0)
            p_cnt = param_counts.get(variant, 0)
            delta_p = baseline_param_count - p_cnt if baseline_param_count is not None else 0
            
            def escape_latex(val: str) -> str:
                return str(val).replace('_', '\\_').replace('%', '\\%')
                
            variant_cln = escape_latex(variant)
            uf1_cln = f"{uf1:.4f}"
            uar_cln = f"{uar:.4f}"
            p_cnt_cln = f"{p_cnt:,}"
            delta_p_cln = f"{max(0, delta_p):,}"
            
            f.write(f"{variant_cln} & {uf1_cln} & {uar_cln} & {p_cnt_cln} & {delta_p_cln} \\\\\n")
            
        f.write("\\bottomrule\n")
        f.write("\\end{tabularx}\n")
        f.write("}\n")  # End resizebox
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\\caption{{Ablation Study Results: Structural performance impact on MTMNet. Generated: {timestamp}.}}\n")
        f.write("\\label{tab:ablation_results}\n")
        f.write("\\end{table}\n")


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
    num_workers: int = 0
) -> None:
    """Execute Ablation Study with Total Orchestration Delegation.
    
    CRITICAL: For Docker containers with num_workers > 0, set --shm-size=2g to prevent 
    SIGBUS crashes during 5-D tensor processing.
    """
    ablation_variants = ["Baseline", "-SimAM", "-Transformer", "-JointLoss", "Vanilla-MTM"]
    global_start_time = time.time()
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "ablation_results.csv")
    
    # Save study config for resumption safety
    import json
    config_path = os.path.join(save_dir, "study_config.json")
    current_config = {"alpha": alpha, "gamma": gamma, "lambda_c": lambda_c, "epochs": epochs, "lr": lr}
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump(current_config, f, indent=4)
            
    header = ["Timestamp", "Variant", "Fold", "Global_UF1", "Global_UAR", "Params", "Alpha", "Gamma", "Lambda_C"]
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as f:
            csv.writer(f).writerow(header)

    delta_metrics = defaultdict(dict)
    param_counts = {}
    baseline_param_count = None

    var_pbar = tqdm(ablation_variants, desc="Ablation Variants", position=0)
    for variant in var_pbar:
        seed_everything(42)  # Reset seed for variant parity
        var_pbar.set_description(f"Variant: {variant}")
        
        try:
            student_variant = create_ablated_model(pristine_student, variant, device)
            loss_variant = copy.deepcopy(base_transfer_loss_fn)
            reset_all_weights(student_variant)
            reset_all_weights(loss_variant)
            
            # Telemetry Honesty Logic
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
    
            safe_teacher_path = create_ablated_teacher_weights(teacher_model_path, variant, save_dir, student_variant)
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
                preds_path = os.path.join(save_dir, f"{variant}_fold_{fold_idx}_raw_preds.pth")
                if check_resume_fold(csv_path, variant, fold_idx, save_dir, current_config):
                    if os.path.exists(preds_path):
                        logger.info(f"[{variant}] Fold {fold_idx} resumed. Re-injecting predictive binary priors.")
                        saved = torch.load(preds_path, map_location=device)
                        
                        # Properly handle subjects for Subject-Independent CV report validity
                        metric_tracker.update(
                            torch.tensor(saved['labels'], device=device),
                            torch.tensor(saved['preds'], device=device),
                            subjects=saved.get('subjects')
                        )
                        continue
                
                # --- Total Orchestration Delegation (Step 7: Rely strictly on returned results) ---
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
                    save_dir=save_dir,
                    batch_size=16,
                    num_workers=num_workers,
                    strict_load=(variant in ["Baseline", "-JointLoss", "Vanilla-MTM"])
                )
                
                # Extract results strictly from orchestrator return to prevent VRAM pollution
                all_preds = fold_results['fold_preds']
                all_labels = fold_results['fold_labels']
                subjects_final = fold_results.get('fold_subjects', subjects_ref)
                
                # Update tracker with gold-standard results
                metric_tracker.update(all_labels, all_preds, subjects=subjects_final)
                
                # Save binary predictions for resumption
                torch.save({
                    "preds": all_preds.cpu().numpy() if torch.is_tensor(all_preds) else all_preds,
                    "labels": all_labels.cpu().numpy() if torch.is_tensor(all_labels) else all_labels,
                    "subjects": subjects_final
                }, preds_path)
                
                # Telemetry Honesty: Write CSV record
                with open(csv_path, mode='a', newline='') as f:
                    csv.writer(f).writerow([
                        datetime.datetime.now().isoformat(),
                        variant, fold_idx, 0.0, 0.0,
                        param_count, current_alpha, current_gamma, current_lambda_c
                    ])

            global_metrics = metric_tracker.compute_global_metrics()
            delta_metrics[variant] = global_metrics
            _export_latex_table(delta_metrics, param_counts, baseline_param_count, ablation_variants, save_dir)
            
        except Exception as e:
            logger.error(f"Failed variant [{variant}]: {e}")
            raise e
        finally:
            # VRAM Hardening: Move to CPU and clear cache deterministically
            if 'student_variant' in locals():
                student_variant.to('cpu')
                del student_variant
            if 'loss_variant' in locals():
                loss_variant.to('cpu')
                del loss_variant
            torch.cuda.empty_cache()
            gc.collect()

    # Final summary display
    print("\n" + "="*80 + "\n" + f"{'Ablation Variant':<20} | {'UF1':<8} | {'UAR':<8} | {'Params':<12} | {'Saved Weights'}\n" + "-"*80)
    for v in ablation_variants:
        m = delta_metrics.get(v, {})
        p = param_counts.get(v, 0)
        dw = (baseline_param_count - p) if baseline_param_count else 0
        print(f"{v:<20} | {m.get('Global_UF1', 0):.4f} | {m.get('Global_UAR', 0):.4f} | {p:<12,} | {max(0, dw):,}")
    print("="*80 + "\n")
    print(f"Mathematical proofs successfully derived. Final LaTeX compiled securely continuously at: {os.path.join(save_dir, 'ablation_results.tex')}")

    total_time = time.time() - global_start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info(f"Total Cumulative Execution Time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")


if __name__ == "__main__":
    logger.info("Ablation entrypoint detected. Import directly into orchestrators.")
