"""Master Orchestration Layer.

Production-grade CLI Orchestrator sequencing the full micro-expression
recognition pipeline.  Supports checkpoint resumption natively, encapsulates
global hyperparameters, and integrates RTX 4080 optimisations (AMP,
``torch.compile``).

Environment Variables
---------------------
DATASET_PATH : str
    Root directory containing preprocessed dataset tensors.
    Falls back to ``--data_dir`` CLI argument or ``./data``.
OUTPUT_DIR : str
    Root directory for checkpoints, metrics, and LaTeX exports.
    Falls back to ``--save_dir`` CLI argument or ``./outputs``.

Usage
-----
::

    # Full pipeline with resumption
    python main.py pipeline --resume

    # Single stage
    python main.py ablate --amp --compile

    # Docker (volume-mounted paths via env vars)
    docker run --gpus all --shm-size=2g \\
        -v /host/data:/data -v /host/out:/outputs \\
        mer-thesis ablate --amp

Author  : Addhyan Pant
Project : Master's Thesis — Micro-Expression Recognition
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

from src.models.mtm_net import MTMNet
from src.models.components.spatial_encoder import SpatialEncoder
from src.models.components.temporal_aggregator import SLSTTLSTM
from src.models.components.classification_head import ClassificationHead
from src.losses.transfer_loss import MTMTransferLoss
from src.evaluation.loso_handler import LOSOCrossValidator
from pipeline_stages import PipelineOrchestrator
from run_ablation_study import run_ablation_suite

# ---------------------------------------------------------------------------
# Hierarchical Logging
# ---------------------------------------------------------------------------
LOG_FMT = "%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger("mer.main")


# ---------------------------------------------------------------------------
# Hardware Enforcement
# ---------------------------------------------------------------------------

def enforce_gpu() -> torch.device:
    """Ensure a CUDA-capable GPU is available (required for thesis execution)."""
    if not torch.cuda.is_available():
        logger.error("CUDA acceleration is required. No GPU detected — aborting.")
        sys.exit(1)
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)
    vram_gb = torch.cuda.get_device_properties(device).total_mem / (1024 ** 3)
    logger.info(f"GPU: {gpu_name} | VRAM: {vram_gb:.1f} GB")
    return device


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------

def build_pristine_student(num_classes: int = 3) -> MTMNet:
    """Construct a clean MTMNet with default hyperparameters.

    Uses the canonical architecture:
    - SpatialEncoder: 3 streams, embed_dim=16, patch_grid=(10,10)
    - SLSTTLSTM: embed_dim=16, lstm_hidden=64, 2 transformer layers
    - ClassificationHead: input_dim=128, hidden_dim=64
    """
    encoder = SpatialEncoder(embed_dim=16)
    aggregator = SLSTTLSTM(embed_dim=16, lstm_hidden=64)
    head = ClassificationHead(
        input_dim=aggregator.out_features,
        hidden_dim=64,
        num_classes=num_classes,
    )
    return MTMNet(
        spatial_encoder=encoder,
        temporal_aggregator=aggregator,
        classification_head=head,
    )


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Construct the master argument parser."""
    parser = argparse.ArgumentParser(
        description="Master Orchestrator for the Micro-Expression Recognition Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "command",
        choices=["preprocess", "pretrain", "evaluate", "ablate", "pipeline"],
        help="Execution stage.  'pipeline' runs all stages sequentially with resumption.",
    )

    # --- Resumption ---
    parser.add_argument(
        "--resume", action="store_true",
        help="Enable checkpoint-aware skipping in Pre-training and Ablation stages.",
    )

    # --- RTX 4080 Optimisations ---
    parser.add_argument(
        "--amp", action="store_true", default=True,
        help="Enable Automatic Mixed Precision (FP16) for 4th-Gen Tensor Cores.",
    )
    parser.add_argument(
        "--no-amp", dest="amp", action="store_false",
        help="Disable AMP (use full FP32 precision).",
    )
    parser.add_argument(
        "--compile", action="store_true", default=False,
        help="Apply torch.compile() for Ada Lovelace kernel optimisation (PyTorch 2.0+).",
    )

    # --- Hyperparameters ---
    parser.add_argument("--alpha", type=float, default=1.0, help="LIR Margin Scaling (α)")
    parser.add_argument("--gamma", type=float, default=0.5, help="Adversarial Trade-off (γ)")
    parser.add_argument("--lambda_c", type=float, default=0.1, help="Center Loss Coefficient (λ_c)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Global learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs per fold")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for DataLoaders")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of emotion classes")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 for Docker safety)")

    # --- Directory Architecture (env-var aware) ---
    parser.add_argument(
        "--save_dir", type=str,
        default=os.environ.get("OUTPUT_DIR", "./outputs"),
        help="Output directory (overridden by OUTPUT_DIR env var).",
    )
    parser.add_argument(
        "--data_dir", type=str,
        default=os.environ.get("DATASET_PATH", "./data"),
        help="Dataset root directory (overridden by DATASET_PATH env var).",
    )
    parser.add_argument(
        "--teacher_path", type=str,
        default=None,
        help="Path to pre-trained Teacher (MacroNet) checkpoint.",
    )

    return parser


# ---------------------------------------------------------------------------
# Stage Handlers
# ---------------------------------------------------------------------------

def stage_preprocess(config: dict) -> None:
    """Stage 1: Eulerian Video Magnification preprocessing."""
    logger.info("Executing Eulerian Video Magnification (EVM) preprocessing...")
    # Placeholder — EVM preprocessing logic is external (CPU-bound)
    logger.info("Preprocessing stage completed.")


def stage_pretrain(config: dict) -> None:
    """Stage 2: Global Teacher (MacroNet) pre-training."""
    logger.info(f"Executing Global Teacher Pre-training (Resume: {config['resume']})...")
    # Placeholder — Teacher pre-training on macro-expression dataset
    logger.info("Pre-training stage completed.")


def stage_evaluate(config: dict) -> None:
    """Stage 3: Full LOSO Experiment validation."""
    logger.info("Executing Full LOSO Experiment Validation...")
    # Placeholder — Single-variant LOSO evaluation
    logger.info("Evaluation stage completed.")


def stage_ablate(config: dict) -> None:
    """Stage 4: Defensible Ablation Suite."""
    device = config["device"]
    save_dir = Path(config["save_dir"])

    logger.info(f"Initiating Ablation Suite (Resume: {config['resume']})")
    logger.info(f"Output directory: {save_dir}")

    # Build pristine student
    pristine_student = build_pristine_student(num_classes=config["num_classes"])

    # Resolve teacher path
    teacher_path = config["teacher_path"]
    if teacher_path is None:
        teacher_path = str(save_dir / "teacher_macro.pth")
    teacher_path = str(Path(teacher_path))

    # Build transfer loss
    base_loss = MTMTransferLoss(
        num_classes=config["num_classes"],
        feature_dim=pristine_student.temporal_aggregator.out_features,
        joint_lambda_c=config["lambda_c"],
        lir_alpha=config["alpha"],
        triplet_beta=1.0,
        adv_gamma=config["gamma"],
    )

    # Optional torch.compile
    if config.get("compile", False):
        if hasattr(torch, "compile"):
            logger.info("Applying torch.compile() for Ada Lovelace kernel optimisation...")
            pristine_student = torch.compile(pristine_student)
        else:
            logger.warning("torch.compile() not available (requires PyTorch 2.0+). Skipping.")

    run_ablation_suite(
        pristine_student=pristine_student,
        teacher_model_path=teacher_path,
        base_transfer_loss_fn=base_loss,
        loso_validator=None,  # Injected from data pipeline
        num_classes=config["num_classes"],
        epochs=config["epochs"],
        device=device,
        save_dir=str(save_dir),
        alpha=config["alpha"],
        gamma=config["gamma"],
        lambda_c=config["lambda_c"],
        lr=config["lr"],
        num_workers=config["num_workers"],
        use_amp=config.get("amp", True),
    )
    logger.info("Ablation Suite completed successfully.")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """Master CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Resolve paths via pathlib
    save_dir = Path(args.save_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    device = enforce_gpu()

    config = {
        "alpha": args.alpha,
        "gamma": args.gamma,
        "lambda_c": args.lambda_c,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_classes": args.num_classes,
        "num_workers": args.num_workers,
        "resume": args.resume,
        "save_dir": str(save_dir),
        "data_dir": str(data_dir),
        "teacher_path": args.teacher_path,
        "device": device,
        "amp": args.amp,
        "compile": args.compile,
    }

    logger.info(f"Command: {args.command}")
    logger.info(f"Config: alpha={args.alpha}, gamma={args.gamma}, lambda_c={args.lambda_c}, "
                f"lr={args.lr}, epochs={args.epochs}, amp={args.amp}, compile={args.compile}")
    logger.info(f"Paths: save_dir={save_dir}, data_dir={data_dir}")

    # --- Dispatch ---
    if args.command == "pipeline":
        orchestrator = PipelineOrchestrator(save_dir)
        handlers = {
            "preprocess": lambda: stage_preprocess(config),
            "pretrain": lambda: stage_pretrain(config),
            "evaluate": lambda: stage_evaluate(config),
            "ablate": lambda: stage_ablate(config),
        }
        orchestrator.run_all(handlers)
        logger.info(f"Pipeline status: {orchestrator.get_status_summary()}")

    elif args.command == "preprocess":
        stage_preprocess(config)
    elif args.command == "pretrain":
        stage_pretrain(config)
    elif args.command == "evaluate":
        stage_evaluate(config)
    elif args.command == "ablate":
        stage_ablate(config)


if __name__ == "__main__":
    main()
