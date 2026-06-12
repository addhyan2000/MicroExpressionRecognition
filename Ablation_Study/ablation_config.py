"""
ablation_config.py — Experiment Configuration for the Stage 1 + Stage 2 Ablation
=================================================================================

This module centralises *all* tunable knobs for the ablation study so that the
model code, the dataset code, and the training loop never hard-code anything.

It defines two dataclasses:

    1. ``AblationConfig``
       The four scientific toggles that define a single experiment cell in the
       ablation matrix:

           A. use_evm          → Stage 1 preprocessing (motion magnification)
           B. use_simam        → Stage 2 parameter-free spatial attention
           C. use_cnn          → Stage 2 3D-CNN spatial feature extractor
           D. use_transformer  → Stage 2 SLSTT temporal sequencer

       NOTE on the scope of each toggle:
           • ``use_evm`` is a *data-level* switch. EVM is applied offline in
             Stage 1, producing a different set of ``.npy`` motion tensors.
             Toggling it therefore selects between two tensor directories
             (magnified vs. raw); it does NOT change the network graph.
           • ``use_simam``, ``use_cnn``, ``use_transformer`` are *model-level*
             switches that conditionally add/remove modules from the forward
             pass (see ``models.py``).

    2. ``ExperimentConfig``
       Dataset paths, the two tensor directories (EVM vs. raw), the emotion
       class map, optimisation hyper-parameters, and output locations. Every
       value here is explicit and overridable — there are NO hidden multi-
       dataset assumptions, subject-range constants, or class-map overrides.

The canonical 8-cell ``ABLATION_MATRIX`` (Phases I–IV from the thesis brief)
is constructed at the bottom of this file.

Author : Addhyan
Stage  : Ablation Study (Stage 1 + Stage 2 isolation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PROJECT ROOT — everything is derived from this single anchor.
# ─────────────────────────────────────────────────────────────────────────────
# Ablation_Study/ lives directly under the repository root, so the parent of
# this file's directory is the project root (same convention as Stage 1/3).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# 2.  THE FOUR SCIENTIFIC TOGGLES — one ablation cell.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AblationConfig:
    """
    A single experiment cell in the 2x2x2x2 ablation matrix.

    Each instance fully describes which components are active for one training
    run. The ``name`` and ``phase`` fields are purely for bookkeeping so that
    results are written to clearly labelled, unique files.

    Parameters
    ----------
    name : str
        Human-readable unique identifier, e.g. ``"config_8_proposed_unified"``.
        Used to name log files, checkpoints, and metric dumps.
    phase : str
        Thesis phase grouping (``"I"``, ``"II"``, ``"III"``, ``"IV"``).
    use_evm : bool
        Variable A — Stage 1 Eulerian Video Magnification.
        Selects the EVM-magnified tensor directory when True, raw otherwise.
    use_simam : bool
        Variable B — Stage 2 SimAM spatial attention (only meaningful when
        ``use_cnn`` is True, since SimAM rescales CNN feature maps).
    use_cnn : bool
        Variable C — Stage 2 three-stream 3D-CNN spatial feature extractor.
        When False, the model falls back to raw flattened spatial patches.
    use_transformer : bool
        Variable D — Stage 2 SLSTT Transformer temporal encoder.
        When False, the model falls back to simple mean/max temporal pooling.
    """

    name: str
    phase: str
    use_evm: bool
    use_simam: bool
    use_cnn: bool
    use_transformer: bool

    # ── Pretty, log-friendly description of the active components ──
    def describe(self) -> str:
        """Return a compact ``[A|B|C|D]`` style component summary string."""

        def tag(flag: bool, label: str) -> str:
            return f"+{label}" if flag else f"-{label}"

        return " ".join([
            tag(self.use_evm, "EVM"),
            tag(self.use_simam, "SimAM"),
            tag(self.use_cnn, "CNN3D"),
            tag(self.use_transformer, "SLSTT"),
        ])

    @property
    def folder_name(self) -> str:
        def tag(flag: bool, label: str) -> str:
            return f"WITH_{label}" if flag else f"no_{label}"
        return f"{self.name}__{tag(self.use_evm, 'evm')}__{tag(self.use_simam, 'simam')}__{tag(self.use_cnn, '3dcnn')}__{tag(self.use_transformer, 'transformer')}"

    # ── Validity guard: SimAM has no meaning without a CNN feature map ──
    def is_valid(self) -> bool:
        """
        Return True if the toggle combination is architecturally meaningful.

        SimAM rescales 3D-CNN feature maps; with the CNN disabled there is no
        feature map to attend over, so ``use_simam=True, use_cnn=False`` is
        considered degenerate. The 8 thesis configs never hit this case, but
        the orchestrator checks it defensively.
        """
        if self.use_simam and not self.use_cnn:
            return False
        return True


# ─────────────────────────────────────────────────────────────────────────────
# 3.  GLOBAL EXPERIMENT CONFIGURATION — paths, classes, hyper-parameters.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ExperimentConfig:
    """
    All non-toggle settings shared across every ablation cell.

    This is intentionally a *mutable* dataclass (not frozen) so a thin CLI in
    ``run_ablation_experiments.py`` can override any field at launch time
    without editing source. Nothing here is dataset-specific beyond the values
    you explicitly pass — there are no hidden subject ranges or class overrides.

    Dataset generalisation
    -----------------------
    ``dataset_filter`` restricts the master CSV to a single dataset tag (e.g.
    ``"CASME_II"``). Set it to ``None`` to use every row in the CSV. This is the
    one knob that abstracts away multi-dataset unification: the rest of the code
    never assumes more than one dataset.
    """

    # ── Data sources ─────────────────────────────────────────────────────────
    csv_path: Path = PROJECT_ROOT / "Processed_Data" / "master_thesis_labels.csv"

    # Variable A is data-level: two precomputed tensor sets must exist on disk.
    #   • tensor_dir_evm : optical flow/strain computed on EVM-MAGNIFIED frames
    #   • tensor_dir_raw : optical flow/strain computed on RAW (non-magnified) frames
    # Generate each by running the Stage 1 Step-2 pipeline once with EVM on and
    # once with EVM off, writing to these two directories respectively.
    tensor_dir_evm: Path = PROJECT_ROOT / "Processed_Data" / "tensors"
    tensor_dir_raw: Path = PROJECT_ROOT / "Processed_Data" / "tensors_raw"

    # ── Dataset generalisation knobs ──────────────────────────────────────────
    dataset_filter: str | None = "CASME_II"        # single-dataset focus
    expression_filter: str | None = "micro-expression"

    # ── Class definition (config-driven; NO hardcoded class overrides) ────────
    # Default: the 3 thesis classes. "Others" is intentionally excluded by not
    # listing it here, so num_classes is derived from this map's size.
    emotion_map: Dict[str, int] = field(default_factory=lambda: {
        "Negative": 0,
        "Positive": 1,
        "Surprise": 2,
    })

    # ── Spatiotemporal tensor geometry (kept fixed per the thesis brief) ──────
    in_channels: int = 3            # u-flow, v-flow, optical strain
    sequence_length: int = 32       # interpolated temporal frames (T)
    spatial_size: int = 224         # H == W

    # ── Backbone / transformer dimensions (kept intact) ───────────────────────
    cnn_mid_channels: int = 16
    cnn_out_channels: int = 32      # ×3 streams = 96 = d_model
    cnn_dropout: float = 0.3
    simam_lambda: float = 1e-4
    d_model: int = 96               # transformer width; also raw-patch proj dim
    transformer_nhead: int = 8
    transformer_num_layers: int = 4
    transformer_dim_ff: int = 256
    transformer_dropout: float = 0.1
    pool_strategy: Literal["mean", "cls"] = "mean"

    # When the 3D-CNN is OFF, raw frames are average-pooled to this small
    # spatial grid then flattened into per-frame "patch" vectors.
    raw_patch_grid: int = 4         # → 3 * 4 * 4 = 48-dim, projected to d_model
    # When the Transformer is OFF, this pooling collapses the time axis.
    temporal_pool: Literal["mean", "max"] = "mean"
    classifier_dropout: float = 0.3

    # ── Loss (decoupled; no SupCon, no XBM, no identity/GRL) ──────────────────
    loss_type: Literal["focal", "cross_entropy"] = "focal"
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05
    use_class_weights: bool = True

    # ── Optimisation ──────────────────────────────────────────────────────────
    epochs: int = 60
    batch_size: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float | None = 1.0
    use_amp: bool = True

    # ── Validation protocol (strict subject-disjoint) ─────────────────────────
    # "holdout"  → single subject-disjoint train/val split (fast; default)
    # "loso"     → full leave-one-subject-out cross-validation (slow; thorough)
    validation_protocol: Literal["holdout", "loso"] = "holdout"
    val_fraction: float = 0.2
    seed: int = 42

    # ── Output locations ──────────────────────────────────────────────────────
    output_root: Path = PROJECT_ROOT / "Ablation_Study" / "results"
    log_dir: Path = PROJECT_ROOT / "Ablation_Study" / "logs"

    # ── Derived helpers ────────────────────────────────────────────────────────
    @property
    def num_classes(self) -> int:
        """Number of emotion classes — derived from the emotion map size."""
        return len(self.emotion_map)

    @property
    def class_names(self) -> List[str]:
        """Class names ordered by their integer label (for confusion matrices)."""
        return [name for name, _ in sorted(self.emotion_map.items(), key=lambda kv: kv[1])]

    def tensor_dir_for(self, use_evm: bool) -> Path:
        """Return the tensor directory matching the EVM toggle (Variable A)."""
        return self.tensor_dir_evm if use_evm else self.tensor_dir_raw


# ─────────────────────────────────────────────────────────────────────────────
# 4.  THE 8-CELL ABLATION MATRIX (Phases I–IV from the thesis brief).
# ─────────────────────────────────────────────────────────────────────────────
# Each row is (name, phase, EVM, SimAM, CNN, Transformer).
_ORIGINAL_MATRIX = [
    AblationConfig("config_1_pure_base",        "I",   False, False, False, False),
    AblationConfig("config_2_temporal_only",    "I",   False, False, False, True),
    AblationConfig("config_3_spatial_only",     "I",   False, False, True,  False),
    AblationConfig("config_4_motion_amp_base",  "II",  True,  False, False, False),
    AblationConfig("config_5_attention_base",   "II",  False, True,  True,  False),
    AblationConfig("config_6_full_stage2_noevm","III", False, True,  True,  True),
    AblationConfig("config_7_full_no_attention","III", True,  False, True,  True),
    AblationConfig("config_8_proposed_unified", "IV",  True,  True,  True,  True),
]

_known_configs = {(c.use_evm, c.use_simam, c.use_cnn, c.use_transformer): c for c in _ORIGINAL_MATRIX}

import itertools

ABLATION_MATRIX: List[AblationConfig] = []
_config_idx = 9
for evm, simam, cnn, trans in itertools.product([False, True], repeat=4):
    key = (evm, simam, cnn, trans)
    if key in _known_configs:
        ABLATION_MATRIX.append(_known_configs[key])
    else:
        name = f"config_{_config_idx}_permutation"
        _config_idx += 1
        ABLATION_MATRIX.append(AblationConfig(name, "Other", evm, simam, cnn, trans))


def get_ablation_matrix() -> List[AblationConfig]:
    """Return a fresh copy of the canonical 8-cell ablation matrix."""
    return list(ABLATION_MATRIX)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MPI FACIAL EXPRESSION DATABASE — LABEL MAPPINGS
# ─────────────────────────────────────────────────────────────────────────────
#
# The MPI database has 51 unique expression classes (13 subsets × ~4 each).
# Two mappings are provided:
#
#   • MPI_EMOTION_MAP_3CLASS  — collapses MPI labels into the same 3 thesis
#     classes (Positive / Negative / Surprise) for direct comparison with
#     CASME-II results. Ambiguous / conversational expressions are mapped to
#     "Others" and excluded by the emotion_map in ExperimentConfig.
#
#   • MPI_EMOTION_MAP_FULL   — assigns a unique integer to each of the 51
#     MPI expressions. Use this when retraining the classifier for the full
#     MPI label space (num_classes == 51).
#
# Academic justification for the 3-class mapping:
#   • Positive: expressions with clear positive valence / enjoyment
#   • Negative: expressions with aversive valence (disgust, fear, pain, etc.)
#   • Surprise: expressions involving sudden realisation or confusion
#   • Others: conversational / socially functional expressions that do not
#     map cleanly to a single emotional valence
# ─────────────────────────────────────────────────────────────────────────────

MPI_EMOTION_MAP_3CLASS: Dict[str, str] = {
    # ── Positive (enjoyment, amusement, satisfaction) ──
    "happy_achievement":      "Positive",
    "happy_laughing":         "Positive",
    "happy_satiated":         "Positive",
    "happy_schadenfreude":    "Positive",
    "smiling_encouraging":    "Positive",
    "smiling_endearment":     "Positive",
    "smiling_flirting":       "Positive",
    "smiling_triumphant":     "Positive",
    "smiling_winning":        "Positive",
    "imagine_positive":       "Positive",
    "remember_positive":      "Positive",
    "impressed":              "Positive",
    "aha-light_bulb_moment":  "Positive",
    "compassion":             "Positive",

    # ── Negative (aversive, distress, aversion) ──
    "annoyed_bothered":       "Negative",
    "annoyed_rolling-eyes":   "Negative",
    "contempt":               "Negative",
    "disgust":                "Negative",
    "embarrassment":          "Negative",
    "fear_oops":              "Negative",
    "fear_terror":            "Negative",
    "sad":                    "Negative",
    "pain_felt":              "Negative",
    "pain_seen":              "Negative",
    "insecurity":             "Negative",
    "imagine_negative":       "Negative",
    "remember_negative":      "Negative",
    "tired":                  "Negative",
    "arrogant":               "Negative",
    "smiling_sardonic":       "Negative",
    "smiling_yeah-right":     "Negative",

    # ── Surprise (sudden realisation, confusion, disbelief) ──
    "confused":               "Surprise",
    "disbelief":              "Surprise",
    "I_did_not_hear":         "Surprise",
    "I_dont_understand":      "Surprise",

    # ── Others (conversational / ambiguous — excluded from 3-class training) ──
    "agree_considered":       "Others",
    "agree_continue":         "Others",
    "agree_pure":             "Others",
    "agree_reluctant":        "Others",
    "disagree_considered":    "Others",
    "disagree_pure":          "Others",
    "disagree_reluctant":     "Others",
    "I_dont_care":            "Others",
    "I_dont_know":            "Others",
    "bored":                  "Others",
    "not_convinced":          "Others",
    "smiling_sad-nostalgia":  "Others",
    "smiling_uncertain":      "Others",
    "thinking_considering":   "Others",
    "thinking_problem-solving": "Others",
    "treudoof_bambi-eyes":    "Others",
}

# All 51 expressions sorted alphabetically → integer label 0–50.
_MPI_ALL_EXPRESSIONS = sorted(MPI_EMOTION_MAP_3CLASS.keys())

MPI_EMOTION_MAP_FULL: Dict[str, int] = {
    expr: idx for idx, expr in enumerate(_MPI_ALL_EXPRESSIONS)
}


def build_mpi_experiment_config(
    label_mode: str = "3class",
    **overrides,
) -> ExperimentConfig:
    """
    Factory: create an ``ExperimentConfig`` pre-configured for the MPI dataset.

    Parameters
    ----------
    label_mode : str
        ``"3class"`` → 3 thesis classes (Positive/Negative/Surprise) for
        direct comparison with CASME-II.
        ``"full"`` → all 51 MPI expression classes.
    **overrides
        Any ``ExperimentConfig`` field to override (e.g., ``epochs=100``).

    Returns
    -------
    ExperimentConfig
        Ready to use with the ablation matrix.
    """
    if label_mode == "3class":
        emotion_map = {
            "Negative": 0,
            "Positive": 1,
            "Surprise": 2,
        }
    elif label_mode == "full":
        emotion_map = dict(MPI_EMOTION_MAP_FULL)
    else:
        raise ValueError(f"Unknown label_mode: {label_mode!r}")

    exp = ExperimentConfig(
        csv_path=PROJECT_ROOT / "Processed_Data" / "mpi_labels.csv",
        tensor_dir_evm=PROJECT_ROOT / "Processed_Data" / "mpi_tensors_evm",
        tensor_dir_raw=PROJECT_ROOT / "Processed_Data" / "mpi_tensors",
        dataset_filter="MPI",
        expression_filter=None,   # MPI has no micro/macro distinction
        emotion_map=emotion_map,
        # MPI has fewer subjects (10) — slightly larger val fraction
        val_fraction=0.2,
        output_root=PROJECT_ROOT / "Ablation_Study" / f"results_mpi_{label_mode}",
        log_dir=PROJECT_ROOT / "Ablation_Study" / f"logs_mpi_{label_mode}",
    )

    # Apply any user overrides
    for key, val in overrides.items():
        if hasattr(exp, key):
            setattr(exp, key, val)
        else:
            raise AttributeError(
                f"ExperimentConfig has no field {key!r}"
            )

    return exp


# ─────────────────────────────────────────────────────────────────────────────
# Standalone sanity print
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 78)
    print("  Ablation Matrix — 8 configurations across 4 phases")
    print("=" * 78)
    header = f"{'#':<3}{'name':<30}{'phase':<7}{'components':<32}{'valid'}"
    print(header)
    print("-" * 78)
    for i, cfg in enumerate(ABLATION_MATRIX, 1):
        print(f"{i:<3}{cfg.name:<30}{cfg.phase:<7}{cfg.describe():<32}{cfg.is_valid()}")
    print("-" * 78)

    exp = ExperimentConfig()
    print(f"\nnum_classes   : {exp.num_classes}")
    print(f"class_names   : {exp.class_names}")
    print(f"EVM tensors   : {exp.tensor_dir_for(True)}")
    print(f"RAW tensors   : {exp.tensor_dir_for(False)}")

    print(f"\n--- MPI Label Maps ---")
    from collections import Counter
    c3 = Counter(MPI_EMOTION_MAP_3CLASS.values())
    print(f"MPI 3-class distribution: {dict(c3)}")
    print(f"MPI full-class count:     {len(MPI_EMOTION_MAP_FULL)} expressions")

    mpi_exp = build_mpi_experiment_config("3class")
    print(f"\nMPI 3-class config:")
    print(f"  num_classes : {mpi_exp.num_classes}")
    print(f"  class_names : {mpi_exp.class_names}")
    print(f"  csv_path    : {mpi_exp.csv_path}")
    print(f"  tensor_raw  : {mpi_exp.tensor_dir_for(False)}")

    mpi_full = build_mpi_experiment_config("full")
    print(f"\nMPI full config:")
    print(f"  num_classes : {mpi_full.num_classes}")

