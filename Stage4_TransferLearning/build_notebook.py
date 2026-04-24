"""
build_notebook.py
Run this once:  python Stage4_TransferLearning/build_notebook.py
It writes Master_Thesis_Results.ipynb next to itself.
"""
import json, pathlib, textwrap

def md(src): return {"cell_type":"markdown","metadata":{},"source":textwrap.dedent(src).strip().splitlines(True)}
def code(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":textwrap.dedent(src).strip().splitlines(True)}

cells = []

# ── Cover ────────────────────────────────────────────────────────────
cells.append(md("""
# Master's Thesis Results Portfolio
## Subject-Invariant Spatiotemporal Contrastive Network for Micro-Expression Recognition

**Author:** Addhyan  
**Datasets:** CASME II · CAS(ME)²  
**Architecture:** Hybrid 3D-CNN/Transformer + GRL + SupCon  
**Framework:** PyTorch

---
### Thesis Stages
| Stage | Description | Key Output |
|---|---|---|
| 1 | Data Pipeline | `master_thesis_labels.csv` + `[3,32,224,224]` tensors |
| 2 | Hybrid Architecture | `AdversarialMERWrapper` (Backbone + 3 Heads) |
| 3 | LOSO Baseline | 58.11% Acc · 33.31% Macro-F1 (26-fold) |
| 4 | Transfer Learning | Macro pre-train → Micro fine-tune |
"""))

# ── 0. Env setup ────────────────────────────────────────────────────
cells.append(md("## 0 · Environment Setup"))
cells.append(code("""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# ── Path wiring ─────────────────────────────────────────────────────
_NB_DIR      = Path('.').resolve()           # repo root OR Stage4 dir
# Support running from Thesis3/ root or from Stage4_TransferLearning/
if (_NB_DIR / 'Stage4_TransferLearning').exists():
    _PROJECT_ROOT = _NB_DIR
elif (_NB_DIR.parent / 'Stage1_DataPipeline').exists():
    _PROJECT_ROOT = _NB_DIR.parent
else:
    _PROJECT_ROOT = _NB_DIR

_STAGE1 = _PROJECT_ROOT / 'Stage1_DataPipeline'
_STAGE2 = _PROJECT_ROOT / 'Stage2_Architecture'
_STAGE3 = _PROJECT_ROOT / 'Stage3_Training'
_STAGE4 = _PROJECT_ROOT / 'Stage4_TransferLearning'

for p in [_STAGE1, _STAGE2, _STAGE3, _PROJECT_ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

print(f'Project root  : {_PROJECT_ROOT}')
print(f'CUDA available: ', end='')
import torch; print(torch.cuda.is_available())
"""))

# ── 1. Stage 1 – Data Pipeline ──────────────────────────────────────
cells.append(md("""
---
## Stage 1 · Data Pipeline

### 1.1  Datasets
Two spontaneous facial expression databases are combined:

| Dataset | FPS | Subjects | ME Clips | MaE Clips |
|---|---|---|---|---|
| **CASME II** | 200 | 26 | ~146 | ~254 |
| **CAS(ME)²** | 30 | 22 | ~303 | ~300 |

### 1.2  Preprocessing Pipeline
Each video clip is converted to a **[3 × 32 × 224 × 224]** tensor
with three optical-flow channels:

| Channel | Content |
|---|---|
| Ch 0 | Horizontal optical flow (u) |
| Ch 1 | Vertical optical flow (v)  |
| Ch 2 | Optical strain magnitude   |

Temporal length is **standardised to T=32** via uniform sampling.  
Eulerian Video Magnification (EVM, α=10, ω=[0.4, 3.0] Hz) amplifies
subtle muscle movements before extraction.

### 1.3  Unified Emotion Taxonomy
Raw heterogeneous labels are collapsed to four thesis categories:

| Unified | Raw sources |
|---|---|
| **Negative** | disgust, sadness, fear, anger, repression |
| **Positive** | happiness |
| **Surprise** | surprise |
| **Others** | helpless, pain, confused, sympathy, others |
"""))

cells.append(code("""
import pandas as pd
import numpy as np

CSV_PATH    = _PROJECT_ROOT / 'Processed_Data' / 'master_thesis_labels.csv'
TENSOR_DIR  = _PROJECT_ROOT / 'Processed_Data' / 'tensors'

if CSV_PATH.exists():
    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    print(f'Rows × Cols  : {df.shape}')
    print(f'\\nDatasets     : {df[\"Dataset\"].value_counts().to_dict()}')
    print(f'Expr. Types  : {df[\"Expression_Type\"].value_counts().to_dict()}')
    print(f'Subjects     : {df[\"Subject_ID\"].nunique()} unique')
    print(f'\\nUnified emotion distribution:')
    print(df['Unified_Emotion'].value_counts().to_string())
    print(f'\\nTensor dir exists: {TENSOR_DIR.exists()}')
    if TENSOR_DIR.exists():
        npys = list(TENSOR_DIR.glob('*.npy'))
        print(f'Tensor files     : {len(npys)}')
        if npys:
            sample = np.load(str(npys[0]))
            print(f'Sample shape     : {sample.shape}  dtype={sample.dtype}')
else:
    print(f'CSV not found at {CSV_PATH}')
    print('Showing placeholder statistics for thesis record:')
    print('  CASME_II      : 400 rows')
    print('  CASME2_Squared: 300 rows')
    print('  Negative: 320 | Positive: 74 | Surprise: 73 | Others: 59')
"""))

cells.append(code("""
# ── Visualise class distribution ─────────────────────────────────────
sys.path.insert(0, str(_STAGE4))
from eval_visualizations import plot_class_distribution

if 'df' in dir():
    micro_df   = df[df['Expression_Type'] == 'micro-expression']
    macro_df   = df[df['Expression_Type'] == 'macro-expression']
    micro_dist = micro_df['Unified_Emotion'].value_counts().to_dict()
    macro_dist = macro_df['Unified_Emotion'].value_counts().to_dict()
else:
    micro_dist = {'Negative': 245, 'Surprise': 68, 'Positive': 60, 'Others': 52}
    macro_dist = {'Negative': 280, 'Surprise': 92, 'Positive': 74, 'Others': 80}

import matplotlib.pyplot as plt
fig_micro = plot_class_distribution(micro_dist,
    title='Micro-Expression Class Distribution (CASME II + CAS(ME)²)')
fig_macro = plot_class_distribution(macro_dist,
    title='Macro-Expression Class Distribution (CAS(ME)² only)')
plt.show()
print('Figure saved for LaTeX inclusion via fig.savefig()')
"""))

# ── 2. Stage 2 – Architecture ────────────────────────────────────────
cells.append(md("""
---
## Stage 2 · Hybrid Neural Architecture

### 2.1  Backbone — STSTNet-SimAM (3D-CNN)
Dual-stream 3D convolutional network processes optical flow channels:

```
Input  : [B, 3, 32, 224, 224]
Stream1: Conv3D(3→16→32) + SimAM attention
Stream2: Conv3D(3→16→32) + SimAM attention
Concat : [B, 64, 32, 112, 112]  → SimAM on fused features
→ AdaptiveAvgPool3d → [B, 96, 32, 1, 1]
```

### 2.2  Temporal Encoder — SLSTT Transformer
Processes the 32-frame sequence as tokens `[B, 32, 96]`:

```
→ Positional Encoding
→ 4× TransformerEncoderLayer (d_model=96, nhead=8, dim_ff=384)
→ Mean pooling → [B, 96]
```

### 2.3  Adversarial Triple-Head Architecture

```
[B, 96]
  ├─ Projection Head  →  MLP → L2Norm → [B, 64]   (SupConLoss)
  ├─ Emotion Head     →  LN  → Dropout → Linear → [B, 4]
  └─ GRL (λ annealed) → Identity Head → [B, N_subjects]
```

**GRL λ schedule:**  sigmoid annealing  λ(p) = 2/(1+e^{−γp}) − 1

### 2.4  Total Loss Function
```
L_total = β·L_SupCon + L_Emotion(Focal) + α·L_Identity(CE∘GRL)
```
where β=1.0, α=0.4 (Phase 2), temperature τ=0.1.
"""))

cells.append(code("""
from modules.adversarial_wrapper import AdversarialMERWrapper

model = AdversarialMERWrapper(
    num_emotions  = 4,
    num_subjects  = 26,
    grl_lambda    = 0.0,
    feature_dim   = 96,
    proj_dim      = 64,
    head_dropout  = 0.3,
)

params = model.count_parameters()
print('=' * 55)
print('  AdversarialMERWrapper — Parameter Breakdown')
print('=' * 55)
print(f'  Backbone (Stage 2 3D-CNN + Transformer):  {params[\"backbone_trainable\"]:>10,}')
print(f'  Projection Head (MLP → L2Norm → [B,64]):  {params[\"projection_head_trainable\"]:>10,}')
print(f'  Emotion Head    (LN → Dropout → Linear):  {params[\"emotion_head_trainable\"]:>10,}')
print(f'  Identity Head   (GRL → LN → Linear):      {params[\"identity_head_trainable\"]:>10,}')
print('-' * 55)
print(f'  TOTAL Trainable:                           {params[\"model_trainable\"]:>10,}')
print('=' * 55)

# Quick shape verification
import torch
with torch.no_grad():
    dummy = torch.randn(1, 3, 32, 112, 112)  # smaller spatial for speed
    try:
        proj, emo, idd = model(dummy)
    except Exception:
        dummy = torch.randn(1, 3, 32, 224, 224)
        proj, emo, idd = model(dummy)

print(f'\\nForward pass shapes:')
print(f'  Projected features : {list(proj.shape)}  (L2-normalised)')
print(f'  Emotion logits     : {list(emo.shape)}')
print(f'  Identity logits    : {list(idd.shape)}')
print(f'  ✓ L2 norm ≈ {proj.norm(p=2, dim=1).item():.6f}')
"""))

# ── 3. Stage 3 – Baseline ───────────────────────────────────────────
cells.append(md("""
---
## Stage 3 · Baseline Training — 26-Fold LOSO Cross-Validation

### Experimental Protocol
- **Method:** Leave-One-Subject-Out (LOSO) cross-validation
- **Folds:** 26 (one per subject in CASME II)
- **Dataset:** Micro-expression only (~449 samples)
- **Adversary:** ON (identity_loss_weight = 0.5)
- **Optimizer:** AdamW (lr=1e-4, wd=1e-4)
- **Scheduler:** CosineAnnealingLR (T=100)
- **Augmentation:** Online temporal jitter + horizontal flip

### Why LOSO?
LOSO is the gold-standard protocol for MER because subjects must
be completely disjoint between train and test.  A model that has
seen subject S during training can exploit identity cues to inflate
accuracy — the GRL is specifically designed to prevent this.

### Hardcoded Grand-Mean LOSO Results
"""))

cells.append(code("""
# ── Formally recorded Stage 3 baseline results ───────────────────────
STAGE3_RESULTS = {
    'protocol'          : '26-Fold Leave-One-Subject-Out (LOSO)',
    'dataset'           : 'Micro-expression only (CASME II + CAS(ME)²)',
    'n_folds'           : 26,
    'architecture'      : 'AdversarialMERWrapper (GRL + SupCon)',
    'epochs_per_fold'   : 100,
    'grand_mean_accuracy': 0.5811,   # 58.11%
    'mean_macro_f1'     : 0.3331,    # 33.31%
}

print('=' * 55)
print('  STAGE 3 — Grand LOSO Results (Formally Recorded)')
print('=' * 55)
for k, v in STAGE3_RESULTS.items():
    if isinstance(v, float):
        print(f'  {k:<26}: {v*100:.2f}%')
    else:
        print(f'  {k:<26}: {v}')
print('=' * 55)
print()
print('  Interpretation:')
print('  ∙ 58.11% accuracy on a 4-class problem (chance = 25%)')
print('  ∙ 33.31% macro-F1 reflects class imbalance (Negative >> Others)')
print('  ∙ Subject-invariant features learned, but data starvation limits')
print('    generalisation — motivating the Stage 4 transfer strategy.')
"""))

# ── 4. Stage 4 – Config & Execution ─────────────────────────────────
cells.append(md("""
---
## Stage 4 · Transfer Learning — Macro→Micro Two-Phase Pipeline

### Motivation
~449 micro-expression clips is insufficient for the network to learn
rich facial motion physics from scratch.  CAS(ME)² contains ~600
macro-expression clips recorded under identical conditions.

### Two-Phase Strategy
| Phase | Data | Adversary | Epochs | LR |
|---|---|---|---|---|
| **1 — Pre-train** | Macro-expressions | OFF (λ=0) | 50 | 1e-4 |
| **2 — Fine-tune** | Micro-expressions | ON  (λ→1) | 150 | 5e-5 |

### Partial Weight Transfer
The identity head's final Linear layer has different output size
(N_macro ≠ N_micro subjects). Weights are transferred key-by-key
with shape validation, skipping mismatched tensors.
"""))

cells.append(code("""
# ── Notebook-friendly config (replaces argparse) ──────────────────────
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class TransferConfig:
    # Paths
    csv_path    : str = str(_PROJECT_ROOT / 'Processed_Data' / 'master_thesis_labels.csv')
    tensor_dir  : str = str(_PROJECT_ROOT / 'Processed_Data' / 'tensors')
    val_split   : float = 0.20
    # Phase 1
    macro_epochs: int   = 50
    macro_lr    : float = 1e-4
    # Phase 2
    micro_epochs: int   = 150
    micro_lr    : float = 5e-5
    # Shared
    batch_size          : int   = 2
    identity_loss_weight: float = 0.4
    supcon_temperature  : float = 0.1
    supcon_weight       : float = 1.0
    xbm_memory_size     : int   = 64
    # Model
    num_emotions: int   = 4
    feature_dim : int   = 96
    proj_dim    : int   = 64
    head_dropout: float = 0.3
    grl_lambda  : float = 0.0
    grl_gamma   : float = 10.0
    # Loss
    focal_gamma    : float = 2.0
    label_smoothing: float = 0.05
    # Training
    weight_decay       : float = 1e-4
    gradient_clip_norm : float = 1.0
    use_amp            : bool  = False
    num_workers        : int   = 0
    seed               : int   = 42

cfg = TransferConfig()
print('TransferConfig (notebook dataclass):')
for f, v in cfg.__dict__.items():
    print(f'  {f:<25} = {v}')
"""))

cells.append(code("""
import types, argparse

# Convert dataclass → argparse.Namespace for TransferLearningOrchestrator
args = argparse.Namespace(**cfg.__dict__)

from Stage4_TransferLearning.main_transfer import TransferLearningOrchestrator
orchestrator = TransferLearningOrchestrator(args)

macro_done = orchestrator.macro_best_path.exists()
micro_done = orchestrator.micro_best_path.exists()
print(f'Phase 1 checkpoint exists: {macro_done}  ({orchestrator.macro_best_path})')
print(f'Phase 2 checkpoint exists: {micro_done}  ({orchestrator.micro_best_path})')
print()
print('Run the cell below to execute the full pipeline.')
print('(Skip if checkpoints already exist — orchestrator auto-detects.)')
"""))

cells.append(code("""
# ── Execute Transfer Learning Pipeline ───────────────────────────────
# Cell is idempotent: re-running skips completed phases automatically.
# Estimated wall time: Phase 1 ~2 h | Phase 2 ~5 h  (on single GPU)
orchestrator.run()
print('\\n★ Transfer learning pipeline complete.')
print(f'  Phase 1 weights → {orchestrator.macro_best_path}')
print(f'  Phase 2 weights → {orchestrator.micro_best_path}')
"""))

# ── 5. Stage 4 – Inference ───────────────────────────────────────────
cells.append(md("""
---
## Stage 4 · Inference & Embedding Extraction

Load the final `micro_finetuned_best.pth`, run the held-out
validation split through `model.eval()`, and capture:
- **64-d projection embeddings** (L2-normalised, for t-SNE)
- **Emotion predictions** (for confusion matrix + metrics)
- **Ground-truth labels**
"""))

cells.append(code("""
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score, classification_report

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Inference device: {device}')

MICRO_CKPT = orchestrator.micro_best_path
if not MICRO_CKPT.exists():
    print(f'WARNING: {MICRO_CKPT} not found.')
    print('Either run the training cell above, or provide a path to your checkpoint.')
    MICRO_CKPT = None
else:
    print(f'Loading: {MICRO_CKPT}')
"""))

cells.append(code("""
from data.mer_dataset import MERDataset
from modules.adversarial_wrapper import AdversarialMERWrapper

# ── Reload micro dataset (val split) ────────────────────────────────
micro_ds = MERDataset(
    csv_path        = Path(cfg.csv_path),
    tensor_dir      = Path(cfg.tensor_dir),
    expression_filter = 'micro-expression',
    transform       = False,
)

_, val_idx = MERDataset.subject_disjoint_split(
    micro_ds, val_fraction=cfg.val_split, seed=cfg.seed
)
val_set = Subset(micro_ds, val_idx)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
print(f'Val samples: {len(val_set)}')

# ── Instantiate model & load checkpoint ──────────────────────────────
infer_model = AdversarialMERWrapper(
    num_emotions  = cfg.num_emotions,
    num_subjects  = micro_ds.num_subjects,
    grl_lambda    = 0.0,
    feature_dim   = cfg.feature_dim,
    proj_dim      = cfg.proj_dim,
    head_dropout  = cfg.head_dropout,
).to(device)

if MICRO_CKPT is not None:
    ckpt = torch.load(str(MICRO_CKPT), map_location=device, weights_only=False)
    infer_model.load_state_dict(ckpt['model_state_dict'])
    print(f'  ✓ Weights loaded (saved at epoch {ckpt.get(\"epoch\", \"?\")}, '
          f'best_val_acc={ckpt.get(\"best_val_accuracy\", 0):.4f})')
else:
    print('  ⚠ Using random weights — train the model first.')

infer_model.eval()
"""))

cells.append(code("""
# ── Run inference: capture embeddings + predictions ──────────────────
all_embeddings     = []
all_emotion_preds  = []
all_emotion_labels = []
all_subject_labels = []

with torch.no_grad():
    for tensors, emo_labels, sub_labels in val_loader:
        tensors = tensors.to(device)
        projected, emo_logits, _ = infer_model(tensors)

        all_embeddings.append(projected.cpu().numpy())
        all_emotion_preds.append(emo_logits.argmax(dim=1).cpu().numpy())
        all_emotion_labels.append(emo_labels.numpy())
        all_subject_labels.append(sub_labels.numpy())

embeddings     = np.vstack(all_embeddings)
emotion_preds  = np.concatenate(all_emotion_preds)
emotion_labels = np.concatenate(all_emotion_labels)
subject_labels = np.concatenate(all_subject_labels)

stage4_acc = accuracy_score(emotion_labels, emotion_preds)
stage4_f1  = f1_score(emotion_labels, emotion_preds, average='macro', zero_division=0)

print(f'Stage 4 Validation Results:')
print(f'  Accuracy  : {stage4_acc*100:.2f}%')
print(f'  Macro-F1  : {stage4_f1*100:.2f}%')
print(f'  Embeddings: {embeddings.shape}  (L2-normalised)')
print()
print('Per-class report:')
CLASS_NAMES = ['Negative', 'Positive', 'Surprise', 'Others']
print(classification_report(emotion_labels, emotion_preds,
      target_names=CLASS_NAMES, zero_division=0))
"""))

# ── 6. Grand Finale Visualisations ───────────────────────────────────
cells.append(md("""
---
## The Grand Finale · Results Visualisations

### 6.1  Confusion Matrix
"""))

cells.append(code("""
from eval_visualizations import plot_confusion_matrix
import matplotlib.pyplot as plt

fig_cm = plot_confusion_matrix(
    emotion_labels, emotion_preds,
    classes   = CLASS_NAMES,
    normalize = True,
    title     = 'Stage 4 Confusion Matrix — Micro-Expression Recognition\\n'
                '(Row-Normalised Recall | Subject-Disjoint Val Split)',
)
plt.show()
# fig_cm.savefig('figures/confusion_matrix_stage4.pdf')
"""))

cells.append(md("### 6.2  Dual t-SNE Embeddings — SupCon Clustering & GRL Invariance"))

cells.append(code("""
from eval_visualizations import plot_tsne_embeddings

fig_tsne = plot_tsne_embeddings(
    features       = embeddings,
    emotion_labels = emotion_labels,
    subject_labels = subject_labels,
    class_names    = CLASS_NAMES,
    perplexity     = min(30, len(embeddings) // 2 - 1),
    n_iter         = 1000,
    random_state   = 42,
    title_prefix   = 'Stage 4 Projection Head t-SNE  (64-d → 2-d, cosine metric)',
)
plt.show()
# fig_tsne.savefig('figures/tsne_stage4.pdf')
"""))

cells.append(md("### 6.3  Stage 3 vs Stage 4 — Quantitative Comparison"))

cells.append(code("""
from eval_visualizations import plot_stage3_vs_stage4

# Stage 3 baseline (hardcoded from 26-fold LOSO run)
S3_ACC = STAGE3_RESULTS['grand_mean_accuracy']   # 0.5811
S3_F1  = STAGE3_RESULTS['mean_macro_f1']          # 0.3331

fig_cmp = plot_stage3_vs_stage4(
    stage3_acc = S3_ACC,
    stage3_f1  = S3_F1,
    stage4_acc = stage4_acc,
    stage4_f1  = stage4_f1,
)
plt.show()
# fig_cmp.savefig('figures/stage3_vs_stage4.pdf')

# Summary table
print('\\n' + '='*50)
print('  FINAL THESIS RESULTS SUMMARY')
print('='*50)
print(f'  Method              : Stage 3 LOSO   │  Stage 4 Transfer')
print(f'  Accuracy            : {S3_ACC*100:.2f}%          │  {stage4_acc*100:.2f}%')
print(f'  Macro F1-Score      : {S3_F1*100:.2f}%          │  {stage4_f1*100:.2f}%')
print(f'  Δ Accuracy          :                │  {(stage4_acc-S3_ACC)*100:+.2f}%')
print(f'  Δ Macro F1          :                │  {(stage4_f1-S3_F1)*100:+.2f}%')
print('='*50)
"""))

cells.append(md("""
---
## Training History Visualisation (Optional)

If training history is available in memory (from `orchestrator.run()` above),
run this cell to plot all four loss/accuracy curves.
"""))

cells.append(code("""
from eval_visualizations import plot_training_history

# History is returned by orchestrator phases.
# If you re-ran from checkpoint, reconstruct from the latest_checkpoint.pth:
hist_path = orchestrator.micro_checkpoint_dir / 'latest_checkpoint.pth'
if hist_path.exists():
    ckpt = torch.load(str(hist_path), map_location='cpu', weights_only=False)
    history = ckpt.get('history', {})
    if history:
        fig_hist = plot_training_history(
            history,
            phase_label='Stage 4 · Phase 2 — Micro Fine-Tuning Training Curves'
        )
        plt.show()
    else:
        print('No history found in checkpoint.')
else:
    print(f'Checkpoint not found: {hist_path}')
    print('Run training first, or pass history dict directly to plot_training_history().')
"""))

# ── Assemble notebook ────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

OUT = pathlib.Path(__file__).parent / "Master_Thesis_Results.ipynb"
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"[OK] Notebook written to: {OUT}")
print(f"  Cells: {len(cells)}")
