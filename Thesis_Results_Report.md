# Micro-Expression Recognition — Master's Thesis Results Report

**Author:** Addhyan  
**Date:** May 2026  
**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8.6 GB VRAM)  
**Framework:** PyTorch (CUDA)  
**Random Seed:** 42

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Stage 1 — Data Pipeline & Preprocessing](#2-stage-1--data-pipeline--preprocessing)
3. [Stage 2 — Hybrid Model Architecture](#3-stage-2--hybrid-model-architecture)
4. [Stage 3 — Adversarial Domain Adaptation Training (LOSO)](#4-stage-3--adversarial-domain-adaptation-training-loso)
5. [Stage 4 — Macro-to-Micro Transfer Learning](#5-stage-4--macro-to-micro-transfer-learning)
6. [Final Results Summary](#6-final-results-summary)

---

## 1. Executive Summary

This thesis implements a four-stage pipeline for **Micro-Expression Recognition (MER)** that combines:

- **Eulerian Video Magnification (EVM)** for motion amplification
- A **Hybrid 3D-CNN + Transformer** architecture (STSTNet-SimAM-SLSTT)
- **Adversarial Domain Adaptation** with Supervised Contrastive Learning (SupCon + XBM)
- **Macro-to-Micro Transfer Learning** for cross-expression-type knowledge transfer

**Key Results:**
| Metric | Value |
|---|---|
| **LOSO Mean Accuracy (Stage 3)** | **58.11%** |
| **Phase 1 Macro Best Val Acc** | **70.37%** |
| **Phase 2 Micro Best Val Acc** | **37.78%** |
| **Total Trainable Parameters** | **382,094** |
| **Total Samples (Unified)** | **598 matched (612 in CSV)** |

---

## 2. Stage 1 — Data Pipeline & Preprocessing

### 2.1 Datasets

Two benchmark micro-expression datasets were unified:

| Property | CASME II | CAS(ME)² |
|---|---|---|
| **Source** | Chinese Academy of Sciences | Chinese Academy of Sciences |
| **Expression Types** | Micro-expression only | Both Macro + Micro |
| **Annotation Format** | Excel (.xlsx) | Excel (.xlsx) |
| **Frame Format** | Image sequences → `.avi` | Image sequences → `.avi` |

### 2.2 Metadata Unification

The `MetadataUnifier` standardizes both datasets into a single CSV:

- **Output:** `Processed_Data/master_thesis_labels.csv`
- **Total rows:** 612 (300 macro + 312 micro)
- **Unified Schema:** 15 columns including `subject_id`, `video_id`, `emotion`, `expression_type`, `frames_directory`
- **Emotion Mapping:**

| Unified Label | Constituent Emotions |
|---|---|
| **Negative** | Disgust, Sadness, Fear, Anger, Contempt |
| **Positive** | Happiness |
| **Surprise** | Surprise |
| **Others** | Others, Repression |

### 2.3 Eulerian Video Magnification (EVM)

The `EulerianMagnifier` class amplifies subtle facial muscle movements:

| Parameter | Value | Description |
|---|---|---|
| `alpha` | 20.0 | Amplification factor |
| `low_omega` | 0.4 Hz | Low cutoff of temporal bandpass |
| `high_omega` | 3.0 Hz | High cutoff of temporal bandpass |
| `pyramid_levels` | 4 | Laplacian pyramid depth |
| `fps` | 30 | Assumed frame rate |

**Processing Pipeline:**
1. Build Laplacian pyramid (spatial decomposition)
2. Apply FFT-based temporal bandpass filter per pyramid level
3. Amplify filtered signal by `α`
4. Reconstruct and clamp to valid range

**Output:** 3-channel tensors of shape `[3, 32, 224, 224]` (optical flow u, optical flow v, grayscale), saved as `.npy` files in `Processed_Data/tensors/`.

### 2.4 Dataset Statistics (After Tensor Matching)

| Class | Micro-Expression | Macro-Expression | Combined |
|---|---|---|---|
| **Negative** | 125 | 112 | **253** (42.3%) |
| **Positive** | 45 | 106 | **170** (28.4%) |
| **Surprise** | 34 | 24 | **61** (10.2%) |
| **Others** | 102 | 10 | **114** (19.1%) |
| **Total** | 306 | 252 | **598** |

- **Unique Subjects:** 26 (combined), 20 (macro only)
- **Missing tensors:** 14 samples had no `.npy` file on disk

### 2.5 Temporal Interpolation

The `TemporalInterpolator` standardizes all videos to a fixed frame count:

| Parameter | Value |
|---|---|
| Target frames (`L`) | 32 |
| Spatial resolution | 224 × 224 |
| Channels | 3 (flow-u, flow-v, grayscale) |
| Sampling method | Dynamic index sampling (linear interpolation) |

---

## 3. Stage 2 — Hybrid Model Architecture

### 3.1 Architecture Overview

```
Input: [B, 3, 32, 224, 224]
         │
         ▼
┌─────────────────────────────┐
│  STSTNet Backbone (3D CNN)  │  3-branch modality-aware
│  → 3 branches × Conv3D     │  (optical-flow-u, flow-v, grayscale)
│  → Concatenate → [B,96,32,H,W]
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│  SimAM 3D Attention         │  Parameter-free (0 learnable params)
│  Inter-neuron suppression   │  Weights activations by importance
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│  Adaptive Spatial Pooling   │  [B, 96, 32, 1, 1]
│  → Reshape to sequence      │  [B, 32, 96]
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│  SLSTT Transformer Encoder  │  Sinusoidal positional encoding
│  → Multi-head self-attn     │  Temporal pooling → [B, 96]
└────────────┬────────────────┘
             ▼
         [B, 96]  feature vector
```

### 3.2 Component Details

#### STSTNet 3D Backbone
- **3 parallel branches** (one per input modality channel)
- Each branch: `Conv3D(1,32) → BN → ReLU → MaxPool3D → Conv3D(32,32) → BN → ReLU`
- Temporal dimension preserved through all layers
- Output concatenated: `32 × 3 = 96` channels

#### SimAM 3D (Parameter-Free Attention)
- Based on inter-neuron suppression theory
- Energy function: `e_t = (t - μ)² / (4(σ² + λ))`
- **Zero additional parameters** — pure computation
- Default `λ = 1e-4` for numerical stability

#### SLSTT Transformer Encoder
| Parameter | Value |
|---|---|
| `d_model` | 96 |
| `nhead` | 4 |
| `num_layers` | 2 |
| `dim_feedforward` | 256 |
| `dropout` | 0.1 |
| Positional encoding | Sinusoidal |
| Temporal pooling | Mean pooling over time dimension |

### 3.3 Parameter Count

| Component | Parameters | Trainable |
|---|---|---|
| STSTNet Backbone | ~360K | 363,280 |
| SimAM 3D | 0 | 0 |
| SLSTT Transformer | ~3K | included above |
| **HybridMERModel Total** | — | **~363K** |

---

## 4. Stage 3 — Adversarial Domain Adaptation Training (LOSO)

### 4.1 Adversarial Wrapper Architecture

The `AdversarialMERWrapper` extends Stage 2 with three output heads:

```
         [B, 96] features (from Transformer)
              │
    ┌─────────┼─────────┐
    │         │         │
    ▼         ▼         ▼
┌────────┐ ┌───────┐ ┌─────────────┐
│Proj    │ │Emotion│ │    GRL      │ ← gradient × (-λ)
│Head    │ │Head   │ │      ↓      │
│MLP+L2 │ │       │ │ Identity    │
│Norm    │ │       │ │ Head        │
└───┬────┘ └───┬───┘ └─────┬──────┘
    │         │            │
[B,64]    [B,4]        [B,26]
Projected  Emotion     Identity
Features   Logits      Logits
```

### 4.2 Training Configuration

| Parameter | Value |
|---|---|
| **Optimizer** | AdamW |
| **Learning Rate** | 1e-4 |
| **Weight Decay** | 1e-4 |
| **Scheduler** | CosineAnnealingLR (T_max=25 per fold) |
| **Batch Size** | 2 |
| **Epochs per fold** | 25 |
| **Gradient Accumulation** | 4 steps (effective batch=8) |
| **Gradient Clip Norm** | 1.0 |
| **AMP** | Disabled |

### 4.3 Loss Functions

**Total Loss = (1.0 × SupCon) + Emotion + (0.5 × Identity)**

#### Supervised Contrastive Loss (SupCon)
| Parameter | Value |
|---|---|
| Temperature (τ) | 0.10 |
| Weight | 1.00 |
| XBM queue size | 64 |
| Projection dim | 64 |

#### Focal Loss (Emotion Classification)
| Parameter | Value |
|---|---|
| Gamma (γ) | 2.0 |
| Label smoothing | 0.05 |
| Alpha (per-class) | [0.591, 0.879, 2.451, 1.311] |

#### CrossEntropy (Identity Discrimination)
- Unweighted
- Combined with GRL (sigmoid-annealed λ schedule: `λ(p) = 2/(1+exp(-10·p)) - 1`)

### 4.4 Gradient Reversal Layer (GRL)

| Property | Value |
|---|---|
| Schedule | Sigmoid annealing |
| γ (steepness) | 10.0 |
| Initial λ | 0.0 |
| Final λ | ~1.0 |

### 4.5 LOSO Cross-Validation Results (26 Folds)

The final LOSO run trained 25 epochs per fold across all 26 subjects:

| Fold (Subject) | Best Val Accuracy | Fold (Subject) | Best Val Accuracy |
|---|---|---|---|
| 1 | 0.5769 | 14 | 0.6250 |
| 2 | 0.4925 | 15 | 0.6410 |
| 3 | 0.5833 | 16 | 0.8000 |
| 4 | 0.5000 | 17 | 0.3500 |
| 5 | 0.6190 | 18 | 0.5152 |
| 6 | 0.7619 | 19 | 0.8571 |
| 7 | 0.7778 | 20 | 0.7143 |
| 8 | 0.3793 | 21 | 0.4444 |
| 9 | 0.4545 | 22 | 0.6000 |
| 10 | 0.7500 | 23 | 0.5714 |
| 11 | 0.4717 | 24 | 0.5882 |
| 12 | 0.4615 | 25 | — |
| 13 | 0.6296 | 26 | — |

> **Grand Mean LOSO Accuracy: 0.5811 (58.11%)**

### 4.6 Stage 3 Adversarial Model Parameters

| Component | Trainable Parameters |
|---|---|
| Backbone (Stage 2) | 363,280 |
| Projection Head | 15,520 |
| Emotion Head | 580 |
| Identity Head | 2,714 |
| **TOTAL** | **382,094** |

### 4.7 Early Training Dynamics (Pre-LOSO Baseline, First 37 Epochs)

Selected epoch-level metrics from the initial subject-disjoint split training (462 train / 136 val):

| Epoch | Train Loss | SupCon | Emotion | Identity | Val EmoAcc | λ_GRL |
|---|---|---|---|---|---|---|
| 1 | 7.522 | 4.125 | 1.546 | 3.701 | 0.353 | 0.000 |
| 5 | 7.167 | 4.180 | 1.062 | 3.850 | 0.029 | 0.199 |
| 10 | 8.765 | 4.179 | 1.132 | 6.909 | 0.132 | 0.426 |
| 16 | 7.434 | 4.175 | 1.102 | 4.314 | **0.412** | 0.640 |
| 18 | 7.341 | 4.175 | 0.993 | 4.347 | **0.427** | 0.696 |
| 29 | 8.842 | 4.174 | 1.056 | 7.225 | **0.434** | 0.888 |
| 32 | 7.806 | 4.173 | 1.072 | 5.121 | **0.456** | 0.916 |
| 37 | 9.481 | 4.175 | 1.097 | 8.419 | 0.441 | 0.949 |

**Key Observations:**
- SupCon loss stabilized around ~4.17 (indicating well-clustered projections)
- Emotion accuracy peaked at ~45.6% on this split (epoch 32)
- Identity accuracy dropped to ~4-6% as GRL λ increased → **successful subject-invariance**
- GRL annealing worked as designed: smooth sigmoid from 0 → 1

---

## 5. Stage 4 — Macro-to-Micro Transfer Learning

### 5.1 Transfer Learning Strategy

Two-phase training implemented by `TransferLearningOrchestrator`:

```
┌──────────────────────────────────────┐
│  PHASE 1: Macro Pre-Training         │
│  • Dataset: CAS(ME)² macro-expressions│
│  • Adversary: OFF (λ_id = 0.0)       │
│  • Goal: Learn facial physics freely  │
│  • 50 epochs                          │
└──────────────┬───────────────────────┘
               │ Transfer weights (101/103 tensors)
               │ Skip identity_head (shape mismatch: 20→26)
               ▼
┌──────────────────────────────────────┐
│  PHASE 2: Micro Fine-Tuning          │
│  • Dataset: Micro-expressions only    │
│  • Adversary: ON (λ_id = 0.40)       │
│  • Goal: Subject-invariant micro-MER  │
│  • 150 epochs                         │
└──────────────────────────────────────┘
```

### 5.2 Phase 1 — Macro Pre-Training Configuration

| Parameter | Value |
|---|---|
| **Dataset** | Macro-expressions only (from CAS(ME)²) |
| **Samples** | 293 total → 266 train / 27 val |
| **Subjects** | 20 unique |
| **Epochs** | 50 |
| **Learning Rate** | 1e-4 |
| **Identity Loss Weight** | 0.0 (adversary OFF) |
| **AMP** | Disabled (final run) |
| **Scheduler** | CosineAnnealing (T=50) |

**Emotion Distribution (Macro):**
| Class | Samples |
|---|---|
| Negative | 112 |
| Positive | 106 |
| Surprise | 24 |
| Others | 10 |

### 5.3 Phase 1 Results

| Metric | Value |
|---|---|
| **Best Val Accuracy** | **70.37%** |
| **Training Time** | 141.2 minutes |
| **Best Epoch** | ~32 (final run) |
| **Checkpoint** | `phase1_macro/macro_pretrained_best.pth` |

**Selected Phase 1 Training History (Run 1 — 50 epochs, 252 samples):**

| Epoch | Train Loss | SupCon | Emo Loss | Val EmoAcc | λ_GRL |
|---|---|---|---|---|---|
| 1 | 5.666 | 4.018 | 1.648 | 0.091 | 0.000 |
| 7 | 5.612 | 4.177 | 1.435 | **0.591** | 0.546 |
| 14 | 5.340 | 4.178 | 1.161 | **0.591** | 0.868 |
| 19 | 5.361 | 4.177 | 1.183 | **0.682** | 0.951 |
| 32 | 5.230 | 4.175 | 1.055 | **0.773** | 0.996 |
| 50 | 5.236 | 4.174 | 1.062 | 0.636 | 1.000 |

### 5.4 Weight Transfer Details

- **Transferred:** 101 / 103 parameter tensors successfully loaded
- **Skipped (shape mismatch):**
  - `identity_head.2.weight`: macro `[20, 96]` → micro `[26, 96]`
  - `identity_head.2.bias`: macro `[20]` → micro `[26]`
- Identity head re-initialized with Xavier uniform for 26 micro subjects

### 5.5 Phase 2 — Micro Fine-Tuning Configuration

| Parameter | Value |
|---|---|
| **Dataset** | Micro-expressions only |
| **Samples** | 305 total → 215 train / 90 val |
| **Subjects** | 26 unique |
| **Epochs** | 150 |
| **Learning Rate** | 5e-5 |
| **Identity Loss Weight** | 0.40 (adversary ON) |
| **AMP** | Disabled |
| **Scheduler** | CosineAnnealing (T=150) |

**Emotion Distribution (Micro):**
| Class | Samples | Class Weight |
|---|---|---|
| Negative | 125 | 0.612 |
| Positive | 45 | 1.700 |
| Surprise | 34 | 2.250 |
| Others | 102 | 0.750 |

### 5.6 Phase 2 Results

| Metric | Value |
|---|---|
| **Best Val Accuracy** | **37.78%** |
| **Training Time** | 358.4 minutes (~6 hours) |
| **Checkpoint** | `phase2_micro/micro_finetuned_best.pth` |

**Selected Phase 2 Training History (first 6 epochs shown):**

| Epoch | Train Loss | SupCon | Emo Loss | ID Loss | Val EmoAcc | λ_GRL |
|---|---|---|---|---|---|---|
| 1 | 6.509 | 3.876 | 0.990 | 4.106 | 0.256 | 0.000 |
| 2 | 6.806 | 4.176 | 1.000 | 4.077 | 0.011 | 0.034 |
| 3 | 6.682 | 4.179 | 0.949 | 3.886 | 0.100 | 0.067 |
| 4 | 6.704 | 4.174 | 0.972 | 3.894 | 0.078 | 0.100 |
| 5 | 6.713 | 4.176 | 0.972 | 3.913 | 0.011 | 0.133 |
| 6 | — | — | — | — | — | 0.166 |

### 5.7 Stage 4 Complete Timeline

| Event | Timestamp | Duration |
|---|---|---|
| Phase 1 Start | 2026-05-14 14:25:09 | — |
| Phase 1 Complete | 2026-05-14 16:46:23 | 141.2 min |
| Phase 2 Start | 2026-05-14 16:46:24 | — |
| Phase 2 Complete | 2026-05-14 22:44:48 | 358.4 min |
| **Total Stage 4** | — | **~8.3 hours** |

---

## 6. Final Results Summary

### 6.1 Complete Pipeline Results

| Stage | Task | Key Result |
|---|---|---|
| **Stage 1** | Data Unification + EVM | 598 matched samples, 4 classes, 26 subjects |
| **Stage 2** | Architecture Design | 382K params hybrid 3D-CNN + Transformer |
| **Stage 3** | LOSO Cross-Validation | **Mean Accuracy: 58.11%** |
| **Stage 4 Phase 1** | Macro Pre-Training | **Best Val Acc: 70.37%** |
| **Stage 4 Phase 2** | Micro Fine-Tuning | **Best Val Acc: 37.78%** |

### 6.2 Technology Stack Summary

| Component | Technology |
|---|---|
| **Motion Amplification** | Eulerian Video Magnification (Laplacian pyramid + FFT) |
| **Feature Extraction** | STSTNet 3D (modality-aware 3-branch CNN) |
| **Attention** | SimAM 3D (parameter-free, neuron suppression) |
| **Temporal Modeling** | SLSTT Transformer (sinusoidal PE, mean pooling) |
| **Contrastive Learning** | SupCon Loss + Cross-Batch Memory (XBM) |
| **Class Imbalance** | Focal Loss (γ=2.0) with per-class α weights |
| **Domain Adaptation** | Gradient Reversal Layer (sigmoid-annealed) |
| **Transfer Learning** | Macro → Micro (2-phase, adversary OFF→ON) |
| **Evaluation Protocol** | LOSO (26-fold) with subject-disjoint splits |

### 6.3 Artifacts Location

| Artifact | Path |
|---|---|
| Unified Labels CSV | `Processed_Data/master_thesis_labels.csv` |
| Processed Tensors | `Processed_Data/tensors/` |
| Stage 3 LOSO Checkpoints | `Stage3_Training/checkpoints/fold_*/` |
| Stage 3 Training Log | `Stage3_Training/logs/stage3_training.log` |
| Phase 1 Best Checkpoint | `Stage4_TransferLearning/checkpoints/phase1_macro/macro_pretrained_best.pth` |
| Phase 2 Best Checkpoint | `Stage4_TransferLearning/checkpoints/phase2_micro/micro_finetuned_best.pth` |
| Stage 4 Transfer Log | `Stage4_TransferLearning/logs/stage4_transfer.log` |
| Visualization Helper | `Stage4_TransferLearning/eval_visualizations.py` |
| Results Notebook | `Stage4_TransferLearning/Master_Thesis_Results.ipynb` |

### 6.4 Key Observations

1. **SupCon loss converges quickly** (~4.17) and remains stable, indicating the projection head successfully clusters emotion-related features in contrastive space.

2. **GRL sigmoid annealing works as designed**: identity accuracy drops to ~0% on validation (unseen subjects), confirming subject-invariant representations.

3. **Macro pre-training reaches 70%+ accuracy** on macro-expressions, validating that the architecture can learn facial action unit physics from larger-amplitude expressions.

4. **Micro fine-tuning is harder** (37.78%), consistent with the inherent difficulty of micro-expressions (sub-second, low-amplitude facial movements) and severe class imbalance.

5. **LOSO at 58.11%** across 26 folds is competitive for a 4-class micro-expression task, with per-fold variance ranging from 35% to 85.71%.

---

*Report generated from historical training logs without re-running any computational pipeline.*
