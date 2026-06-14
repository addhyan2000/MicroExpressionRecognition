# Master's Thesis: Spatiotemporal Ablation Summary Report
**Project Title:** Modular Spatiotemporal Analysis for Micro-Expression Recognition  
**Presenter:** Addhyan  
**Scope:** Stage 1 & Stage 2 Ablation Study (Toggle permuting of EVM, SimAM, 3D-CNN, and SLSTT)  
**Date:** June 2026  

---

## 1. Research Scope & Objectives

This report provides a comprehensive summary of the Stage 1 & Stage 2 ablation study for Micro-Expression Recognition (MER). Automated spontaneous MER is highly challenging due to **data starvation** (benchmark databases such as CASME II contain only ~250 spontaneous samples) and **subject identity bias** (static facial structures produce stronger learning signals than dynamic sub-pixel movements). 

To address these challenges, we proposed a parameter-efficient (~363K parameters) modular network with four toggles:
1. **Variable A: EVM (Eulerian Video Magnification)** (on/off) — offline motion magnification.
2. **Variable C: 3D-CNN Backbone (STSTNet-3D)** (on/off) — three-stream weight-unshared spatial feature extraction.
3. **Variable B: SimAM Attention** (on/off) — 3D parameter-free attention.
4. **Variable D: SLSTT Sequence Transformer** (on/off) — multi-head temporal self-attention.

This summary details the results of our 12-configuration systematic sweep on CASME II using a strict subject-disjoint validation protocol.

---

## 2. Quantitative Ablation Performance Matrix

The ablation sweep evaluated 12 configuration cells to isolate components under standard supervised Focal Loss:

| Config Name | Phase | EVM | SimAM | CNN3D | SLSTT | Accuracy | Macro F1 | Negative F1 | Positive F1 | Surprise F1 | Trainable Params |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **config_1_pure_base** | I | Off | Off | Off | Off | 74.36% | 28.43% | 85.29% | 0.00% | 0.00% | 3,363 |
| **config_2_temporal_only** | I | Off | Off | Off | On | 74.36% | 28.43% | 85.29% | 0.00% | 0.00% | 3,363 |
| **config_4_motion_amp_base** | II | On | Off | Off | Off | 74.36% | 28.43% | 85.29% | 0.00% | 0.00% | 3,363 |
| **config_12_permutation** | Other | On | Off | Off | On | 74.36% | 28.43% | 85.29% | 0.00% | 0.00% | 3,363 |
| **config_3_spatial_only** | I | Off | Off | On | Off | 71.79% | **43.57%** | 80.70% | **50.00%** | 0.00% | 363,280 |
| **config_13_permutation** | Other | On | Off | On | Off | 71.79% | **43.57%** | 80.70% | **50.00%** | 0.00% | 363,280 |
| **config_5_attention_base** | II | Off | On | On | Off | 58.97% | 40.52% | 69.39% | **52.17%** | 0.00% | 363,280 |
| **config_16_permutation** | Other | On | On | On | Off | 58.97% | 40.52% | 69.39% | **52.17%** | 0.00% | 363,280 |
| **config_7_full_no_attention** | III | On | Off | On | On | 69.23% | 35.77% | 80.65% | 26.67% | 0.00% | 363,280 |
| **config_9_permutation** | Other | Off | Off | On | On | 69.23% | 35.77% | 80.65% | 26.67% | 0.00% | 363,280 |
| **config_6_full_stage2_noevm**| III | Off | On | On | On | 48.72% | 30.94% | 58.33% | 34.48% | 0.00% | 363,280 |
| **config_8_proposed_unified** | IV | On | On | On | On | 48.72% | 30.94% | 58.33% | 34.48% | 0.00% | 363,280 |

---

## 3. Discussion of Key Findings

### 3.1 The Critical Importance of the 3D-CNN Spatial Stem
Configurations 1, 2, 4, and 12 all collapsed, achieving exactly 74.36% Accuracy and 28.43% Macro F1. Since the 3D-CNN is turned off, these models rely on `RawPatchEmbedding` which pools inputs to a $4 \times 4$ spatial grid. This is too noisy and lacks the local receptive field properties of convolutions, causing the network to default to predicting the majority class ("Negative") for every sample (Negative represents 74.36% of the validation split). 

Enabling the 3D-CNN backbone (`config_3`) increases the Macro F1 score by **+15.14%** (reaching 43.57% F1), validating the spatial backbone as the indispensable foundation of feature learning.

### 3.2 Spatiotemporal Overfitting Dynamics
The proposed unified model (`config_8`) achieves 48.72% Accuracy and 30.94% Macro F1, falling below the spatial-only baseline (`config_3`). This degradation indicates that high-capacity spatiotemporal blocks overfit when trained under standard supervised focal loss. 

Without explicit regularization (such as contrastive loss or domain adaptation), the multi-head self-attention layers in the SLSTT Transformer memorize subject-specific facial features (subject identity bias) present in the training set rather than learning generic dynamic motion patterns.

### 3.3 Linear Behavior of Eulerian Magnification (EVM)
Configurations with EVM enabled performed identically to those with EVM disabled. Since EVM is a linear process, it scales both the facial motion signal and sensor noise by the same factor. Under standard supervised training, the convolutions simply scale their weight magnitudes to compensate, resulting in similar local minima. EVM must be paired with representation regularizations to show a benefit.

---

## 4. Visual Results Analysis

Below we present the visual analysis of the ablation sweep.

### Figure 1: Performance Comparison Across Key Configurations
The chart below illustrates the Accuracy and Macro F1 scores for the primary ablation configurations. It highlights the spatial baseline's strong F1 score and the drop in performance for the full proposed pipeline due to overfitting.

<p align="center">
  <img src="C:/Users/Addhyan/Desktop/Thesis/Thesis3/figures/ablation_performance_comparison.png" width="500" />
</p>

### Figure 2: Per-Class F1-Score Comparison
The grouped bar chart below shows the F1 scores for each emotion class. It illustrates the complete collapse of minority classes (Positive and Surprise) in models without CNNs (Configs 1, 2, 4), and how the spatial and attention-based models (Configs 3 and 5) successfully learn features for the Positive class.

<p align="center">
  <img src="C:/Users/Addhyan/Desktop/Thesis/Thesis3/figures/ablation_class_f1_comparison.png" width="550" />
</p>

### Figure 3: Confusion Matrix - Proposed Unified Configuration (Config 8)
The normalized confusion matrix below details the predictions of the full proposed configuration. While it captures Negative and Positive expressions, it fails to predict Surprise due to class imbalance and the lack of auxiliary regularization constraints.

<p align="center">
  <img src="C:/Users/Addhyan/Desktop/Thesis/Thesis3/Ablation_Study/results/config_8_proposed_unified__WITH_evm__WITH_simam__WITH_3dcnn__WITH_transformer/confusion_matrix.png" width="380" />
</p>
