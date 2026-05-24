"""
generate_documents.py
======================
Generates the Master's Thesis Presentation in three formats:
1. Markdown (.md)
2. MS Word (.docx)
3. PDF (.pdf) - styled as landscape A4 presentation slides.

Embeds the generated figures in the .docx and .pdf files.
"""
import sys
import os
import pathlib
import subprocess

# Ensure document libraries are installed
for pkg in ["python-docx", "markdown", "xhtml2pdf"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
import markdown
from xhtml2pdf import pisa

# Root directory
ROOT_DIR = pathlib.Path(__file__).parent.resolve()
FIGURES_DIR = ROOT_DIR / "figures"

# Define Slide Data
SLIDES = [
    {
        "id": 1,
        "title": "Spatiotemporal Adversarial Contrastive Network for Subject-Invariant MER",
        "subtitle": "Master's Thesis Presentation | Academic Year 2026",
        "author": "Addhyan",
        "layout": "title",
        "content": [
            "Objective: Address the twin challenges of Micro-Expression Recognition (MER): Data Starvation and Subject Identity Bias.",
            "Methodology: Eulerian Video Magnification (EVM) for motion amplification + Modality-Aware 3D-CNN Backbone + Sequence-Level Spatio-Temporal Transformer (SLSTT).",
            "Optimization: Supervised Contrastive Learning (SupCon) with Cross-Batch Memory (XBM) + Domain-Adversarial Subject-Classifier with Gradient Reversal Layer (GRL).",
            "Transfer Scheme: Two-Phase Macro-to-Micro Transfer Learning strategy."
        ],
        "image": None,
        "speech": (
            "Good morning/afternoon, distinguished members of the committee. Welcome to my master's thesis presentation. "
            "My research is focused on developing a 'Subject-Invariant Spatiotemporal Contrastive Network' for Micro-Expression Recognition. "
            "Micro-Expression Recognition, or MER, is a critical task in computer vision with applications in clinical psychology, lie detection, and security. "
            "However, it is plagued by two fundamental challenges: first, 'data starvation', because micro-expressions are rare and difficult to record, resulting in very small datasets. "
            "Second, 'subject identity bias', as deep models naturally lock onto static facial structures (identities) rather than transient muscle movements. "
            "In this project, we design an integrated pipeline combining Eulerian Video Magnification, a hybrid 3D-CNN/Transformer network, "
            "supervised contrastive learning with cross-batch memory, adversarial domain adaptation, and a two-phase transfer learning framework. "
            "Today, I will explain our design choices, implementation details, and the results we achieved along the way."
        )
    },
    {
        "id": 2,
        "title": "The Core Challenge: Micro-Expressions vs. Identity Bias",
        "layout": "standard",
        "content": [
            "What are Micro-Expressions? Brief, involuntary facial movements revealing suppressed emotions. Typically last less than 500ms with extremely low muscle displacement.",
            "Identity Bias Bottleneck: Deep models struggle to generalize to unseen test subjects because static facial geometry (jawline, eyes, nose shape) presents a stronger learning signal than subtle, brief temporal displacements.",
            "Data Starvation Bottleneck: Annotation requires expensive frame-by-frame analysis by certified facial action coders. Benchmark datasets are tiny (e.g., CASME II has only 255 micro-expression samples from 26 subjects).",
            "Goal: Design a system that is both spatiotemporally sensitive (to capture sub-second motions) and subject-invariant (to generalize to unseen individuals)."
        ],
        "image": None,
        "speech": (
            "To understand our design decisions, we must analyze the physics of micro-expressions. Unlike macro-expressions which are voluntary and prolonged, "
            "micro-expressions are rapid, involuntary, and localized to specific face zones. Because they last less than half a second, the displacement is minute. "
            "When we train deep neural networks on small datasets, they easily memorize the static face geometry of the few available subjects. "
            "Consequently, if we evaluate the model on a new subject, it performs poorly because it classifies based on facial structure rather than facial motion. "
            "This identity bias, coupled with the severe lack of annotated samples, means we cannot train massive architectures like ViTs or heavy 3D-CNNs from scratch. "
            "We need a model that is parameter-efficient, captures brief temporal dynamics, and actively strips out identity cues. This is the goal of our work."
        )
    },
    {
        "id": 3,
        "title": "Stage 1: Preprocessing & Dataset Unification",
        "layout": "standard",
        "content": [
            "Source Databases: CASME II (high-speed camera at 200 FPS, cropped faces) and CAS(ME)² (standard camera at 30 FPS, full frames).",
            "Metadata Unification: Standardized disparate Excel labels, frame annotations, and naming schemas into a single unified CSV file ('master_thesis_labels.csv') with 612 samples (300 macro, 312 micro).",
            "Unified Emotion Taxonomy: Mapped diverse raw labels into four primary classes to ensure training stability: Negative (disgust, sadness, fear, anger, contempt, repression), Positive (happiness), Surprise, and Others.",
            "Data Distribution: Severe class imbalance (Negative: 258, Others: 114, Positive: 179, Surprise: 61 in unified metadata, representing 598 active samples on disk after file matching)."
        ],
        "image": "eda_micro_distribution.png",
        "speech": (
            "Our first stage was to unify the datasets. We used two benchmark repositories: CASME II, recorded with a high-speed camera at 200 frames per second, "
            "and CAS(ME)², captured under standard conditions at 30 frames per second. We developed a metadata unification module to standardize onset, apex, "
            "and offset frames and normalize subject and folder IDs. In addition, the original datasets contained many tiny emotion classes with only a few samples. "
            "To prevent training instability, we collapsed them into a unified four-class taxonomy: Positive, Negative, Surprise, and Others. "
            "As shown in the class distribution chart on the right, the unified dataset exhibits severe class imbalance, with Negative emotions dominating. "
            "This required us to implement class-weighting and focal losses during training. After matching the CSV rows with actual folders on disk, "
            "we obtained 598 matched samples."
        )
    },
    {
        "id": 4,
        "title": "Stage 1: Motion Amplification — Eulerian Video Magnification",
        "layout": "standard",
        "content": [
            "Why EVM? Micro-expressions involve displacements that are sub-pixel, often below the camera's sensor noise floor.",
            "Spatial Decomposition: Decompose each video frame into spatial frequency bands using a 4-level Laplacian pyramid.",
            "Temporal Bandpass Filtering: Apply Fast Fourier Transform (FFT) on each spatial level over time. Cutoff frequencies set to [0.4, 3.0] Hz to isolate facial muscle movement frequencies and exclude camera noise/head sway.",
            "Amplification & Reconstruction: Multiply the bandpass signal by an amplification factor (alpha = 20.0), add it back to the original frames, and collapse the Laplacian pyramid to reconstruct the magnified video clip."
        ],
        "image": None,
        "speech": (
            "To make sub-threshold facial displacements visible, we incorporated Eulerian Video Magnification (EVM). "
            "EVM works by treating facial movement as a temporal signal and amplifying it in the frequency domain. "
            "For every video, we build a 4-level Laplacian pyramid to decompose the frames spatially. "
            "Then, we apply an FFT-based temporal bandpass filter to the pixel values over time. "
            "We set the bandpass cutoffs to 0.4 Hz and 3.0 Hz, which correspond to the typical speed of facial muscle contractions during a micro-expression. "
            "This filters out high-frequency sensor noise and very low-frequency head motion. "
            "The filtered signal is multiplied by an amplification factor of 20, and the video is reconstructed. "
            "This amplifies the micro-movements, providing a much stronger signal-to-noise ratio for downstream neural network learning."
        )
    },
    {
        "id": 5,
        "title": "Stage 1: Modality Extraction & Temporal Interpolation",
        "layout": "standard",
        "content": [
            "Modality-Aware Representation: Instead of raw RGB, we extract motion-specific representations. Input is represented as a 3-channel tensor:",
            "  - Channel 0: Horizontal optical flow (u) - captures horizontal facial muscle shifts.",
            "  - Channel 1: Vertical optical flow (v) - captures vertical facial muscle shifts (e.g. brow raise).",
            "  - Channel 2: Optical strain magnitude - captures local skin compression and stretching.",
            "Temporal Standardization: Original clips have highly variable frame counts (10 to 150 frames). A TemporalInterpolator with dynamic index sampling standardizes all sequences to exactly T = 32 frames.",
            "Output Shape: Standardized 3-channel tensors of shape [B, 3, 32, 224, 224] saved as float32 NumPy arrays (.npy) on disk."
        ],
        "image": None,
        "speech": (
            "After magnifying the motion, we extract three distinct motion-related channels rather than using raw RGB frames. "
            "We compute the dense optical flow between frames. Channel 0 contains horizontal flow, capturing lateral muscle shifts. "
            "Channel 1 contains vertical flow, capturing vertical movements such as the raising of the eyebrows or dropping of the jaw. "
            "Channel 2 holds the optical strain magnitude, which reflects local skin compression and stretching. "
            "Furthermore, because the original clips range from 10 to 150 frames, we implement a temporal interpolator. "
            "It uses linear interpolation and index sampling to standardize every video to exactly 32 frames. "
            "The final output is a 3D volume of shape 3 by 32 by 224 by 224, stored directly as numpy arrays on disk to accelerate training."
        )
    },
    {
        "id": 6,
        "title": "Stage 2: Hybrid Neural Architecture — 3D-CNN Backbone",
        "layout": "standard",
        "content": [
            "3D-CNN Backbone: A modality-aware, three-stream 3D-CNN inspired by STSTNet (Shallow Triple Stream 3D CNN).",
            "Unshared Weight Streams: Each input channel (flow-u, flow-v, strain) is processed by an independent, unshared shallow 3D-CNN. This enables kernels to specialize in the distinct physics of each motion modality.",
            "Stream Architecture: Conv3D (1->16, kernel=1x3x3) -> BN -> ReLU -> Dropout -> Conv3D (16->32, kernel=1x3x3) -> BN -> ReLU -> Dropout -> MaxPool3D (kernel=1x2x2).",
            "SimAM 3D Attention: Parameter-free attention module based on inter-neuron suppression. Computes a neuron-level energy function without adding any learnable parameters, focusing the network on active facial regions.",
            "Output Fusion: Modality feature maps are concatenated along the channel dimension to form a [B, 96, 32, 112, 112] tensor."
        ],
        "image": None,
        "speech": (
            "For our neural network architecture in Stage 2, we designed a hybrid model. "
            "The first component is a three-branch 3D CNN backbone inspired by the STSTNet framework. "
            "Rather than sharing weights, each stream has its own independent parameters. "
            "This allows the horizontal flow, vertical flow, and skin strain to be processed by specialized filters. "
            "Each branch contains two 3D convolutions with a kernel size of 1 along the temporal axis. "
            "This design choice ensures we halve the spatial dimensions to 112 while leaving the temporal depth at 32 completely untouched. "
            "After the CNN layers, we apply 3D SimAM attention to each branch. SimAM is a parameter-free attention module that "
            "calculates neuron-level weights based on a spatial energy function. It adds zero parameters to our model, avoiding overfitting, "
            "yet effectively highlights the active zones of the face. Finally, we concatenate the three branches to form a 96-channel feature map."
        )
    },
    {
        "id": 7,
        "title": "Stage 2: Temporal Modeling — SLSTT Transformer Encoder",
        "layout": "standard",
        "content": [
            "Spatio-Temporal Transition: Adaptive spatial average pooling compresses the fused CNN tensor from [B, 96, 32, 112, 112] to [B, 96, 32, 1, 1], which is reshaped into a sequence of 32 tokens: [B, 32, 96].",
            "SLSTT (Sequence-Level Spatio-Temporal Transformer): Processes the 32 temporal feature tokens as a sequence.",
            "Sinusoidal Positional Encoding: Injects fixed, deterministic absolute position information (Vaswani et al., 2017) to teach the transformer the chronological progression of the expression.",
            "Transformer Encoder: 2 layers, 4 attention heads, 256 feedforward dimension, and dropout of 0.1. Uses pre-norm (layer norm before attention) for training stability.",
            "Temporal Mean Pooling: Compresses the 32 sequence tokens into a single 96-dimensional global feature representation."
        ],
        "image": None,
        "speech": (
            "Once local spatial features are extracted, we model their temporal progression across the 32 frames. "
            "We apply adaptive spatial average pooling to compress the spatial grid, leaving us with a sequence of 32 temporal tokens, each with a 96-dimensional vector. "
            "This sequence is fed into the Sequence-Level Spatio-Temporal Transformer, or SLSTT. "
            "We first inject sinusoidal positional encoding. Since self-attention is permutation-invariant, positional encoding is "
            "necessary to teach the transformer the temporal order of the onset, apex, and offset phases. "
            "The sequence is processed by a 2-layer Transformer Encoder with 4 attention heads. The multi-head self-attention mechanism "
            "captures long-range temporal correlations, allowing the model to look at the relationship between frames. "
            "Finally, we mean-pool the sequence over time to obtain a single 96-dimensional global feature vector representing the entire clip."
        )
    },
    {
        "id": 8,
        "title": "Stage 3: Subject-Invariant Adversarial Domain Adaptation",
        "layout": "standard",
        "content": [
            "Triple-Head Architecture: The 96-d feature vector is routed to three parallel task heads:",
            "  1. Emotion Head: LN -> Dropout -> Linear -> 4 emotion logits (Focal Loss).",
            "  2. Projection Head: MLP -> L2Norm -> 64-d contrastive features (SupCon Loss).",
            "  3. Identity Head: GRL -> LN -> Linear -> N subject logits (Cross-Entropy).",
            "Gradient Reversal Layer (GRL): Inserted before the identity head. In the forward pass, it acts as an identity mapping. In the backward pass, it multiplies the incoming gradients by a negative weight (-lambda).",
            "GRL Sigmoid Annealing Schedule: Gradually increases lambda from 0.0 to 1.0 using: lambda(p) = 2 / (1 + exp(-10*p)) - 1. Prevents destabilization in early epochs."
        ],
        "image": None,
        "speech": (
            "To remove subject identity bias and make the network generalize to unseen individuals, Stage 3 introduces "
            "Adversarial Domain Adaptation. We design a triple-head network. The global feature vector is sent to three heads: "
            "the emotion classification head, a projection head for contrastive learning, and an identity classification head. "
            "Crucially, we insert a Gradient Reversal Layer, or GRL, before the identity head. "
            "During forward propagation, the GRL passes features unmodified to the identity head so it can try to predict who the subject is. "
            "During backpropagation, however, the GRL multiplies the gradients by negative lambda. "
            "This forces the backbone to adjust its weights in a direction that makes it *impossible* to identify the subject. "
            "We use a sigmoid annealing schedule for lambda, starting at zero and scaling to 1.0, so the network can learn basic "
            "motion features before being subjected to the adversarial gradient."
        )
    },
    {
        "id": 9,
        "title": "Stage 3: Supervised Contrastive Learning & Losses",
        "layout": "standard",
        "content": [
            "Supervised Contrastive Loss (SupCon): Pulls same-emotion representations together while pushing different emotions apart in a normalized 64-d projection space. Promotes tight, class-specific clusters.",
            "Cross-Batch Memory (XBM): Resolves the VRAM-constrained batch size limit (batch = 2). A queue of size 64 stores historical embeddings from previous batches, providing a large pool of positive/negative pairs for stable contrastive learning.",
            "Focal Loss (Emotion Classification): Handles severe class imbalance. Uses gamma = 2.0, label smoothing of 0.05, and per-class alpha weights: Negative (0.591), Positive (0.879), Surprise (2.451), Others (1.311).",
            "Total Objective: L_total = (1.0 x L_SupCon) + L_Emotion(Focal) + (alpha x L_Identity)"
        ],
        "image": None,
        "speech": (
            "To structure a highly discriminative feature space, we incorporate Supervised Contrastive Learning, or SupCon. "
            "The projection head projects features into a 64-dimensional, L2-normalized space. "
            "The SupCon loss forces features belonging to the same emotion category to cluster tightly, while pushing different emotions apart. "
            "Due to VRAM limits on our laptop GPU, we had to use a small batch size of 2, which is normally too small for contrastive learning. "
            "To solve this, we integrated a Cross-Batch Memory queue of size 64. The queue stores embeddings from previous batches, "
            "enabling rich contrastive comparisons. The overall optimization is a multi-objective loss combining SupCon, "
            "class-weighted Focal Loss to counter dataset imbalance, and the adversarial identity loss."
        )
    },
    {
        "id": 10,
        "title": "Stage 3 Results: Leave-One-Subject-Out Validation",
        "layout": "standard",
        "content": [
            "Validation Protocol: 26-fold Leave-One-Subject-Out (LOSO) cross-validation (standard for MER, ensuring strict subject-disjoint splits).",
            "Evaluation Metrics: Calculated Grand Mean Accuracy and Macro F1-Score across all 26 subject folds.",
            "Quantitative Results: Grand Mean Accuracy = 58.11% | Macro F1-Score = 33.31%.",
            "Analysis: An accuracy of 58.11% on a 4-class problem (chance is 25%) is highly competitive. The lower Macro F1-score (33.31%) reflects the severe difficulty in predicting under-represented classes (Surprise and Positive) under strict subject-disjoint validation."
        ],
        "image": "stage3_vs_stage4_comparison.png",
        "speech": (
            "We evaluated Stage 3 using a 26-fold Leave-One-Subject-Out validation protocol. In each fold, "
            "we trained on 25 subjects and tested on the remaining unseen subject. This is the most rigorous testing protocol in MER. "
            "The model achieved a Grand Mean Accuracy of 58.11%, which is far above the 25% random baseline, demonstrating that it successfully "
            "learned facial movement patterns. The macro F1-score was 33.31%. As you can see in the bar chart on the right, "
            "this reflects the difficulty of classifying under-represented classes on unseen subjects. "
            "Since the dataset is small and heavily skewed, the model struggle to generalize on rare classes. "
            "This limitation led us to develop Stage 4: the Macro-to-Micro transfer learning strategy."
        )
    },
    {
        "id": 11,
        "title": "Stage 4: Two-Phase Macro-to-Micro Transfer Learning",
        "layout": "standard",
        "content": [
            "Motivation: ~300 micro-expression clips is insufficient to train a spatiotemporal transformer from scratch. CAS(ME)² macro-expressions (~300 clips) capture similar facial action unit physics with larger amplitude.",
            "Phase 1: Macro Pre-Training (Duration: 141.2 min):",
            "  - Dataset: 293 macro-expression clips (20 subjects).",
            "  - Adversary: OFF (lambda_id = 0.0) to learn facial dynamics without constraints.",
            "  - Result: Best Validation Accuracy = 70.37%.",
            "Phase 2: Micro Fine-Tuning (Duration: 358.4 min):",
            "  - Dataset: 305 micro-expression clips (26 subjects).",
            "  - Transferred 101/103 parameter tensors; re-initialized identity head (20 -> 26 subjects).",
            "  - Adversary: ON (lambda_id = 0.40) to enforce subject-invariance.",
            "  - Result: Best Validation Accuracy = 37.78% (Subject-Disjoint Split)."
        ],
        "image": "training_history_curves.png",
        "speech": (
            "To combat data starvation, we proposed a two-phase transfer learning framework. "
            "We pre-trained the model on macro-expressions from CAS(ME)². Macro-expressions share the same facial action units "
            "but feature larger muscle displacements. During Phase 1, we disabled the identity adversary, allowing the backbone "
            "to learn facial dynamics freely. This pre-training took 141 minutes and achieved a validation accuracy of 70.37%. "
            "For Phase 2, we transferred the pre-trained weights to a micro-expression model. "
            "We re-initialized the identity head because the number of subjects changed from 20 to 26, and fine-tuned on the micro-expression data "
            "with the adversary enabled at a weight of 0.4. This phase took approximately 6 hours, achieving a validation accuracy of 37.78%. "
            "The training curves on the right show that while the SupCon loss stabilized quickly around 4.17, "
            "the validation accuracy peaked at 37.78%, highlighting the difficulty of transferring features to low-amplitude micro-movements."
        )
    },
    {
        "id": 12,
        "title": "Stage 4 Results: Embedding Space & Feature Invariance",
        "layout": "standard",
        "content": [
            "Validation Accuracy: 37.78% on a subject-disjoint validation set.",
            "Confusion Matrix Analysis: Excellent recall on Negative (56%) and Others (44%). Low recall on Positive and Surprise due to extreme class imbalance.",
            "t-SNE Embeddings Visual Verification (64-d projection space):",
            "  - Plot A (Emotion): Shows distinct clusters for emotion classes, proving that Supervised Contrastive Loss successfully groups emotional content.",
            "  - Plot B (Subject): Shows complete, uniform mixing of subject IDs, proving that the Gradient Reversal Layer successfully stripped out identity bias.",
            "Conclusion: The network achieved high subject-invariance, but classification performance remains bottlenecked by class imbalance."
        ],
        "image": "tsne_embeddings.png",
        "speech": (
            "Let's look at the visual evidence of our results. The t-SNE plot of the projection head's feature space "
            "tells a powerful story. In Plot A, where points are colored by emotion class, we can see distinct, tight clusters forming. "
            "This proves that our Supervised Contrastive Loss successfully organized the embedding space based on emotional content, "
            "making features highly discriminative. In Plot B, where the same points are colored by subject ID, the subjects are completely mixed. "
            "If the network had retained subject bias, we would see clustering by subject ID. This complete mixing is empirical "
            "proof that our Gradient Reversal Layer successfully stripped out identity cues. "
            "While our validation accuracy of 37.78% on this split is constrained by class imbalance, we have succeeded in creating "
            "a subject-invariant feature space."
        )
    },
    {
        "id": 13,
        "title": "Stage 4 Results: Confusion Matrix",
        "layout": "standard",
        "content": [
            "Confusion Matrix Detail: Row-normalized confusion matrix on the subject-disjoint validation split.",
            "Recall breakdown per unified class:",
            "  - Negative: 56.25% (highest recall, represents the dominant class)",
            "  - Others: 44.44% (moderate recall, second largest class)",
            "  - Positive: 0.00% (failed to predict, due to only 47 training samples)",
            "  - Surprise: 0.00% (failed to predict, due to only 35 training samples)",
            "Implication: Class imbalance remains a major hurdle. The model tends to predict majority classes (Negative and Others), while completely missing rare classes (Positive and Surprise) in the validation set."
        ],
        "image": "confusion_matrix.png",
        "speech": (
            "This slide shows the detailed row-normalized confusion matrix for our Stage 4 validation set. "
            "As we can see, the model achieved a recall of 56% on Negative expressions and 44% on the Others class. "
            "However, it failed to predict any Positive or Surprise samples. This is a direct consequence of the extreme class imbalance. "
            "With only 47 Positive and 35 Surprise samples in the entire micro-expression dataset, and after splitting for validation, "
            "the training set had very few examples for these categories. During fine-tuning, the model over-indexed on the majority classes "
            "to minimize loss. This highlights that while transfer learning and domain adaptation solve the subject-invariance problem, "
            "they cannot fully overcome the lack of training data for rare classes. This must be addressed in future work using specialized "
            "data augmentation."
        )
    },
    {
        "id": 14,
        "title": "Technical Critique & Future Directions",
        "layout": "standard",
        "content": [
            "Strengths:",
            "  - High Subject-Invariance: GRL effectively stripped identity details from the embedding space (verified by t-SNE).",
            "  - Extreme Parameter Efficiency: Total model size is only 382K parameters, making it highly suitable for small datasets and deployment on low-power devices.",
            "Weaknesses & Bottlenecks:",
            "  - Class Imbalance: Classification macro F1 remains low due to data scarcity in Positive and Surprise classes.",
            "  - Framerate Mismatch: CASME II (200 FPS) and CAS(ME)² (30 FPS) have different movement speeds, causing temporal mismatches during weight transfer.",
            "Future Work Recommendations:",
            "  1. Dynamic Time Warping (DTW): Normalize temporal framerates before sequence modeling.",
            "  2. Generative Augmentation: Use Diffusion models or GANs to generate synthetic Positive and Surprise micro-expressions.",
            "  3. Large-Scale Macro Pre-Training: Pre-train on much larger datasets (e.g. CK+, Oulu-CASIA) to capture richer facial dynamics."
        ],
        "image": None,
        "speech": (
            "To conclude my presentation, let us look at a technical critique of our work and recommendations for the future. "
            "The model has major strengths: it achieves high subject-invariance and is extremely parameter-efficient, containing only "
            "382,000 parameters. This prevents overfitting and makes it ideal for real-time edge applications. "
            "However, the performance is limited by two factors: class imbalance and the framerate mismatch between our pre-training and fine-tuning datasets. "
            "To take this research further, I propose three improvements: first, implementing Dynamic Time Warping to align the framerates of the two datasets. "
            "Second, using generative models like Diffusion Models or GANs to synthesize rare expressions, particularly Positive and Surprise. "
            "Third, expanding the pre-training phase to much larger macro-expression datasets like CK+ or Oulu-CASIA. "
            "This will give the transformer a richer foundation of facial muscle physics. Thank you for your time, and I am now ready to take your questions."
        )
    }
]

# Academic Reference Table Data
PAPERS_TABLE = [
    ("Eulerian Video Magnification (Wu et al., TOG 2012)", 
     "Motion amplification to reveal sub-threshold, imperceptible facial muscle contractions.",
     "Spatial Laplacian decomposition, FFT-based temporal bandpass filtering, and motion reconstruction."),
    
    ("STSTNet (Liong et al., FG 2019)", 
     "Shallow, parameter-efficient spatiotemporal backbone designed to avoid overfitting on small MER datasets.",
     "Three-branch 3D-CNN architecture processing horizontal flow, vertical flow, and skin strain independently (unshared weights)."),
    
    ("SimAM (Yang et al., ICML 2021)", 
     "Lightweight, parameter-free attention mechanism to focus features on active facial zones without adding model complexity.",
     "3D spatial-temporal neuron energy function (inter-neuron suppression theory) applied element-wise."),
    
    ("SLSTT (Zhang et al., IEEE TAFFC 2022)", 
     "Sequence-level temporal modeling to capture long-range relations across frames.",
     "Sinusoidal positional encoding, Multi-head self-attention Transformer Encoder layers, and temporal mean pooling."),
    
    ("Domain-Adversarial Training (Ganin et al., JMLR 2016)", 
     "Force the network to learn subject-invariant (domain-invariant) features, preventing identity memorization.",
     "Gradient Reversal Layer (GRL) inserted before the identity classification head with a sigmoid annealing lambda schedule."),
    
    ("Supervised Contrastive Learning (Khosla et al., NeurIPS 2020)", 
     "Establish a highly discriminative feature space by pulling same-emotion embeddings together and pushing different emotions apart.",
     "64-d projection head optimization using SupCon loss in L2-normalized space."),
    
    ("Cross-Batch Memory (XBM) (Wang et al., CVPR 2020)", 
     "Circumvent VRAM-constrained batch size limits (batch = 2) for effective contrastive learning.",
     "Memory queue of size 64 to store historical embeddings, enabling negative/positive mining across batches."),
    
    ("Focal Loss (Lin et al., ICCV 2017)", 
     "Handle severe class imbalance and prevent majority classes from dominating the gradients.",
     "Focal loss with class-weight scaling (Negative, Positive, Surprise, Others) and label smoothing.")
]

# ─────────────────────────────────────────────────────────────────────
# 1. GENERATE MARKDOWN FILE
# ─────────────────────────────────────────────────────────────────────
def generate_markdown():
    md_path = ROOT_DIR / "Thesis_Presentation.md"
    print(f"Generating Markdown: {md_path}")
    
    lines = []
    lines.append("# Subject-Invariant Spatiotemporal Contrastive Network for Micro-Expression Recognition")
    lines.append("**Master's Thesis Presentation Portfolio**  ")
    lines.append("**Author:** Addhyan  ")
    lines.append("**Date:** May 2026  \n")
    lines.append("---")
    
    for slide in SLIDES:
        lines.append(f"\n## Slide {slide['id']}: {slide['title']}")
        if "subtitle" in slide:
            lines.append(f"*{slide['subtitle']}*  ")
            lines.append(f"**Presenter:** {slide['author']}  \n")
            
        lines.append("\n### Slide Content")
        for bullet in slide["content"]:
            lines.append(f"- {bullet}")
            
        if slide["image"]:
            img_path = FIGURES_DIR / slide["image"]
            # Convert backslashes for markdown compatibility on windows
            img_uri = img_path.as_uri()
            lines.append(f"\n### Visual Result")
            lines.append(f"![{slide['title']}]({img_uri})")
            
        lines.append("\n### Speech Notes")
        lines.append(f"> *{slide['speech']}*")
        lines.append("\n---")
        
    # Reference papers slide
    lines.append("\n## Slide 15: Reference Papers & Implemented Features")
    lines.append("### Academic References")
    lines.append("| Research Paper / Source | Reason for Reference | Key Features Implemented |")
    lines.append("|---|---|---|")
    for paper, reason, feature in PAPERS_TABLE:
        lines.append(f"| **{paper}** | {reason} | {feature} |")
        
    lines.append("\n### Speech Notes")
    lines.append(
        "> *This final slide lists the core academic literature that served as the foundation of my thesis. "
        "Each method—from Eulerian Video Magnification to the Gradient Reversal Layer, SimAM attention, "
        "and Cross-Batch Memory—was systematically chosen, adapted, and integrated to solve the specific "
        "challenges of Micro-Expression Recognition. I would like to thank the authors of these works "
        "for sharing their codebases and methodologies, which made this research possible.*"
    )
    
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print("Markdown presentation generated successfully.")

# ─────────────────────────────────────────────────────────────────────
# 2. GENERATE MS WORD FILE (.docx)
# ─────────────────────────────────────────────────────────────────────
def generate_docx():
    docx_path = ROOT_DIR / "Thesis_Presentation.docx"
    print(f"Generating MS Word: {docx_path}")
    
    doc = docx.Document()
    
    # Configure margins (1 inch)
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # Styles helper
    def set_font(run, name="Segoe UI", size=Pt(11), bold=False, italic=False, color=None):
        run.font.name = name
        run.font.size = size
        run.font.bold = bold
        run.font.italic = italic
        if color:
            run.font.color.rgb = color

    # Style colors
    NAVY = RGBColor(26, 54, 93)      # #1a365d (titles)
    GRAY = RGBColor(74, 85, 104)     # #4a5568 (body)
    LIGHT_BLUE = "EDF2F7"            # Background for notes

    for slide in SLIDES:
        # Title
        p_title = doc.add_paragraph()
        run_title = p_title.add_run(f"Slide {slide['id']}: {slide['title']}")
        set_font(run_title, name="Segoe UI", size=Pt(18), bold=True, color=NAVY)
        p_title.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p_title.paragraph_format.space_after = Pt(12)
        
        if "subtitle" in slide:
            p_sub = doc.add_paragraph()
            run_sub = p_sub.add_run(f"{slide['subtitle']}\nPresenter: {slide['author']}")
            set_font(run_sub, name="Segoe UI", size=Pt(12), italic=True, color=GRAY)
            p_sub.paragraph_format.space_after = Pt(20)

        # Slide content heading
        p_ch = doc.add_paragraph()
        run_ch = p_ch.add_run("Slide Content:")
        set_font(run_ch, name="Segoe UI", size=Pt(13), bold=True, color=NAVY)
        p_ch.paragraph_format.space_after = Pt(6)

        # Bullet points
        for bullet in slide["content"]:
            p_bullet = doc.add_paragraph(style='List Bullet')
            run_bullet = p_bullet.add_run(bullet)
            set_font(run_bullet, name="Segoe UI", size=Pt(11), color=GRAY)
            p_bullet.paragraph_format.space_after = Pt(4)

        # Image if exists
        if slide["image"]:
            img_path = FIGURES_DIR / slide["image"]
            if img_path.exists():
                doc.add_paragraph().paragraph_format.space_before = Pt(10)
                p_img = doc.add_paragraph()
                p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r_img = p_img.add_run()
                r_img.add_picture(str(img_path), width=Inches(4.5))
                p_img_lbl = doc.add_paragraph()
                p_img_lbl.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r_img_lbl = p_img_lbl.add_run(f"Figure: {slide['title']}")
                set_font(r_img_lbl, name="Segoe UI", size=Pt(9.5), italic=True, color=GRAY)
                p_img_lbl.paragraph_format.space_after = Pt(10)

        # Speech notes
        p_nh = doc.add_paragraph()
        p_nh.paragraph_format.space_before = Pt(12)
        run_nh = p_nh.add_run("Speech Notes:")
        set_font(run_nh, name="Segoe UI", size=Pt(12), bold=True, color=NAVY)
        p_nh.paragraph_format.space_after = Pt(4)

        # Notes block (styled in a shaded single-cell table)
        table = doc.add_table(rows=1, cols=1)
        table.autofit = False
        table.columns[0].width = Inches(6.5)
        
        # Shade table cell light gray
        cell = table.cell(0, 0)
        shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{LIGHT_BLUE}"/>')
        cell._tc.get_or_add_tcPr().append(shading)
        
        p_notes = cell.paragraphs[0]
        p_notes.paragraph_format.left_indent = Inches(0.1)
        p_notes.paragraph_format.right_indent = Inches(0.1)
        run_notes = p_notes.add_run(slide["speech"])
        set_font(run_notes, name="Segoe UI", size=Pt(10), italic=True, color=GRAY)

        # Page Break for next slide
        doc.add_page_break()

    # References Slide
    p_title = doc.add_paragraph()
    run_title = p_title.add_run("Slide 15: Reference Papers & Implemented Features")
    set_font(run_title, name="Segoe UI", size=Pt(18), bold=True, color=NAVY)
    p_title.paragraph_format.space_after = Pt(12)

    # Add reference table
    table_ref = doc.add_table(rows=len(PAPERS_TABLE) + 1, cols=3)
    table_ref.style = 'Table Grid'
    
    # Headers
    hdr_cells = table_ref.rows[0].cells
    hdr_cells[0].text = "Research Paper / Source"
    hdr_cells[1].text = "Reason for Reference"
    hdr_cells[2].text = "Key Features Implemented"
    
    for cell in hdr_cells:
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                set_font(r, name="Segoe UI", size=Pt(10), bold=True, color=RGBColor(255, 255, 255))
        # Color header background navy
        shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="1A365D"/>')
        cell._tc.get_or_add_tcPr().append(shading)

    # Fill data
    for i, (paper, reason, feature) in enumerate(PAPERS_TABLE):
        row_cells = table_ref.rows[i+1].cells
        row_cells[0].text = paper
        row_cells[1].text = reason
        row_cells[2].text = feature
        
        # Format text and add zebra shading
        bg_color = "F7FAFC" if i % 2 == 0 else "FFFFFF"
        for col_idx, cell in enumerate(row_cells):
            shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{bg_color}"/>')
            cell._tc.get_or_add_tcPr().append(shading)
            for p in cell.paragraphs:
                for r in p.runs:
                    set_font(r, name="Segoe UI", size=Pt(9), color=GRAY)
                    if col_idx == 0:
                        r.font.bold = True

    # Speech notes for reference slide
    p_nh = doc.add_paragraph()
    p_nh.paragraph_format.space_before = Pt(16)
    run_nh = p_nh.add_run("Speech Notes:")
    set_font(run_nh, name="Segoe UI", size=Pt(12), bold=True, color=NAVY)
    p_nh.paragraph_format.space_after = Pt(4)

    table = doc.add_table(rows=1, cols=1)
    table.autofit = False
    table.columns[0].width = Inches(6.5)
    cell = table.cell(0, 0)
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{LIGHT_BLUE}"/>')
    cell._tc.get_or_add_tcPr().append(shading)
    p_notes = cell.paragraphs[0]
    p_notes.paragraph_format.left_indent = Inches(0.1)
    p_notes.paragraph_format.right_indent = Inches(0.1)
    run_notes = p_notes.add_run(
        "This final slide lists the core academic literature that served as the foundation of my thesis. "
        "Each method—from Eulerian Video Magnification to the Gradient Reversal Layer, SimAM attention, "
        "and Cross-Batch Memory—was systematically chosen, adapted, and integrated to solve the specific "
        "challenges of Micro-Expression Recognition. I would like to thank the authors of these works "
        "for sharing their codebases and methodologies, which made this research possible."
    )
    set_font(run_notes, name="Segoe UI", size=Pt(10), italic=True, color=GRAY)

    doc.save(docx_path)
    print("docx presentation generated successfully.")

# ─────────────────────────────────────────────────────────────────────
# 3. GENERATE PDF FILE (.pdf)
# ─────────────────────────────────────────────────────────────────────
def generate_pdf():
    pdf_path = ROOT_DIR / "Thesis_Presentation.pdf"
    print(f"Generating Landscape PDF: {pdf_path}")
    
    # Custom HTML for Slide Layout
    html_slides = []
    
    # Title Slide HTML
    title_slide = SLIDES[0]
    title_html = f"""
    <div class="slide title-slide">
        <h1 style="font-size: 30px; margin-top: 50px;">{title_slide['title']}</h1>
        <h2 style="font-size: 18px; margin-top: 15px; color: #bee3f8;">{title_slide['subtitle']}</h2>
        <p style="font-size: 14px; margin-top: 40px; color: #e2e8f0;">Presenter: <strong>{title_slide['author']}</strong></p>
        <p style="font-size: 11px; margin-top: 80px; color: #a0aec0;">Thesis Advisor Evaluators Panel</p>
    </div>
    """
    html_slides.append(title_html)
    
    # Standard Slides HTML
    for slide in SLIDES[1:]:
        img_tag = ""
        
        # If there's an image, split content side-by-side using tables
        if slide["image"]:
            img_file = FIGURES_DIR / slide["image"]
            if img_file.exists():
                img_tag = f'<td class="slide-right"><img src="{img_file.resolve()}" width="330" style="margin-top: 15px; border: 1px solid #cbd5e0; border-radius: 4px;" /></td>'

        bullets = "".join([f"<li>{bullet}</li>" for bullet in slide["content"]])
        
        slide_body = f"""
        <div class="slide">
            <h1>Slide {slide['id']}: {slide['title']}</h1>
            <table width="100%" cellpadding="0" cellspacing="0" border="0">
                <tr>
                    <td class="slide-left">
                        <ul>
                            {bullets}
                        </ul>
                    </td>
                    {img_tag}
                </tr>
            </table>
            
            <div class="notes-box">
                <div class="notes-label">Speech Notes:</div>
                <p>{slide['speech']}</p>
            </div>
        </div>
        """
        html_slides.append(slide_body)
        
    # Reference Paper Table Slide HTML
    ref_rows = ""
    for paper, reason, feature in PAPERS_TABLE:
        ref_rows += f"""
        <tr>
            <td><strong>{paper}</strong></td>
            <td>{reason}</td>
            <td>{feature}</td>
        </tr>
        """
        
    ref_slide_html = f"""
    <div class="slide">
        <h1>Slide 15: Reference Papers & Implemented Features</h1>
        <table class="ref-table">
            <tr style="background-color: #1a365d; color: white;">
                <td style="font-weight: bold; color: white;">Research Paper / Source</td>
                <td style="font-weight: bold; color: white;">Reason for Reference</td>
                <td style="font-weight: bold; color: white;">Key Features Implemented</td>
            </tr>
            {ref_rows}
        </table>
        
        <div class="notes-box">
            <div class="notes-label">Speech Notes:</div>
            <p>This final slide lists the core academic literature that served as the foundation of my thesis.
            Each method—from Eulerian Video Magnification to the Gradient Reversal Layer, SimAM attention,
            and Cross-Batch Memory—was systematically chosen, adapted, and integrated to solve the specific
            challenges of Micro-Expression Recognition. I would like to thank the authors of these works
            for sharing their codebases and methodologies, which made this research possible.</p>
        </div>
    </div>
    """
    html_slides.append(ref_slide_html)
    
    # Full HTML wrapper with landscape CSS styling
    full_html = f"""<!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8"/>
        <style>
            @page {{
                size: A4 landscape;
                margin: 0.8cm;
            }}
            body {{
                font-family: 'Helvetica', 'Arial', sans-serif;
                color: #2d3748;
                font-size: 11px;
                line-height: 1.5;
            }}
            .slide {{
                page-break-after: always;
                background-color: #ffffff;
            }}
            .title-slide {{
                background-color: #1a365d;
                color: #ffffff;
                text-align: center;
            }}
            h1 {{
                font-size: 18px;
                color: #1a365d;
                border-bottom: 2px solid #2b6cb0;
                padding-bottom: 5px;
                margin-top: 0;
                margin-bottom: 12px;
                font-weight: bold;
            }}
            .title-slide h1 {{
                color: #ffffff;
                border-bottom: none;
            }}
            ul {{
                margin-top: 5px;
                margin-bottom: 5px;
                padding-left: 20px;
            }}
            li {{
                font-size: 10.5px;
                margin-bottom: 6px;
                line-height: 1.4;
            }}
            .slide-left {{
                vertical-align: top;
                padding-right: 15px;
            }}
            .slide-right {{
                vertical-align: top;
                text-align: center;
            }}
            .notes-box {{
                margin-top: 15px;
                background-color: #edf2f7;
                border-left: 3.5px solid #3182ce;
                padding: 6px 10px;
            }}
            .notes-box p {{
                margin: 0;
                font-size: 9px;
                color: #4a5568;
                font-style: italic;
                line-height: 1.35;
            }}
            .notes-label {{
                font-weight: bold;
                font-size: 9px;
                color: #2b6cb0;
                margin-bottom: 2px;
            }}
            .ref-table {{
                border-collapse: collapse;
                width: 100%;
                margin-top: 8px;
                margin-bottom: 8px;
            }}
            .ref-table th {{
                background-color: #1a365d;
                color: white;
                font-size: 9.5px;
                padding: 5px;
                text-align: left;
                font-weight: bold;
            }}
            .ref-table td {{
                border: 0.5px solid #cbd5e0;
                padding: 5px;
                font-size: 8.5px;
                line-height: 1.25;
            }}
            .ref-table tr:nth-child(even) {{
                background-color: #f7fafc;
            }}
        </style>
    </head>
    <body>
        {"".join(html_slides)}
    </body>
    </html>
    """
    
    # Render PDF using Pisa
    with open(pdf_path, "wb") as f:
        pisa_status = pisa.CreatePDF(full_html, dest=f)
        
    if pisa_status.err:
        print(f"Error occurred during PDF generation: {pisa_status.err}")
    else:
        print(f"Landscape PDF generated successfully -> {pdf_path}")

if __name__ == "__main__":
    generate_markdown()
    generate_docx()
    generate_pdf()
    print("\n[SUCCESS] Generated files in root workspace:")
    print("  1. Thesis_Presentation.md")
    print("  2. Thesis_Presentation.docx")
    print("  3. Thesis_Presentation.pdf")
