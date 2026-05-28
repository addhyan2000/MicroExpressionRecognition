# Summary: Improving Micro-Expression Recognition through Temporal and Motion Modeling

## 1. Thesis Introduction & Core Rationale

Automated **Micro-Expression Recognition (MER)** is a critical frontier in affective computing with applications in clinical psychology, threat detection, and human-computer interaction. However, training robust deep neural networks is severely bottlenecked by two twin challenges:
1. **Data Scarcity**: High-speed video sequences are extremely scarce (e.g., CASME II contains only 255 clips) due to the necessity of manual frame-by-frame indexing by FACS-certified practitioners. Standard spatiotemporal networks suffer from severe overfitting.
2. **Subject Identity Bias**: Static structural facial features (bone structure, skin texture) are numerically stronger than transient muscle movements. Deep networks naturally latch onto these static features, classifying emotions based on *who the person is* rather than *what they are expressing*, leading to poor generalization on unseen subjects.

**Proposed Solution**: I implement a unified, parameter-efficient (~382K parameters) hybrid network combining a modality-aware 3D-CNN backbone, a parameter-free 3D attention mechanism, and a Sequence Transformer. To enforce subject-invariance, I train the backbone via joint Domain-Adversarial learning (Gradient Reversal Layer) and Supervised Contrastive Learning (SupCon) with a Cross-Batch Memory queue.

---

## 2. Step-by-Step Methodology & Parameters

### Step 1: Preprocessing & Physics-Based Modality Extraction
To standardize inputs and remove static identity features, raw RGB video clips are converted into physics-based dynamic motion representations.
- **EVM (Eulerian Video Magnification)**: Decomposes frames spatially using a Laplacian pyramid, applies an ideal temporal bandpass filter to isolate facial muscle frequencies, and amplifies the motion signal before pyramid collapse.
- **Modality Extraction**: Dense Farneback optical flow is calculated between consecutive frames to yield horizontal flow ($u$), vertical flow ($v$), and optical skin strain ($os$), capturing spatial shear and skin compression.
- **Temporal Standardization**: Variable frame sequences (ranging from 10 to 150 frames) are uniformly sampled using linear temporal interpolation to output exactly $L=33$ frames ($T=32$ frame pairs).

#### Core References:
* *EVM*: **Wu et al., TOG 2012** (Eulerian Video Magnification).
* *Optical Flow*: **Farneback, 2003** (Dense polynomial expansion flow).

#### Preprocessing & Modality Parameters:
| Parameter | Value Setting | How it is Used | Why it was Chosen (Rationale) |
| :--- | :--- | :--- | :--- |
| **$\alpha$ (EVM Amplification)** | `20.0` | Multiplies the bandpass-filtered temporal motion signal before frame reconstruction. | Amplifies subtle sub-pixel facial muscle contractions to make them visible to the spatial feature kernels. |
| **$f_{\text{low}}$ (EVM Cutoff)** | `0.4 Hz` | Lower boundary of the temporal bandpass FFT filter. | Excludes slow, irrelevant motions such as head sway, breathing, or camera drift. |
| **$f_{\text{high}}$ (EVM Cutoff)** | `3.0 Hz` | Upper boundary of the temporal bandpass FFT filter. | Excludes high-frequency noise (e.g., pixel jitter, sensor noise, or environmental lighting flicker). |
| **$L_{\text{pyr}}$ (Pyramid Levels)** | `4` | Constructs a 4-level spatial Laplacian pyramid. | Allows spatial motion filtering at multiple resolutions (from coarse contours to fine micro-twitches). |
| **Assumed FPS** | `30` | Establishes the frequency index boundaries for filtering. | Calibrates temporal filtering bounds to standard camera rates (30 FPS). |
| **Target Length ($L$)** | `33 frames` | Interpolates variable input sequences using uniform linear sampling. | Standardizes video lengths to exactly 33 frames, providing $T=32$ consecutive frame pairs for dense optical flow computation. |
| **Input Resolution** | `224 × 224` | Bilinear resizing of cropped faces. | Standardizes spatial dimensions for uniform volumetric 3D convolutional operations. |

---

### Step 2: Hybrid Network Architecture
To process the spatiotemporal motion tensors without overfitting, I design a highly parameter-efficient hybrid network:
- **Modality-Aware Shallow 3D-CNN Backbone (STSTNet-3D)**: Routes flow-u, flow-v, and skin strain through three parallel, unshared branches. Weights are unshared to let kernels specialize in each distinct physical motion type. Standard 3D convolutions use a temporal kernel depth of 1, preventing premature temporal mixing.
- **3D SimAM Attention**: A parameter-free attention mechanism that computes an analytical energy score for each neuron based on spatial-temporal neighborhood statistics. It rescales feature maps element-wise, highlighting active facial regions without adding learnable parameters.
- **SLSTT Sequence Transformer**: Spatial dimensions are averaged using global pooling. Deterministic sinusoidal positional encodings are added to the sequence, which is then processed by a 2-layer pre-norm Sequence Transformer to capture temporal dynamics (onset $\to$ apex $\to$ offset).

#### Core References:
* *STSTNet Backbone*: **Liong et al., FG 2019** (Shallow parameter-efficient 3D-CNN).
* *SimAM Attention*: **Yang et al., ICML 2021** (Neuroscience-inspired parameter-free attention).
* *SLSTT Transformer*: **Zhang et al., IEEE TAFFC 2022** (Sequence-level spatiotemporal transformer).
* *Positional Encodings*: **Vaswani et al., NeurIPS 2017** (Sinusoidal positional encoding).

#### Hybrid Architecture Parameters:
| Parameter | Value Setting | How it is Used | Why it was Chosen (Rationale) |
| :--- | :--- | :--- | :--- |
| **Branch Split** | `1 channel × 3` | Splitting the 3-channel input tensor into three independent branches (flow-u, flow-v, strain). | Implements a **weight-unshared design**. Since horizontal, vertical, and skin strain movements have different physics, unshared weights allow separate kernels to specialize in each modality. |
| **Backbone Conv3D Kernels** | `[1, 3, 3]` | Convolutional layers have a temporal kernel depth of 1 (width/height = 3). | Preserves the temporal dimension entirely (32 steps are not mixed spatially), delegating all sequence modeling to the downstream Transformer. |
| **Dropout Rate** | `0.3` | Applied after activation functions and before fully connected heads. | Regularizes the network to prevent the model from memorizing specific subject identities. |
| **Spatial Pooling** | `MaxPool3D [1,2,2]` | Halves spatial size ($224 \to 112$) while preserving temporal depth (32). | Reduces spatial dimensionality to prevent over-parameterization while retaining temporal fidelity. |
| **SimAM Suppression ($\lambda$)** | `1e-4` | Added to the variance in the SimAM energy function: $e_t = \frac{(t-\mu)^2}{4(\sigma^2 + \lambda)} + 0.5$. | Acts as a regularizer to prevent division by zero during energy minimization in homogeneous background regions. |
| **Adaptive Pooling Output** | `[32, 1, 1]` | Crushes spatial dimensions ($112 \times 112 \to 1 \times 1$). | Aggregates all spatial motions into 32 sequential tokens, preparing the features for transformer sequence processing. |
| **Token Dimension ($d_{\text{model}}$)** | `96` | Output of modality channel concatenation (3 branches × 32 features). | Combines spatial descriptors from all three modalities into a single spatiotemporal token. |
| **Transformer Layers** | `2` | 2-layer pre-norm Multi-Head Self-Attention. | Captures global temporal progression (onset $\to$ apex $\to$ offset) without over-parameterization. |
| **Attention Heads** | `4` | Splits the 96-d token into 4 subspaces of size 24. | Allows parallel attention mechanisms to focus on different facial regions simultaneously (e.g., mouth vs. eyebrows). |
| **Feed-Forward Size ($d_{\text{ff}}$)** | `256` | Intermediate projection dimension inside FFN. | Maps sequence tokens to a higher-dimensional space for richer non-linear feature representation. |

---

### Step 3: Joint Domain-Adversarial & Contrastive Learning
The output feature vector is routed through a triple-head wrapper to enforce subject-invariance and optimize emotion class boundaries:
- **Gradient Reversal Layer (GRL)**: Placed before the identity classification head. It acts as an identity map during forward propagation, but reverses and scales identity gradients by $-\lambda$ during backward propagation. This forces the backbone to strip out subject identity details.
- **Supervised Contrastive Loss (SupCon)**: Applied to 64-d projection embeddings to pull same-emotion embeddings together and push different emotions apart.
- **Cross-Batch Memory (XBM)**: A FIFO queue of size 64 that caches historical embeddings. This bypasses VRAM-bounded batch constraints ($B=2$), expanding the contrastive comparison size to 66 anchors.
- **Focal Loss**: Optimizes emotion classification, using inverse-frequency weights and a focusing parameter ($\gamma=2.0$) to penalize majority classes and handle dataset imbalance.

#### Core References:
* *GRL Domain Adaptation*: **Ganin & Lempitsky, JMLR 2016** (Domain-adversarial neural networks).
* *Supervised Contrastive Loss*: **Khosla et al., NeurIPS 2020** (SupCon representation learning).
* *Cross-Batch Memory*: **Wang et al., CVPR 2020** (XBM for memory-efficient contrastive mining).
* *Focal Loss*: **Lin et al., ICCV 2017** (Focal loss for dense object detection / class imbalance).

#### Learning & Loss Parameters:
| Parameter | Value Setting | How it is Used | Why it was Chosen (Rationale) |
| :--- | :--- | :--- | :--- |
| **Embedding Size** | `64-d` | Projector output size, L2-normalized. | Standardizes features for contrastive comparisons, preventing the model from embedding static identity traits. |
| **GRL Annealing Slope ($\gamma$)** | `10.0` | Controls GRL coefficient scheduling: $\lambda(p) = \frac{2}{1+e^{-\gamma \cdot p}} - 1$. | Dictates the steepness of the GRL weight ramp. Ramping slowly from 0.0 to 1.0 allows the backbone to learn basic spatiotemporal features before GRL forces identity-invariant learning. |
| **Identity Head Weight** | `0.40` | Scaling coefficient for GRL backpropagated gradients during Stage 4. | Balances the adversarial identity loss against the classification loss to prevent the adversary from destabilizing feature extraction. |
| **SupCon Temperature ($\tau$)** | `0.10` | Cosine similarity scaling parameter. | A low temperature (0.10) scales similarity scores exponentially, penalizing hard negatives and forcing the model to construct extremely tight, class-specific emotion clusters. |
| **XBM Queue Size ($M$)** | `64` | Caches detached historical embeddings in a FIFO queue. | **Bypasses VRAM boundaries.** Due to physical batch size constraints ($B=2$), the memory bank expands contrastive comparisons to $64 + 2 = 66$ anchors without requiring additional gradient memory. |
| **Focal Loss Gamma ($\gamma_{\text{focal}}$)** | `2.0` | Modulating exponent: $\text{FL} = -\alpha_t (1-p_t)^{\gamma} \log(p_t)$. | Automatically down-weights easy, well-classified majority samples to focus training gradients on hard, misclassified minority samples. |
| **Label Smoothing** | `0.05` | Distributes 5% target weight to incorrect classes during Focal Loss computation. | Prevents model overconfidence on noisy and highly subjective micro-expression boundary frames. |
| **Focal Loss Weights ($\alpha$)** | `Inverse Freq` | Per-class scaling derived from class distribution. | Mitigates severe class imbalance (Negative/Others vs. Positive/Surprise) in the training corpus. |

---

### Step 4: Macro-to-Micro Transfer Learning
To combat sample scarcity, I implement a two-phase transfer learning framework:
1. **Phase 1 (Macro Pre-training)**: I pre-train the full architecture on 293 macro-expression clips from the CAS(ME)² dataset. The GRL adversary is disabled ($\lambda_{id} = 0.0$), allowing the model to learn general spatiotemporal facial action kinematics.
2. **Phase 2 (Micro Fine-tuning)**: Pre-trained weights are transferred to a micro-expression model. Because the number of subjects increases from $N=20$ (macro) to $N=26$ (micro), the identity classifier layer is re-initialized. Fine-tuning is conducted with GRL enabled ($\lambda_{id} = 0.40$) to strip subject identity.

#### Transfer Learning & Optimization Parameters:
| Parameter | Value Setting | How it is Used | Why it was Chosen (Rationale) |
| :--- | :--- | :--- | :--- |
| **Optimizer** | `AdamW` | Decouples weight decay ($L_2$ regularisation) from gradient updates. | Prevents model parameters from growing too large and memorizing subject characteristics. |
| **Weight Decay** | `1e-4` | $L_2$ regularization coefficient. | Regularizes weights to prevent overfitting on the small fine-tuning dataset. |
| **Physical Batch Size** | `2` | Number of video sequence tensors processed per iteration. | The maximum batch size that fits in GPU VRAM (RTX 4060 8.6 GB) when loading massive 5D volumetric flow inputs. |
| **Gradient Accumulation** | `4 steps` | Accumulates gradients over 4 batches before stepping the optimizer. | Simulates an effective batch size of **$B_{\text{eff}} = 8$** ($2 \times 4$) to stabilize gradients under hardware constraints. |
| **Stage 3 Peak LR** | `1e-4` | Learning rate for scratch LOSO training. | The optimal step size for joint contrastive-adversarial optimization. |
| **Stage 4 Phase 1 Peak LR** | `1e-4` | Learning rate for macro pre-training. | Moderately high rate to rapidly extract facial motion dynamics with GRL disabled. |
| **Stage 4 Phase 2 Peak LR** | `5e-5` | Learning rate for micro fine-tuning. | A smaller learning rate to fine-tune the spatiotemporal weights on micro-expressions without overriding the pre-trained macro mechanics. |
| **Scheduler Minimum LR** | `1e-7` | The lower bound for CosineAnnealingLR. | Ensures smooth, stable convergence as the learning rate decays along a cosine curve. |

---

## 3. Results & Evaluation Key Findings

1. **Stage 3 (26-Fold scratch LOSO)**: Achieves a grand mean accuracy of **58.11%** and Macro F1 of **33.31%**. This validates the baseline model's classification capability on subject-disjoint splits.
2. **Stage 4 (Macro-to-Micro Transfer)**: Pre-training achieves **70.37%** validation accuracy on macro-expressions. Fine-tuning on micro-expressions yields **37.78%** accuracy and **18.41%** Macro F1.
3. **Identity-Invariance Verification**: Projecting 64-d projection embeddings via t-SNE shows that subject IDs are completely and uniformly mixed (proving the GRL successfully stripped static identity details), while emotion classes form distinct clusters.
4. **Minority Class Starvation**: The confusion matrix reveals a major bottleneck: the model fails to predict Positive and Surprise classes (0% recall). This is a direct consequence of dataset imbalance, as these classes contain only 45 and 34 fine-tuning samples respectively.

---
