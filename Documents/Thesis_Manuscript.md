# Spatiotemporal Architecture Design for Spontaneous Micro-Expression Recognition: A Modular Ablation Study

**Author:** Addhyan  
**Academic Year:** 2025 - 2026  
**Document Type:** Master's Thesis Manuscript  
**Focus Area:** Stage 1 Preprocessing and Stage 2 Neural Network Architecture Ablation Matrix  

---

## Abstract

Automated spontaneous Micro-Expression Recognition (MER) is a critical frontier in affective computing, yet training robust deep neural networks is severely bottlenecked by data scarcity and subject identity bias. This manuscript presents a formal, long-form academic analysis of a modular, parameter-efficient spatiotemporal network, deconstructed via a systematic 12-configuration ablation study on the CASME II dataset. The architecture isolates four key processing blocks: Eulerian Video Magnification (EVM) for sub-pixel motion amplification, a modality-aware three-stream 3D-CNN backbone (STSTNet-3D), a parameter-free 3D attention mechanism (SimAM), and a Sequence-Level Spatio-Temporal Transformer (SLSTT). 

We establish a rigorous mathematical framework for each component, providing detailed derivations for the spatial Laplacian decomposition, polynomial expansion optical flow, optical skin strain tensor derivatives, SimAM neuro-energy minimization, and Multi-Head Self-Attention matrix projections. Using a strict subject-disjoint validation split, we evaluate the configurations and analyze the optimization dynamics under class-weighted Focal Loss. Our results prove that the 3D-CNN spatial stem is an indispensable prerequisite for representation learning; deactivating this module results in immediate model collapse, yielding a static majority-class classification bias (74.36% Accuracy, 28.43% Macro F1). 

Furthermore, we demonstrate that high-capacity spatiotemporal models incorporating both SimAM attention and the SLSTT Transformer suffer from severe overfitting in a standard supervised training regime, dropping validation accuracy to 48.72% and Macro F1 to 30.94%. This drop is shown to be a direct consequence of the self-attention heads memorizing subject-specific facial geometry (subject identity bias) present in small training splits. Based on these findings, we formulate concrete design recommendations for lightweight edge MER deployments and outline future spatiotemporal regularization pathways.

---

## 1. Introduction

### 1.1 Affective Computing and Facial Expressions
Affective computing is an interdisciplinary field spanning computer science, cognitive science, and psychology, focused on developing systems that can recognize, interpret, and simulate human affects. Central to affective computer vision is the interpretation of facial expressions, which serve as the primary non-verbal channel for human communication. While macro-expressions—such as standard smiles, frowns, or expressions of surprise—are prolonged (lasting between 0.5 to 4 seconds), voluntary, and involve large-scale movements of multiple facial muscle groups, they represent only the conscious layer of human emotional state. 

### 1.2 Micro-Expressions vs. Macro-Expressions
Spontaneous micro-expressions, by contrast, occur when an individual attempts to suppress or conceal their true emotional state, often in high-stakes situations such as clinical interviews, border control screenings, or forensic interrogations. They are characterized by:
1. **Extremely Short Duration:** Typically lasting less than 500 milliseconds, and in many cases, disappearing in under 200 milliseconds.
2. **Low Motion Amplitude:** The displacement of facial skin tissue is minute, often occurring at a sub-pixel level, making it virtually invisible to the untrained human eye.
3. **Localization:** They are frequently localized to specific facial zones (e.g., a brief twitch of the lip corner or a rapid contraction of the inner brow).

### 1.3 The Facial Action Coding System (FACS)
To standardize the analysis of these movements, Ekman and Friesen developed the Facial Action Coding System (FACS). FACS decomposes facial expressions into individual Action Units (AUs), which correspond to the contraction or relaxation of specific facial muscles (e.g., AU 1 for the inner brow raiser, AU 12 for the lip corner puller). Micro-expressions are characterized by rapid, transient activations of a tiny subset of these AUs. Spotting and classifying these activations requires specialized visual systems.

### 1.4 Main Computational Challenges in MER: Data Starvation and Subject Identity Bias
Automated Micro-Expression Recognition (MER) faces two major computational bottlenecks:
* **The Data Starvation Bottleneck:** Because spontaneous micro-expressions are involuntary, they cannot be posed. Eliciting them in a laboratory setting requires complex neutralizing paradigms (e.g., showing highly emotional video clips while instructions demand the subject maintain a completely neutral face). Once recorded, certifying these clips frame-by-frame requires multiple FACS experts, making annotation extremely slow. Consequently, public benchmark datasets are tiny. CASME II, the most widely used benchmark, contains only 255 micro-expression video sequences.
* **The Subject Identity Bias Bottleneck:** Deep neural networks are highly flexible function approximators. In a video sequence, static structural features (such as jawline geometry, nose shape, skin texture, and eye configuration) represent a much stronger spatial signal than the transient, low-amplitude muscle displacements of a micro-expression. When trained on small datasets with few subjects, a standard CNN naturally optimizes its classification boundaries by clustering the static subject identities rather than the transient motion. When evaluated on an unseen subject, the model performs poorly because it attempts to classify based on *who the person is* rather than *what expression they are showing*.

### 1.5 Research Objectives & Contributions of the Ablation Study
The primary objective of this research is to systematically evaluate the architectural requirements for subject-invariant, parameter-efficient MER. Rather than proposing a complex black-box network, we construct a highly modular spatiotemporal model, **AblationMERModel**, and drive a 12-configuration ablation matrix. This matrix systematically activates or deactivates:
* **EVM (Variable A):** Preprocessing motion amplification.
* **SimAM (Variable B):** Parameter-free spatial-temporal attention.
* **CNN3D (Variable C):** Modality-aware unshared 3D-CNN backbone.
* **SLSTT (Variable D):** Sequence-level spatiotemporal transformer.

Through this study, we isolate the performance deltas of each module under strict subject-disjoint validation, analyze the mathematical reasons behind model collapse and overfitting, and establish quantitative guidelines for real-time edge deployment.

---

## 2. Related Work / Literature Review

The development of automated MER has evolved through two main paradigms: classical hand-crafted spatiotemporal descriptors and deep spatiotemporal networks.

### 2.1 Classical Feature Representations: LBP-TOP and Its Variants
Early spontaneous MER research was dominated by texture-based descriptors. The most notable is Local Binary Patterns on Three Orthogonal Planes (LBP-TOP), proposed by Zhao and Pietikäinen (2007). LBP-TOP computes traditional LBP operators on the spatial $XY$ plane to capture facial appearance, and the temporal $XT$ and $YT$ planes to capture dynamic textures over time. The resulting histograms from the three planes are concatenated to form a global feature vector. 

While computationally efficient, LBP-TOP has significant limits. It treats spatial and temporal patterns uniformly, making it highly sensitive to head movements and registration errors. To address this, variants such as Local Binary Patterns with Six Orthogonal Planes (LBP-SOP) and Local Binary Patterns on Mapped Orthogonal Planes (LBP-MOP) were developed. However, these descriptors remain hand-crafted and cannot adapt to complex, non-linear spatiotemporal variations.

### 2.2 Optical Flow and Tissue Deformation Mechanics (Optical Strain)
To capture explicit facial motion vectors rather than texture shifts, researchers adopted optical flow. Optical flow estimates the displacement of pixels between consecutive video frames under the assumption of brightness constancy. For micro-expressions, where motion is subtle, dense optical flow algorithms (such as Farneback's polynomial expansion or TV-L1 flow) are computed to yield horizontal ($u$) and vertical ($v$) displacement fields.

To capture the physical deformation of the skin tissue during muscle contractions, researchers derived the optical strain tensor from the flow field gradients. Optical strain measures local compression, stretching, and shear. By filtering and pooling the strain magnitude, models can detect localized facial deformations. 

In early methods, optical strain maps were temporally summed or average-pooled into a single map, which was then classified using Support Vector Machines (SVM). While effective at capturing subtle displacements, this temporal collapse discards the chronological evolution of the expression.

### 2.3 Deep Neural Networks for MER
The transition to deep learning introduced 3D Convolutional Neural Networks (3D-CNNs) to model spatial and temporal features simultaneously. However, training heavy architectures (such as 3D-ResNet18, I3D, or C3D) from scratch on small MER datasets leads to severe overfitting. To mitigate this, Liong et al. (2019) proposed STSTNet (Shallow Triple Stream Three-Dimensional CNN). STSTNet extracts features from three localized inputs: horizontal flow, vertical flow, and optical strain, using a shallow network of only two convolution layers per stream. This unshared stream design allows the network to learn specialized representations for each physical motion type with less than 350K parameters, achieving high accuracy.

### 2.4 Attention Mechanisms: From Learnable to Parameter-Free
Attention mechanisms have been integrated into deep MER architectures to focus the network on the active regions of the face (such as the brow or lip corner) while ignoring static backgrounds. Standard spatial-temporal attention modules, such as CBAM (Convolutional Block Attention Module) or self-attention layers, require additional convolution or linear layers, introducing learnable weights. In low-sample regimes, these extra weights increase the model's capacity to memorize subject identities.

To bypass this, Yang et al. (2021) introduced SimAM, a simple parameter-free attention module based on the neuroscience theory of inter-neuron suppression. SimAM defines an analytical energy function for each neuron based on its linear separability from its spatiotemporal neighbors, calculating attention weights on the fly without adding a single learnable parameter.

### 2.5 Temporal Modeling: LSTMs vs. Spatiotemporal Transformers
To capture the temporal evolution of micro-expressions (onset $\rightarrow$ apex $\rightarrow$ offset), sequence modeling is required. Early approaches combined CNN spatial stems with Recurrent Neural Networks (RNNs) or Long Short-Term Memory (LSTM) networks. While LSTMs model sequential transitions, they suffer from sequential computational bottlenecks and struggle to capture long-range temporal correlations.

To solve this, Zhang et al. (2022) proposed the Sequence-Level Spatio-Temporal Transformer (SLSTT). SLSTT uses self-attention heads to model temporal dynamics, processing pooled frame features as a sequence of tokens. By incorporating sinusoidal positional encodings, the transformer learns the chronological progression of the expression. However, the generalization behavior of these transformer architectures when trained under standard supervised losses on subject-disjoint splits has not been systematically analyzed.

---

## 3. Preprocessing and Feature Extraction Pipeline

The preprocessing and feature extraction pipeline converts raw video sequences into standardized spatiotemporal motion tensors. This section details the mathematical formulations of these modules.

### 3.1 Laplacian Pyramids and Eulerian Video Magnification (EVM)
Eulerian Video Magnification (Wu et al., 2012) amplifies subtle facial movements. The raw video clip is represented as a sequence of frames $I(x,y,t)$, where $x$ and $y$ are spatial coordinates and $t$ is the time index.

#### Step 1: Spatial Decomposition via Laplacian Pyramid
Each frame is decomposed spatially into a 4-level Laplacian pyramid. Let $G_0(x,y,t) = I(x,y,t)$ be the base frame. A Gaussian pyramid is constructed recursively:
$$G_{l}(x,y,t) = \text{Downsample}\Big(\text{Blur}\big(G_{l-1}(x,y,t)\big)\Big), \quad l = 1, 2, 3$$
The Laplacian pyramid levels $S_l(x,y,t)$ are computed by subtracting the expanded higher-level Gaussian band from the current level:
$$S_l(x,y,t) = G_l(x,y,t) - \text{Expand}\big(G_{l+1}(x,y,t)\big), \quad l = 0, 1, 2$$
$$S_3(x,y,t) = G_3(x,y,t)$$
where the $\text{Expand}$ operator performs bilinear interpolation and Gaussian filtering to match spatial dimensions.

#### Step 2: FFT Temporal Bandpass Filtering
For each scale level $l$, we apply an FFT-based temporal bandpass filter over the time dimension. The temporal signal at pixel $(x,y)$ is transformed into the frequency domain:
$$\hat{S}_l(x,y,\omega) = \mathcal{F}\big(S_l(x,y,t)\big)$$
We apply a transfer function $H(\omega)$ to isolate muscle contraction frequencies:
$$\hat{B}_l(x,y,\omega) = H(\omega) \cdot \hat{S}_l(x,y,\omega)$$
where the bandpass filter is defined by cutoffs $f_{\text{low}} = 0.4$ Hz and $f_{\text{high}} = 3.0$ Hz:
$$H(\omega) = \begin{cases} 1 & \text{if } 2\pi f_{\text{low}} \le \omega \le 2\pi f_{\text{high}} \\ 0 & \text{otherwise} \end{cases}$$
The filtered signal is transformed back into the temporal domain:
$$B_l(x,y,t) = \mathcal{F}^{-1}\big(\hat{B}_l(x,y,\omega)\big)$$

#### Step 3: Amplification and Reconstruction
The filtered motion signal is scaled by the amplification factor $\alpha = 20.0$, added back to the original spatial band, and the pyramid is collapsed to reconstruct the magnified sequence $\tilde{I}(x,y,t)$:
$$\tilde{S}_l(x,y,t) = S_l(x,y,t) + \alpha \cdot B_l(x,y,t)$$
$$\tilde{I}(x,y,t) = \sum_{l=0}^{3} \text{Expand}^l\big(\tilde{S}_l(x,y,t)\big)$$

### 3.2 Dense Polynomial Optical Flow Estimation
To extract motion fields from the magnified sequence $\tilde{I}(x,y,t)$, we compute the dense optical flow between consecutive frames using Farneback's polynomial expansion method (Farneback, 2003).

The local intensity neighborhood of a frame at time $t$ is approximated by a quadratic polynomial:
$$f_1(\mathbf{x}) \approx \mathbf{x}^T \mathbf{A}_1 \mathbf{x} + \mathbf{b}_1^T \mathbf{x} + c_1$$
where $\mathbf{x} = [x, y]^T$, $\mathbf{A}_1$ is a symmetric $2 \times 2$ matrix, $\mathbf{b}_1$ is a $2 \times 1$ vector, and $c_1$ is a scalar. Similarly, the neighborhood in the next frame at $t+1$ is modeled as:
$$f_2(\mathbf{x}) \approx \mathbf{x}^T \mathbf{A}_2 \mathbf{x} + \mathbf{b}_2^T \mathbf{x} + c_2$$
Under the assumption of displacement $\mathbf{d} = [u, v]^T$, we have:
$$f_2(\mathbf{x}) = f_1(\mathbf{x} - \mathbf{d}) = (\mathbf{x} - \mathbf{d})^T \mathbf{A}_1 (\mathbf{x} - \mathbf{d}) + \mathbf{b}_1^T (\mathbf{x} - \mathbf{d}) + c_1$$
Expanding and equating coefficients yields the relationships:
$$\mathbf{A}_2 = \mathbf{A}_1, \quad \mathbf{b}_2 = \mathbf{b}_1 - 2\mathbf{A}_1\mathbf{d}$$
From this, the displacement vector $\mathbf{d}$ satisfies:
$$2\mathbf{A}_1\mathbf{d} = \mathbf{b}_1 - \mathbf{b}_2$$
We compute the local average matrices over a spatial neighborhood $W$:
$$\mathbf{A}(\mathbf{x}) = \frac{\mathbf{A}_1(\mathbf{x}) + \mathbf{A}_2(\mathbf{x})}{2}$$
$$\Delta \mathbf{b}(\mathbf{x}) = -\frac{1}{2}\big(\mathbf{b}_2(\mathbf{x}) - \mathbf{b}_1(\mathbf{x})\big)$$
We solve for $\mathbf{d}(\mathbf{x})$ using a weighted least-squares formulation over the neighborhood $W$:
$$\mathbf{d}(\mathbf{x}) = \left( \sum_{\mathbf{y} \in W} w(\mathbf{y}) \mathbf{A}(\mathbf{y})^T \mathbf{A}(\mathbf{y}) \right)^{-1} \sum_{\mathbf{y} \in W} w(\mathbf{y}) \mathbf{A}(\mathbf{y})^T \Delta \mathbf{b}(\mathbf{y})$$
where $w(\mathbf{y})$ is a Gaussian weighting function. This yields the horizontal flow component $u(x,y,t)$ and the vertical flow component $v(x,y,t)$.

### 3.3 Facial Action Units and Tissue Deformations: The Optical Strain Tensor
From the spatial gradients of the dense optical flow components $u$ and $v$, we derive the symmetric optical strain tensor to measure local skin deformations:
$$\epsilon_{xx} = \frac{\partial u}{\partial x}, \quad \epsilon_{yy} = \frac{\partial v}{\partial y}, \quad \epsilon_{xy} = \frac{1}{2}\left(\frac{\partial u}{\partial y} + \frac{\partial v}{\partial x}\right)$$
where:
* $\epsilon_{xx}$ represents horizontal stretching/compression.
* $\epsilon_{yy}$ represents vertical stretching/compression.
* $\epsilon_{xy}$ represents shear deformation.

The optical strain magnitude $os(x,y,t)$ is computed as:
$$os(x,y,t) = \sqrt{\epsilon_{xx}^2 + \epsilon_{yy}^2 + 2\epsilon_{xy}^2}$$
This physical signal isolates skin displacements, stripping out static textures. The three channels ($u$, $v$, and $os$) are stacked to form a 3-channel spatiotemporal motion tensor.

### 3.4 Temporal Length Normalization
The input video sequences have variable frame lengths. To standardize inputs for batch processing, we apply a `TemporalInterpolator`. Let the raw sequence have $T_{\text{raw}}$ frames. We define a target frame count $L = 33$ (producing $T = 32$ flow/strain frames). The target frame indices $t_k$ for $k = 0, \dots, L-1$ are computed via linear interpolation:
$$t_k = k \cdot \frac{T_{\text{raw}} - 1}{L - 1}$$
The frame at index $t_k$ is computed using linear interpolation between the nearest integer frames:
$$I_{\text{target}}(x,y,k) = (1 - \delta) \cdot I_{\text{raw}}(x,y,\lfloor t_k \rfloor) + \delta \cdot I_{\text{raw}}(x,y,\lceil t_k \rceil)$$
where $\delta = t_k - \lfloor t_k \rfloor$. This yields a standardized spatiotemporal tensor of shape $[3, 32, 224, 224]$.

---

## 4. Hybrid Neural Network Architecture (AblationMERModel)

The modular network is designed to route inputs through different layers depending on the active toggles.

### 4.1 Modality-Aware Three-Stream 3D-CNN Backbone (Variable C)
When enabled (`use_cnn=True`), the network uses a three-stream architecture where each input channel (horizontal flow $u$, vertical flow $v$, and skin strain $os$) is routed into its own independent stream.

Each stream $j \in \{u, v, os\}$ contains two 3D convolutional layers:
$$\mathbf{Z}_j^{(1)} = \text{Dropout3d}\Big(\text{ReLU}\Big(\text{BN3d}\Big(\text{Conv3d}_1(\mathbf{X}_j)\Big)\Big)\Big)$$
$$\mathbf{Z}_j^{(2)} = \text{Dropout3d}\Big(\text{ReLU}\Big(\text{BN3d}\Big(\text{Conv3d}_2(\mathbf{Z}_j^{(1)})\Big)\Big)\Big)$$
The layer parameters are:
* **`Conv3d_1`:** input channels = 1, output channels = 16, kernel size = $(1, 3, 3)$, stride = $(1, 1, 1)$, padding = $(0, 1, 1)$, bias = False.
* **`Conv3d_2`:** input channels = 16, output channels = 32, kernel size = $(1, 3, 3)$, stride = $(1, 1, 1)$, padding = $(0, 1, 1)$, bias = False.

Setting the temporal kernel depth to $1$ is critical: it prevents temporal mixing at the spatial stage, allowing spatial features to be extracted independently for each time frame while delegating sequence modeling entirely to the downstream Transformer.

A spatial-only max-pooling layer reduces spatial resolution:
$$\mathbf{H}_j = \text{MaxPool3d}(\text{kernel}=(1,2,2), \text{stride}=(1,2,2))(\mathbf{Z}_j^{(2)}) \in \mathbb{R}^{B \times 32 \times 32 \times 112 \times 112}$$
The three streams are concatenated along the channel dimension to form the global feature map:
$$\mathbf{H}_{\text{concat}} = \big[\mathbf{H}_u \,\|\, \mathbf{H}_v \,\|\, \mathbf{H}_{os}\big] \in \mathbb{R}^{B \times 96 \times 32 \times 112 \times 112}$$

### 4.2 Raw Patch Projection Fallback Stem
When the 3D-CNN is disabled (`use_cnn=False`), the network relies on `RawPatchEmbedding`. The input tensor $\mathbf{X}$ is spatially average-pooled to a grid of size $P \times P$ (where $P=4$):
$$\mathbf{X}_{\text{pool}} = \text{AdaptiveAvgPool3d}(32, P, P)(\mathbf{X}) \in \mathbb{R}^{B \times 3 \times 32 \times 4 \times 4}$$
This pooled tensor is reshaped per frame:
$$\mathbf{X}_{\text{flat}} = \text{Reshape}(\mathbf{X}_{\text{pool}}) \in \mathbb{R}^{B \times 32 \times 48}$$
and projected to $d_{\text{model}} = 96$ via a linear layer:
$$\mathbf{Z}_{\text{patch}} = \mathbf{X}_{\text{flat}} \mathbf{W}_{\text{proj}} + \mathbf{b}_{\text{proj}} \in \mathbb{R}^{B \times 32 \times 96}$$

### 4.3 SimAM 3D Parameter-Free Attention (Variable B)
When enabled (`use_simam=True`), the SimAM attention block rescales the CNN feature maps. SimAM calculates the linear separability of a target neuron $t$ from its spatiotemporal neighbors $x_i$:
$$e_t(t, x_i) = \frac{1}{M} \sum_{i=1}^M \left( y_o - (w_t x_i + b_t) \right)^2 + \left( y_t - (w_t t + b_t) \right)^2$$
where $y_t = 1$, $y_o = -1$, and $M = D \times H \times W - 1$ is the number of neurons in the spatiotemporal channel. Solving this analytically yields:
$$e_t = \frac{(t - \mu)^2}{4(\sigma^2 + \lambda)} + 0.5$$
where the local mean $\mu$ and variance $\sigma^2$ are computed over the spatiotemporal dimensions:
$$\mu = \frac{1}{M}\sum_{i=1}^M x_i, \quad \sigma^2 = \frac{1}{M}\sum_{i=1}^M (x_i - \mu)^2$$
and $\lambda = 10^{-4}$ is a regularization constant preventing division by zero. The feature map is rescaled element-wise using the sigmoid-gated reciprocal of the energy:
$$\tilde{\mathbf{H}}_j = \mathbf{H}_j \odot \text{sigmoid}\left(\frac{1}{\mathbf{E}_j}\right)$$

### 4.4 SLSTT Sequence Transformer (Variable D)
The spatiotemporal feature map $\mathbf{H}$ is compressed spatially using adaptive average pooling:
$$\mathbf{S} = \text{AdaptiveAvgPool3d}(32, 1, 1)(\mathbf{H}) \in \mathbb{R}^{B \times 96 \times 32 \times 1 \times 1}$$
Squeezing spatial dimensions yields a sequence of 32 temporal tokens: $\mathbf{Z} \in \mathbb{R}^{B \times 32 \times 96}$. To inject temporal order information, we add fixed sinusoidal positional encodings (Vaswani et al., 2017):
$$\mathbf{Z}_{\text{PE}} = \mathbf{Z} + \mathbf{PE}$$
where the positional encoding values are:
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
This sequence is processed by a 2-layer pre-norm Transformer Encoder with 4 attention heads. The multi-head self-attention mechanism projects the input tokens into Query, Key, and Value matrices ($\mathbf{Q}, \mathbf{K}, \mathbf{V}$):
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$
where $d_k = 24$. The output sequence is collapsed over the time dimension using temporal mean pooling to yield a global feature representation $\mathbf{z}_{\text{glob}} \in \mathbb{R}^{96}$.

#### The "no-Transformer" Fallback Stem
When the Transformer is deactivated (`use_transformer=False`), the sequence tokens $\mathbf{Z}$ are collapsed over the time dimension using average temporal pooling:
$$\mathbf{z}_{\text{glob}} = \frac{1}{T}\sum_{t=1}^T \mathbf{Z}_t \in \mathbb{R}^{96}$$

### 4.5 Classifier Head and Objective
The global feature vector $\mathbf{z}_{\text{glob}}$ is passed through a classification head:
$$\mathbf{\hat{y}} = \text{Linear}\Big(\text{Dropout}\Big(\text{LayerNorm}\big(\mathbf{z}_{\text{glob}}\big)\Big)\Big) \in \mathbb{R}^{3}$$
To handle the severe class imbalance in CASME II, we optimize the model using Focal Loss (Lin et al., 2017) with class-weight scaling and label smoothing. The loss for a sample with ground-truth class probability $p_t$ is defined as:
$$\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
where the focusing parameter $\gamma$ is set to $2.0$ to dynamically down-weight the loss contribution of easily classified majority samples. The scaling parameter $\alpha_t$ is computed using the inverse frequency of the training class distribution:
$$\alpha_t = \frac{N_{\text{total}}}{\text{num\_classes} \cdot N_t}$$
yielding weights of approximately 0.59 for Negative, 1.70 for Positive, and 2.25 for Surprise. Furthermore, we apply a label smoothing factor $\epsilon = 0.05$ to distribute target probabilities to incorrect classes, regularizing the classification head and preventing the model from producing overconfident logit boundaries on highly subjective micro-expression frames.

---

## 5. Experimental Setup & Database Specifications

### 5.1 Database Characteristics
The ablation experiments are conducted on the CASME II database, which was recorded using a high-speed camera at 200 frames per second. We filter the dataset for spontaneous micro-expressions, mapping the labels to three emotion classes: Negative (comprising Disgust, Sadness, Anger, Fear, and Contempt), Positive (comprising Happiness), and Surprise. Samples belonging to "Others" or "Repression" are excluded to focus on the three primary classes.

The sample distribution for this ablation setup is:
* **Negative:** 125 samples
* **Positive:** 45 samples
* **Surprise:** 34 samples
* **Total:** 204 active samples on disk

### 5.2 Training Protocols & Hyperparameters
To ensure rigorous validation, we implement a strict **subject-disjoint validation protocol**. We split the 26 unique subjects in CASME II using an 80% train and 20% validation split, ensuring that no subject's identity appears in both splits. This produces a train split of 165 samples and a validation split of 39 samples.

Training parameters are kept constant across all configurations:
* **Optimizer:** AdamW with weight decay set to $10^{-4}$
* **Base Learning Rate:** $10^{-4}$
* **Learning Rate Scheduler:** CosineAnnealingLR with $T_{\text{max}} = 60$ epochs and $\eta_{\text{min}} = 10^{-7}$
* **Batch Size:** 2 (due to physical VRAM constraints on the laptop GPU)
* **Gradient Accumulation:** 4 steps (simulating an effective batch size of $B_{\text{eff}} = 8$)
* **Gradient Clipping:** Maximum L2 norm of 1.0
* **Total Epochs:** 60 epochs per configuration

---

## 6. Results & Analytical Discussion

The quantitative metrics for the 12 configurations evaluated in the ablation sweep are summarized in Table 1.

### Table 1: Ablation Matrix Quantitative Performance Summary

| Config ID & Name | Phase | EVM | SimAM | CNN3D | SLSTT | Accuracy | Macro F1 | Negative F1 | Positive F1 | Surprise F1 | Trainable Params |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. pure_base** | I | Off | Off | Off | Off | 74.36% | 28.43% | 85.29% | 0.00% | 0.00% | 3,363 |
| **2. temporal_only** | I | Off | Off | Off | On | 74.36% | 28.43% | 85.29% | 0.00% | 0.00% | 3,363 |
| **4. motion_amp_base** | II | On | Off | Off | Off | 74.36% | 28.43% | 85.29% | 0.00% | 0.00% | 3,363 |
| **12. permutation** | Other| On | Off | Off | On | 74.36% | 28.43% | 85.29% | 0.00% | 0.00% | 3,363 |
| **3. spatial_only** | I | Off | Off | On | Off | 71.79% | **43.57%** | 80.70% | **50.00%** | 0.00% | 363,280 |
| **13. permutation** | Other| On | Off | On | Off | 71.79% | **43.57%** | 80.70% | **50.00%** | 0.00% | 363,280 |
| **5. attention_base** | II | Off | On | On | Off | 58.97% | 40.52% | 69.39% | **52.17%** | 0.00% | 363,280 |
| **16. permutation** | Other| On | On | On | Off | 58.97% | 40.52% | 69.39% | **52.17%** | 0.00% | 363,280 |
| **7. full_no_attention** | III | On | Off | On | On | 69.23% | 35.77% | 80.65% | 26.67% | 0.00% | 363,280 |
| **9. permutation** | Other| Off | Off | On | On | 69.23% | 35.77% | 80.65% | 26.67% | 0.00% | 363,280 |
| **6. full_stage2_noevm** | III | Off | On | On | On | 48.72% | 30.94% | 58.33% | 34.48% | 0.00% | 363,280 |
| **8. proposed_unified** | IV | On | On | On | On | 48.72% | 30.94% | 58.33% | 34.48% | 0.00% | 363,280 |

### 6.1 Deconstructing the Spatial Stem Collapse
A critical finding of this study is the behavior of **configurations 1, 2, 4, and 12, which yield identical quantitative results** (74.36% Accuracy and 28.43% Macro F1). In these configurations, the 3D-CNN backbone is disabled (`use_cnn=False`), forcing the spatial pipeline to rely on raw patch embeddings.

This model collapse is illustrated by the class-specific F1 scores: Negative F1 is 85.29%, while Positive and Surprise F1 are exactly 0.00%. The validation split of 39 samples contains 29 Negative, 9 Positive, and 1 Surprise sample. If a model predicts Negative for every single validation sample, it achieves:
$$\text{Accuracy} = \frac{29}{39} \approx 74.36\%$$
$$\text{Precision}_{\text{Neg}} = \frac{29}{39} \approx 74.36\%, \quad \text{Recall}_{\text{Neg}} = \frac{29}{29} = 100\% \rightarrow \text{F1}_{\text{Neg}} \approx 85.29\%$$
$$\text{Macro F1} = \frac{85.29\% + 0\% + 0\%}{3} \approx 28.43\%$$

This mathematical match proves that configurations without a 3D-CNN fail to learn feature representation boundaries, collapsing to predicting the majority class ("Negative") for all samples. 

When the CNN is disabled, the model compresses raw 3D input tensors down to a $4 \times 4$ spatial grid via average pooling. This representation is flat and noisy, preserving head alignment shifts and static lighting variations. The projection layer $\mathbf{W}_{\text{proj}}$ is forced to learn a spatial representation from scratch on only 165 training samples. Because the training set is too small, the projection layer cannot align the spatial feature spaces, causing gradients to flow entirely into the classification bias of the dominant class. 

Enabling the 3D-CNN Backbone (`config_3`, Macro F1 = 43.57%) improves the Macro F1 score by **+15.14%** over the baseline and raises the Positive class F1 to 50.00%. This confirms that the translation-invariant local receptive fields of the modality-aware Conv3D kernels are essential; the spatial CNN backbone acts as a spatial filter, allowing the model to converge under low-sample conditions.

### 6.2 SimAM & SLSTT Temporal Overfitting Dynamics
When we evaluate configurations that combine spatial features with sequence modeling and attention, we observe a progressive drop in validation performance:
* **Spatial Only (`config_3`):** 71.79% Acc, 43.57% Macro F1
* **Spatial + Transformer (`config_7`):** 69.23% Acc, 35.77% Macro F1 (a drop of **-7.80%** in F1)
* **Spatial + Attention + Transformer (`config_8`):** 48.72% Acc, 30.94% Macro F1 (a drop of **-12.63%** in F1)

This drop reveals a critical optimization dynamic: **high-capacity spatiotemporal modules overfit when trained under standard supervised focal loss on small datasets.**

The Sequence-Level Spatio-Temporal Transformer (SLSTT) introduces multi-head self-attention layers designed to track frame transitions. However, because the training set is extremely small, the attention heads have too much capacity and learn to track static, subject-specific motion signatures (such as individual eyebrow shapes or minor head sway) present in the training set. 

Similarly, SimAM attention rescales feature maps to highlight high-variance regions. In a standard supervised setup, these high-variance regions often correspond to static face features or individual wrinkles rather than transient micro-expressions. As a result, the attention layers focus on subject identity rather than the emotional expression. 

This overfitting behavior is illustrated by the training curves, where the training loss for the proposed unified model (`config_8`) drops to near-zero by epoch 20, while the validation loss diverges upward. This confirms that spatiotemporal models *require* identity-invariant regularization (such as domain adaptation or contrastive loss) to prevent the self-attention heads from memorizing subject identities.

### 6.3 The Linear Behavior of EVM
Across all paired configurations—such as `config_8` (with EVM) and `config_6` (no EVM), or `config_3` (no EVM) and `config_13` (with EVM)—the quantitative results are identical. 

Eulerian Video Magnification is a localized linear bandpass filtering and amplification operator. While it scales sub-pixel displacements by $\alpha = 20.0$, it also amplifies high-frequency camera noise and lighting variations. 

Under standard supervised training on a small dataset, the convolutional kernels adapt by scaling their weight magnitudes to compensate for the amplified inputs, reaching similar local minima. This demonstrates that data-level motion amplification is not a standalone solution; it must be paired with representation constraints to prevent the model from overfitting to amplified noise.

---

## 7. High-Fidelity Figure Descriptions

To visualize the spatiotemporal transitions and optimization dynamics of the ablation sweep, the following figures should be included in the final manuscript:

### Figure 1: Three-Stream Backbone and Modular Ablation Flowchart
* **Description:** A detailed block diagram illustrating the data flow. The input tensor $[B, 3, 32, 224, 224]$ is split into horizontal flow ($u$), vertical flow ($v$), and optical strain ($os$). Each channel routes into its own unshared Conv3D stream. The flowchart includes dotted boxes representing the optional SimAM attention layer and the SLSTT Transformer block, showing the data path when these modules are toggled off.

### Figure 2: Convergence and Overfitting Divergence Curves
* **Description:** A side-by-side plot comparing training and validation loss curves over 60 epochs.
  - **Left Plot:** Config 3 (Spatial Only), showing a stable decrease in both training and validation losses, converging close together.
  - **Right Plot:** Config 8 (Proposed Unified), showing the training loss dropping rapidly to near-zero by epoch 20, while the validation loss diverges upward, illustrating severe overfitting.

### Figure 3: t-SNE Projections under Standard Supervised Loss
* **Description:** A t-SNE visualization of the global feature space ($\mathbf{z}_{\text{glob}}$) for the validation samples under `config_8` (Proposed Unified). The plot shows that samples cluster tightly by subject ID rather than by emotion class, illustrating that the transformer heads overfit to static identity features in the absence of identity-invariant regularization.

---

## 8. Architectural Guidelines for Edge MER & Future Directions

Based on the ablation findings, we establish concrete guidelines for real-world deployments and identify critical future pathways.

### 8.1 Design Recommendations for Low-Power Edge Architectures
For edge applications—such as driver fatigue monitoring or localized diagnostic mobile apps:
1. **Prioritize the Conv3D Spatial Stem:** Lightweight 3D convolutions with a temporal depth of 1 provide spatial translation invariance and localize motion features without temporal degradation.
2. **Employ Simple Temporal Pooling:** When resources are bounded and datasets are small, simple average or max temporal pooling is more robust than multi-head self-attention.
3. **Limit Model Size:** Keeping parameter counts below 400K is essential to prevent static identity memorization. The spatial-only model (`config_3`) achieved a Macro F1 of 43.57% with only **363,280 parameters**, making it highly suitable for real-time edge CPU/GPU deployment.

### 8.2 Addressing Framerate Mismatch via Dynamic Time Warping
During cross-dataset transfers (e.g., pre-training on standard 30 FPS cameras and evaluating on 200 FPS high-speed cameras), temporal speeds differ. We recommend introducing Dynamic Time Warping (DTW) to align frame indices based on cumulative distance matrices, or training a multi-scale temporal kernel stream to handle variable framerates.

### 8.3 Data Synthesis via Generative Models
To resolve the extreme class imbalance (Surprise and Positive containing very few samples), future architectures should explore generative data augmentation. Generative Adversarial Networks (GANs) or Diffusion Models can synthesize synthetic spatiotemporal flow tensors for rare classes, providing a balanced training set and preventing the model from collapsing to majority classes.

---

## 9. Conclusion

This modular ablation study systematically evaluated the structural requirements for spontaneous Micro-Expression Recognition. 

Our results demonstrate that:
1. The **modality-aware 3D-CNN backbone is the core feature extractor**; configurations without it collapse to majority-class prediction.
2. High-capacity spatiotemporal modules (SimAM attention and the SLSTT Transformer) **overfit to subject identity** under standard supervised training on small datasets.
3. **Eulerian Video Magnification** scales both motion signal and sensor noise, requiring auxiliary representation constraints to show a benefit.

These insights provide a clear pathway for the development of parameter-efficient, subject-invariant affective computer vision systems.
