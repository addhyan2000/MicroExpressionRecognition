# Thesis Defense Presentation: Modular Spatiotemporal Analysis for Micro-Expression Recognition
**Focus:** Stages 1 & 2 Ablation Study  
**Presenter:** Addhyan  
**Date:** June 2026  

---

## Slide 1: Title & Research Objective

### On-Screen Text (High School Level)
* **What are we doing?** We are testing which parts of an AI network are actually needed to read tiny, split-second facial movements (micro-expressions).
* **Our Goal:** To strip down a complex AI system, toggle different parts on and off, and find the perfect balance between speed, size, and accuracy.
* **Why it matters:** Standard AIs are too big and memorize faces instead of reading motions. We want a model that works on simple computers and generalizes to new people.

### Slide Visuals
* **Layout:** Landscape dual-panel. The left panel shows the title, author name, and department. The right panel displays a conceptual flowchart of the four toggles: Input video -> [EVM Toggle] -> [3D-CNN Toggle] -> [SimAM Toggle] -> [Transformer Toggle] -> Emotion Prediction.
* **Annotations:** Highlight each toggle as a light switch that can be flipped `ON` or `OFF`.

### Presenter Speech Notes (University/Academic Level)
* **Script:** Good afternoon, members of the committee. Welcome to the presentation of my thesis research. Today, we will examine a systematic, 12-cell ablation matrix designed to isolate and evaluate the spatiotemporal modeling blocks of a hybrid network for Micro-Expression Recognition (MER). 
* **Depth:** The core scientific objective of this ablation study is to address the twin bottlenecks of affective computing: extreme sample scarcity ($N=255$ clips in CASME II) and subject identity bias. Large deep networks with high parameter counts easily overfit to static face geometry (skin texture, bone structure) rather than capturing the transient muscle contractions of interest.
* **Math & Logic:** We define our search space across four binary architectural variables: Eulerian Video Magnification (EVM), a three-stream modality-aware 3D-CNN spatial stem, a parameter-free SimAM attention layer, and a Sequence-Level Spatio-Temporal Transformer (SLSTT). We evaluate these modules using a strict subject-disjoint validation protocol to ensure that identity leakage does not inflate our performance metrics.

---

## Slide 2: The Twin Bottlenecks of Micro-Expression Recognition

### On-Screen Text (High School Level)
* **Problem 1: The Face Stamp (Identity Bias):** AIs naturally look at a person’s overall face structure (jawline, nose, eyes) instead of their actual expressions. When tested on a new person, the AI gets confused because it has memorized the training subjects' faces.
* **Problem 2: The Tiny Photo Album (Data Scarcity):** Micro-expressions are super fast (less than half a second) and rare. Humans have to label them frame-by-frame. That means our datasets are tiny (only 255 videos in CASME II), and big models immediately overfit.
* **The Solution:** We must build a small, parameter-efficient network that strips out face structure and looks only at movement.

### Slide Visuals
* **Layout:** Split slide with two illustrative diagrams. 
  - **Left Diagram:** "Face Stamp" showing how a heavy CNN model mistakenly groups samples by subject identity (Subject A, Subject B) rather than by emotion (Negative, Positive).
  - **Right Diagram:** A histogram comparing the sample sizes of typical macro-expression datasets (e.g., CK+ or Oulu-CASIA, with thousands of samples) against CASME II (only 255 samples), illustrating "Data Starvation".

### Presenter Speech Notes (University/Academic Level)
* **Script:** To understand our ablation variables, we must define the mathematical bottlenecks of the MER domain. Micro-expressions are characterized by their rapid onset-to-offset duration (typically $T \le 500$ ms) and low spatial amplitude (often sub-pixel muscle movements).
* **Depth:** Mathematically, let a video clip be a spatiotemporal tensor $\mathbf{X} \in \mathbb{R}^{C \times T \times H \times W}$. In a standard CNN, the feature extraction kernel learns a combination of static structural variables (identity $\mathbf{z}_{id}$) and dynamic expressive variables ($\mathbf{z}_{exp}$). Because the variance of $\mathbf{z}_{id}$ in the feature space is orders of magnitude larger than $\mathbf{z}_{exp}$, the network minimizes loss by optimizing projection boundaries for subject clustering rather than expression classification.
* **Math & Logic:** Furthermore, training a model $f_\theta$ on a low-sample regime ($D \approx 250$ videos) leads to catastrophic overfitting. By moving to a shallow, modular architecture with less than 400K parameters, we drastically reduce the model's capacity to memorize static details, forcing it to specialize in temporal shifts.

---

## Slide 3: The 12-Configuration Ablation Matrix

### On-Screen Text (High School Level)
* **The Light Switch Experiment:** We test our model by turning four main technologies on and off:
  1. **EVM (Motion Amplifier):** Makes tiny muscle twitches bigger.
  2. **3D-CNN (Local Scanner):** Scans the face for motion in small areas.
  3. **SimAM (Smart Highlight):** Spotlights active facial zones without adding extra size.
  4. **Transformer (Time Tracker):** Watches how the expression changes over time.
* **The Grid:** By testing permutations of these switches, we map out how each module changes accuracy and size.

### Slide Visuals
* **Layout:** A large, clean grid table showing the 12 active configurations from our sweep.
* **Table Columns:** `Config Name`, `EVM (Magnify)`, `3D-CNN (Spatial)`, `SimAM (Attention)`, `SLSTT (Temporal)`, `Accuracy (%)`, `Macro F1 (%)`, `Trainable Parameters`.
* **Color Coding:** Highlight configurations without a 3D-CNN in red (to indicate model collapse) and proposed modular designs in green.

### Presenter Speech Notes (University/Academic Level)
* **Script:** This slide displays our 12-configuration experimental matrix. The ablation sweep is designed as a $2^4$ factorial design, though degenerate combinations (such as applying SimAM attention without a 3D-CNN feature map) are discarded.
* **Depth:** We split our analysis into four scientific phases. Phase I represents our non-magnified baselines, starting from a pure raw patch projection baseline (`config_1`) up to spatial-only (`config_3`) and temporal-only (`config_2`) models. Phase II introduces Eulerian Video Magnification and SimAM attention individually. Phase III and IV assemble these blocks into full pipelines, comparing the impact of motion amplification (`config_8` vs `config_6`) and spatial attention (`config_8` vs `config_7`).
* **Math & Logic:** In terms of parameter complexity, the models fall into two distinct classes. Configurations with the 3D-CNN backbone deactivated (`use_cnn=False`) rely on a `RawPatchEmbedding` which pools the input spatial dimensions down to a $4 \times 4$ grid, flattens them, and projects them to $d_{\text{model}} = 96$ via a single linear layer. These models contain roughly 3.3K parameters. Enabling the 3D-CNN backbone introduces a three-stream Conv3D stem, raising the trainable parameter count to approximately 363K.

---

## Slide 4: Variable A: Eulerian Video Magnification (EVM)

### On-Screen Text (High School Level)
* **The Heartbeat Zoom Lens:** Imagine looking at a brick wall and trying to see it expand as the sun heats it. It’s too small for your eyes. EVM is a software lens that amplifies sub-pixel motion.
* **How it works:**
  1. Slice each frame into different detail layers (coarse to fine).
  2. Filter the video in time to capture only the speed of muscle twitches (0.4 to 3.0 Hz).
  3. Boost this speed signal by **20 times**.
  4. Put the layers back together.
* **Result:** Subtle muscle micro-movements become visible to the AI.

### Slide Visuals
* **Layout:** A three-step horizontal pipeline diagram.
  - **Step 1:** Raw input video frame $\rightarrow$ Laplacian Pyramid (showing four levels of spatial resolution).
  - **Step 2:** FFT Temporal Bandpass Filter (graph showing frequency window $[0.4, 3.0]$ Hz and a peak corresponding to facial twitches).
  - **Step 3:** Reconstruction (a magnified frame with a zoomed inset of the eyebrow area showing amplified displacement vectors).

### Presenter Speech Notes (University/Academic Level)
* **Script:** Variable A represents our offline data-level motion amplification module, implemented via Eulerian Video Magnification, or EVM, following Wu et al. (2012). 
* **Depth:** The mathematical pipeline operates as follows. We decompose each raw video frame $I(x,y,t)$ spatially using a 4-level Laplacian pyramid to isolate distinct spatial frequency bands. For each spatial scale $S_l(x,y,t)$ at level $l$, we apply a temporal bandpass filter implemented via Fast Fourier Transform (FFT) over the time dimension:
  $$B_l(x,y,t) = \mathcal{F}^{-1} \left( H(\omega) \cdot \mathcal{F}(S_l(x,y,t)) \right)$$
  where the temporal bandpass transfer function $H(\omega)$ is parameterized by lower and upper cutoffs $f_{\text{low}} = 0.4$ Hz and $f_{\text{high}} = 3.0$ Hz.
* **Math & Logic:** This frequency window is specifically calibrated to exclude low-frequency head drift and high-frequency camera noise. The bandpass-filtered signal is then scaled by the amplification factor $\alpha = 20.0$, and added back to the original pyramid:
  $$\tilde{S}_l(x,y,t) = S_l(x,y,t) + \alpha \cdot B_l(x,y,t)$$
  Collapsing the pyramid yields the reconstructed motion-magnified sequence $\tilde{I}(x,y,t)$, presenting a significantly enhanced signal-to-noise ratio for downstream spatial kernels.

---

## Slide 5: Physics-Based Modality Extraction & Interpolation

### On-Screen Text (High School Level)
* **Moving Beyond Simple Colors:** Raw color pixels contain too much useless information like skin tone or lighting. Instead, we extract three movement channels:
  1. **Horizontal Flow (Left-Right):** Tracks side-to-side muscle pulls.
  2. **Vertical Flow (Up-Down):** Tracks up-and-down movements (like raised eyebrows).
  3. **Skin Strain:** Measures local skin compression, stretching, and shearing.
* **Standard Time:** Videos are stretched or squeezed to be exactly **32 frame pairs** long.

### Slide Visuals
* **Layout:** Grid of four images.
  - **Image 1:** Cropped face template.
  - **Image 2:** Horizontal optical flow $u$ map (colored using red/blue mapping to show direction).
  - **Image 3:** Vertical optical flow $v$ map (yellow/blue mapping).
  - **Image 4:** Optical strain map (heat map showing active skin compression around the eye and mouth regions).
* **Formula Overlay:** Keep the optical strain formulation on the slide.

### Presenter Speech Notes (University/Academic Level)
* **Script:** Rather than feeding raw RGB pixel tensors to our model, we compute a physics-based modality-aware representation to strip out static appearance cues.
* **Depth:** For each consecutive pair of motion-magnified frames, we compute the dense optical flow field $\mathbf{v}(x,y,t) = [u(x,y,t), v(x,y,t)]^T$ using Farneback's polynomial expansion algorithm (Farneback, 2003). From this flow field, we compute the symmetric optical strain tensor to capture local tissue deformations:
  $$\epsilon_{xx} = \frac{\partial u}{\partial x}, \quad \epsilon_{yy} = \frac{\partial v}{\partial y}, \quad \epsilon_{xy} = \frac{1}{2} \left( \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} \right)$$
  The optical skin strain magnitude is computed as:
  $$os(x,y,t) = \sqrt{\epsilon_{xx}^2 + \epsilon_{yy}^2 + 2\epsilon_{xy}^2}$$
* **Math & Logic:** This formulation isolates the shear and normal deformations of the skin tissue, which are highly correlated with underlying Facial Action Units. We pack $u$, $v$, and $os$ into a 3-channel spatiotemporal tensor. Since raw clips have variable lengths (10 to 150 frames), we apply a `TemporalInterpolator` which standardizes the sequence to exactly $L=33$ frames (producing $T=32$ flow/strain maps) via linear index sampling. The input size is thus standardized to $[B, 3, 32, 224, 224]$.

---

## Slide 6: Variable C: Three-Stream Shallow 3D-CNN Backbone

### On-Screen Text (High School Level)
* **Unshared Streams (Specialized Scanners):** Instead of processing horizontal movements, vertical movements, and skin strain with the same filters, we build three independent spatial scanners.
* **Why unshared?** Because horizontal muscle slides follow completely different physics than vertical brow rises.
* **Shallow for Safety:** We use only two convolution layers per branch. This keeps the network small and prevents it from overfitting.
* **Preserving Time:** The convolutions only scan spatially, leaving the 32 time steps untouched.

### Slide Visuals
* **Layout:** Detailed block diagram of the `ThreeStreamCNNBackbone`.
  - Input: $[B, 3, 32, 224, 224] \rightarrow$ split into Channel 0 (horizontal), Channel 1 (vertical), and Channel 2 (strain).
  - Each channel routes into a separate column: `Conv3D (1->16, k=[1,3,3]) -> BN -> ReLU -> Dropout -> Conv3D (16->32, k=[1,3,3]) -> BN -> ReLU -> MaxPool3D (k=[1,2,2])`.
  - The outputs ($3 \times [B, 32, 32, 112, 112]$) are concatenated along the channel dimension to yield $[B, 96, 32, 112, 112]$.

### Presenter Speech Notes (University/Academic Level)
* **Script:** Variable C dictates the deactivation or activation of our 3D-CNN spatial feature stem. When activated, we employ a weight-unshared three-stream network inspired by STSTNet (Liong et al., 2019).
* **Depth:** Since horizontal flow, vertical flow, and skin strain capture fundamentally different physical phenomena, forcing shared kernels to process all channels reduces feature resolution. Our design processes each channel through an independent backbone stream:
  $$\mathbf{H}_u = \text{Stream}_u(\mathbf{X}_u), \quad \mathbf{H}_v = \text{Stream}_v(\mathbf{X}_v), \quad \mathbf{H}_{os} = \text{Stream}_{os}(\mathbf{X}_{os})$$
  where each stream consists of two consecutive volumetric convolutions with kernels of size $1 \times 3 \times 3$ and stride $1 \times 1 \times 1$. 
* **Math & Logic:** Setting the temporal kernel depth to $1$ is critical: it prevents temporal mixing at the spatial stage, allowing spatial features to be extracted independently for each time frame while delegating sequence modeling entirely to the downstream Transformer. A spatial-only MaxPool3D of size $1 \times 2 \times 2$ halves the spatial resolution ($224 \times 224 \rightarrow 112 \times 112$) while maintaining the temporal sequence length at 32. The concatenated output tensor shape is $[B, 96, 32, 112, 112]$.

---

## Slide 7: Variable B: SimAM 3D Parameter-Free Attention

### On-Screen Text (High School Level)
* **Smart Focusing (Without the Weight):** Normal attention layers require a lot of math and make the model heavy, causing it to overfit.
* **SimAM is different:** It has **zero learnable parameters**. It uses a mathematical formula based on brain science (how active neurons suppress quiet neighbors) to calculate a "brightness" score for every single pixel.
* **Gating:** It multiplies the features by this score, automatically highlighting active parts of the face (like the mouth or eyebrows) and ignoring the static background.

### Slide Visuals
* **Layout:** A diagram showing the attention mapping.
  - Left: A noisy spatial feature map with active muscle twitches alongside head sway.
  - Middle: The SimAM energy grid (visualized as a high-contrast mask showing bright white spots only on the eyebrows and mouth edges).
  - Right: The resulting rescaled feature map where the background is dimmed and the eyebrows/mouth are bright.
* **Equation Box:** $e_t = \frac{(t-\mu)^2}{4(\sigma^2 + \lambda)} + 0.5$

### Presenter Speech Notes (University/Academic Level)
* **Script:** Variable B controls the integration of SimAM, a neuroscience-inspired parameter-free attention mechanism (Yang et al., 2021), adapted here for 3D spatiotemporal tensors.
* **Depth:** SimAM estimates the linear separability of a target neuron from its spatio-temporal neighbors. For a given channel, the energy of a neuron $t$ in a spatiotemporal neighborhood of size $M = D \times H \times W - 1$ is calculated as:
  $$e_t(t, x_i) = \frac{1}{M} \sum_{i=1}^M \left( y_o - (w_t x_i + b_t) \right)^2 + \left( y_t - (w_t t + b_t) \right)^2$$
  By defining binary targets $y_t = 1$ and $y_o = -1$ and solving this objective analytically, we get:
  $$e_t = \frac{(t - \mu)^2}{4(\sigma^2 + \lambda)} + 0.5$$
  where $\mu = \frac{1}{M}\sum_{i=1}^M x_i$ and $\sigma^2 = \frac{1}{M}\sum_{i=1}^M (x_i - \mu)^2$ are the mean and variance of all neurons in that channel, and $\lambda = 10^{-4}$ is a regularization constant preventing division by zero.
* **Math & Logic:** The attention weight is computed as $\text{sigmoid}(1/e_t)$, which inversely scales with the energy $e_t$. The feature maps are scaled element-wise: $\tilde{\mathbf{X}} = \mathbf{X} \odot \text{sigmoid}(1/\mathbf{E})$. Because this calculation relies entirely on local neighborhood statistics and introduces no trainable weights, it focuses the model on high-frequency muscle deformations without exacerbating the overfitting risk on small datasets.

---

## Slide 8: Variable D: Sequence-Level Spatio-Temporal Transformer (SLSTT)

### On-Screen Text (High School Level)
* **Tracking the Story over Time:** Micro-expressions have a clear sequence: they start (onset), peak (apex), and fade (offset).
* **Positional Stamps:** Since transformers process all time frames at once, we add mathematical stamps (sinusoidal curves) to each frame so the model knows the chronological order.
* **Multi-Head Self-Attention:** Allows the model to look at different parts of the face and compare the muscle states of early frames with later frames.
* **Output:** We take the average of all 32 frames to get a single summary vector for classification.

### Slide Visuals
* **Layout:** Diagram showing the flow of tokens.
  - Spatial feature maps compressed via Adaptive Pooling to $[B, 32, 96]$.
  - Add block: Sinusoidal Positional Encoding (wave graphics).
  - SLSTT Encoder (2 layers, each showing Multi-Head Attention followed by FFN).
  - Temporal Mean Pooling block collapsing the 32 tokens into a final $1 \times 96$ vector.

### Presenter Speech Notes (University/Academic Level)
* **Script:** Variable D represents our temporal sequence model, implemented via a 2-layer pre-norm Sequence-Level Spatio-Temporal Transformer (SLSTT) inspired by Zhang et al. (2022).
* **Depth:** First, spatial dimensions are collapsed via adaptive average pooling: $[B, 96, 32, 112, 112] \rightarrow [B, 96, 32, 1, 1]$. Reshaping yields a sequence of 32 temporal tokens: $\mathbf{Z}_0 \in \mathbb{R}^{B \times 32 \times 96}$. To inject temporal order into the permutation-invariant self-attention block, we add deterministic sinusoidal positional encodings (Vaswani et al., 2017):
  $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
  where $d_{\text{model}} = 96$ and $pos \in [0, 31]$.
* **Math & Logic:** The sequence is processed through 2 layers of multi-head self-attention with 4 heads. The attention mechanism projects queries, keys, and values into subspaces of dimension $d_k = 24$, computing:
  $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$
  This captures long-range temporal dynamics, tracking the progression from neutral to apex states. The time dimension is collapsed using temporal mean pooling to yield a global feature representation $\mathbf{z}_{glob} \in \mathbb{R}^{96}$, which is passed to the final classification head.

---

## Slide 9: The Decoupled Optimization Objective

### On-Screen Text (High School Level)
* **Balancing the Scales (Weighted Focal Loss):** Our dataset is highly imbalanced: we have a lot of Negative expressions, but very few Positive or Surprise expressions.
* **How we solve it:**
  1. We use **Focal Loss** which automatically ignores "easy" samples and penalizes the model heavily when it gets "hard" samples wrong.
  2. We apply **inverse-frequency class weights** so that predicting a rare Surprise expression correctly is worth much more than predicting a common Negative expression.
  3. We add **Label Smoothing** (5%) to prevent the model from becoming overconfident.

### Slide Visuals
* **Layout:** Side-by-side panel.
  - **Left Panel:** Formula for Focal Loss in a clear text box.
  - **Right Panel:** Bar chart illustrating the class weights applied to the three classes: Negative (weight = 0.59), Positive (weight = 1.70), and Surprise (weight = 2.25). Shows how Surprise is scaled to have a stronger gradient contribution.

### Presenter Speech Notes (University/Academic Level)
* **Script:** During the ablation sweep, the models are trained using a decoupled classification objective to isolate the architectural contributions of Stage 1 and Stage 2 without the influence of auxiliary domain-adversarial or contrastive losses.
* **Depth:** To counter the severe class imbalance in CASME II, we implement a class-weighted Focal Loss (Lin et al., 2017) with label smoothing. The focal loss for a sample with ground-truth class probability $p_t$ is defined as:
  $$\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
  where the focusing parameter $\gamma$ is set to $2.0$ to dynamically down-weight the loss contribution of easily classified majority samples.
* **Math & Logic:** The scaling parameter $\alpha_t$ is computed using the inverse frequency of the training class distribution:
  $$\alpha_t = \frac{N_{\text{total}}}{\text{num\_classes} \cdot N_t}$$
  yielding weights of approximately 0.59 for Negative, 1.70 for Positive, and 2.25 for Surprise. Furthermore, we apply a label smoothing factor $\epsilon = 0.05$ to distribute target probabilities to incorrect classes, regularizing the classification head and preventing the model from producing overconfident logit boundaries on highly subjective micro-expression frames.

---

## Slide 10: Quantitative Ablation Performance Matrix

### On-Screen Text (High School Level)
* **What did we learn?** The numbers reveal a clear story:
  - Models **without** a 3D-CNN completely fail. They get 74.36% accuracy, but only because they predict the most common class ("Negative") for every single video.
  - Turning on the **3D-CNN** immediately solves this, raising the Macro F1 score to **43.57%**.
  - Interestingly, adding the Transformer and SimAM together on this small dataset causes accuracy to drop. Why? Because the model has too much learning capacity and overfits to the small validation set.

### Slide Visuals
* **Layout:** A structured bar chart comparing the accuracy and Macro F1 of the key configurations:
  - `config_1` (Pure Base): 74.36% Acc / 28.43% F1
  - `config_3` (Spatial Only): 71.79% Acc / 43.57% F1
  - `config_7` (Proposed without SimAM): 69.23% Acc / 35.77% F1
  - `config_8` (Proposed Unified): 48.72% Acc / 30.94% F1
* **Visual Annotations:** Use brackets to highlight the delta in Macro F1 (+15.14%) when moving from pure base to spatial-only, and the drop in Macro F1 (-12.63%) when adding the Transformer + SimAM without regularization.

### Presenter Speech Notes (University/Academic Level)
* **Script:** Let us analyze the quantitative results of our 12-configuration sweep. We evaluate models using a subject-disjoint validation split of 39 samples. 
* **Depth:** The pure baseline configuration (`config_1`) achieves an accuracy of 74.36% but a Macro F1 score of only 28.43%. Looking at the per-class metrics, we find that the Negative F1 is 85.29%, while Positive and Surprise F1 are exactly 0.0%. This indicates that the model has collapsed to the majority class, predicting Negative for every validation sample.
* **Math & Logic:** Enabling the 3D-CNN Backbone (`config_3`) drops raw accuracy slightly to 71.79% but increases the Macro F1 to 43.57% (Negative F1 = 80.70%, Positive F1 = 50.00%). This represents a substantial +15.14% improvement in Macro F1, proving that the spatial backbone is the core feature extractor. However, when we enable the full proposed pipeline (`config_8` with EVM, SimAM, CNN, and Transformer), the accuracy drops to 48.72% and Macro F1 falls to 30.94%. This drop indicates that high-capacity spatiotemporal modeling blocks overfit severely in a standard supervised training regime on small datasets, highlighting the necessity of representation constraints.

---

## Slide 11: Deconstructing Model Collapse (The Spatial Stem)

### On-Screen Text (High School Level)
* **The Danger of Flat Patches (No CNN):** 
  - When the 3D-CNN is turned off, the model splits the raw frames into a flat grid of patches.
  - This flat grid is full of noise (lighting shifts, skin texture).
  - Because the dataset is tiny, the model cannot learn spatial rules from this noise. It gives up and predicts the majority class ("Negative") for everything.
* **The CNN Shield:** Convolutions act as a spatial filter, extracting localized movements and shielding the model from full-frame noise.

### Slide Visuals
* **Layout:** Comparative diagram.
  - **Left Side:** "Flat Patch Path" showing a raw frame divided into 16 square patches. A red arrow leads to a feature vector that clusters all subjects together, resulting in majority class collapse.
  - **Right Side:** "3D-CNN Path" showing the three unshared streams extracting local edges and motion boundaries. A green arrow leads to a feature vector that separates Positive from Negative emotions.

### Presenter Speech Notes (University/Academic Level)
* **Script:** A key finding of this ablation study is the catastrophic collapse of all configurations where the 3D-CNN backbone is deactivated.
* **Depth:** Specifically, `config_1` (base), `config_2` (temporal only), `config_4` (EVM base), and `config_12` (EVM + Transformer) all yield identical metrics: 74.36% Accuracy and 28.43% Macro F1. When the 3D-CNN is deactivated, the model uses `RawPatchEmbedding`, average-pooling the raw motion tensors to a $4 \times 4$ spatial grid. This collapses the spatial resolution from $224 \times 224$ to $4 \times 4$ before linear projection.
* **Math & Logic:** Without the translation-invariant local receptive fields of convolutional kernels, the raw flattened patches preserve static noise and alignment offsets. The linear projection layer $W \in \mathbb{R}^{d_{\text{model}} \times 48}$ is forced to learn spatial representations from scratch on only 215 training samples. Due to this extreme data scarcity, the projection layer fails to align feature spaces, causing the gradients to backpropagate purely into the classification bias of the dominant class, resulting in model collapse. This confirms that a deep spatial stem is an absolute prerequisite for feature learning in small-sample MER.

---

## Slide 12: SimAM & Transformer Overfitting Dynamics

### On-Screen Text (High School Level)
* **Too Much Power (Overfitting):**
  - Our full model (`config_8`) has a 3D-CNN, SimAM attention, and a Sequence Transformer.
  - This combination is extremely powerful at tracking both spatial zones and temporal changes.
  - However, because the dataset is tiny, this power backfires. The model uses its extra capacity to memorize the exact faces and motions of the training subjects.
  - As a result, it performs worse on unseen validation subjects than a simple spatial-only CNN.

### Slide Visuals
* **Layout:** Dual plot panel.
  - **Plot A:** Training and Validation loss curves for `config_3` (Spatial Only). Show a slow, stable decrease in both training and validation losses, converging close together.
  - **Plot B:** Training and Validation loss curves for `config_8` (Proposed Unified). Show training loss dropping rapidly to near-zero, while validation loss diverges upward after epoch 15, illustrating severe overfitting.

### Presenter Speech Notes (University/Academic Level)
* **Script:** Let us analyze the spatiotemporal interaction dynamics between the attention block and the sequence transformer.
* **Depth:** In the spatial-only baseline (`config_3`), we achieve a Macro F1 of 43.57%. When we add the SLSTT Transformer (`config_7` / `config_9`), the Macro F1 drops to 35.77%. When we enable both the Transformer and SimAM attention (`config_6` / `config_8`), the Macro F1 drops further to 30.94%. This represents a progressive performance degradation as model complexity increases.
* **Math & Logic:** In a standard supervised regime optimized with Focal Loss, the self-attention heads of the Transformer are highly flexible, modeling global temporal correlation matrices $\mathbf{A} = \text{softmax}(\mathbf{Q}\mathbf{K}^T/\sqrt{d_k})$. Without regularizing constraints, the heads specialize in tracking static subject-specific kinematics (e.g., individual-specific facial twitches or head sways) present in the training set. Similarly, SimAM attention rescales feature maps to highlight high-variance zones, which frequently capture subject-specific wrinkles or static features. This empirical result proves that high-capacity spatiotemporal architectures *require* identity-invariant representation learning (such as domain adaptation or contrastive training) to generalize to unseen subjects.

---

## Slide 13: The Role of Eulerian Magnification in Ablation

### On-Screen Text (High School Level)
* **Does Boosting Motion Help?**
  - In our offline tests, enabling EVM (`config_8` vs `config_6` or `config_13` vs `config_3`) resulted in identical validation scores.
  - Why? Because EVM is a linear process that amplifies both the micro-expression twitches and the camera noise.
  - Without special representation learning, standard models struggle to separate the boosted expression signal from the boosted noise.

### Slide Visuals
* **Layout:** Side-by-side comparison.
  - Left panel: A signal diagram showing a small micro-expression wave nested inside low-level sensor noise.
  - Right panel: The amplified signal after EVM. Show how both the micro-expression wave and the noise spikes are scaled upward by 20x, maintaining the same signal-to-noise ratio.

### Presenter Speech Notes (University/Academic Level)
* **Script:** An interesting finding in our ablation sweep is that the paired configurations with and without EVM—such as `config_8` and `config_6`, or `config_13` and `config_3`—yield identical accuracy and F1 scores under the holdout validation split.
* **Depth:** Let us analyze the mathematics of this behavior. Eulerian Video Magnification is a localized linear operator. While it amplifies sub-pixel displacements by a factor of $\alpha$, it also scales high-frequency camera noise and illumination fluctuations:
  $$\tilde{\mathbf{X}} = \mathbf{X} + \alpha \cdot \mathbf{X}_{\text{filtered}}$$
  In a standard supervised training setup using a standard classification head, the spatial convolution kernels learn weights that adapt to the absolute input magnitudes. 
* **Math & Logic:** Because the relative signal-to-noise ratio (SNR) remains unchanged, the model simply scales its internal weights to compensate, arriving at the same optimization minima. This demonstrates that data-level motion amplification alone is not a silver bullet. To fully leverage EVM, the network must be combined with contrastive objectives that pull together different amplification views of the same sample, or domain-adversarial constraints that prevent the network from latching onto magnified camera artifacts.

---

## Slide 14: Technical Takeaways & Edge Recommendations

### On-Screen Text (High School Level)
* **Key Lessons for Real-World AI:**
  1. **CNN is the Core:** You cannot skip the 3D-CNN backbone. Without it, the model collapses completely.
  2. **Shallow is Better:** Keep spatial networks shallow (2 layers) to prevent overfitting on tiny datasets.
  3. **Control the capacity:** If you use a Transformer or Attention mechanism, you must add constraints to prevent the model from memorizing subject identities.
  4. **Efficient Size:** The full 3D-CNN backbone and Transformer use only **363K parameters**, making it perfect for real-time applications on small devices.

### Slide Visuals
* **Layout:** A structured, high-contrast matrix summary.
  - Left panel: A hierarchy list of component importance (1. CNN Backbone, 2. Spatial Pooling, 3. Temporal Pooling, 4. Attention/Transformer).
  - Right panel: A table showing Parameter Count and estimated floating-point operations (FLOPs) for the spatial-only vs. full spatiotemporal model, proving extreme efficiency.

### Presenter Speech Notes (University/Academic Level)
* **Script:** In conclusion, our 12-configuration ablation study provides critical architectural guidelines for the design of spatiotemporal networks in low-data regimes.
* **Depth:** First, the ThreeStream3DCNN backbone is the indispensable foundation, reducing feature noise and providing spatial translation invariance. Models relying on raw patch projection are mathematically unable to converge. Second, while attention mechanisms and transformers are theoretically superior for temporal modeling, their high capacity leads to severe overfitting under standard cross-entropy or focal losses.
* **Math & Logic:** For hardware-constrained edge deployment, the spatial-only convolutional configuration (`config_3`) provides the most robust baseline, achieving a Macro F1 of 43.57% with an extremely small computational footprint. When deploying the full spatiotemporal model (`config_8`), which has 363,280 parameters and requires approximately $1.6 \times 10^9$ FLOPs per inference, it is scientifically imperative to incorporate auxiliary representation constraints (such as contrastive embeddings or adversarial training) to regularize the self-attention heads and prevent identity bias from degrading generalization on unseen subjects. Thank you, and I am open to your questions.
