<!-- markdownlint-disable MD024 MD025 MD060 -->

# Thesis FAQ — Micro-Expression Recognition

**Thesis:** Improving Micro-Expression Recognition through Temporal and Motion Modelling  
**Use this document for:** viva preparation, supervisor questions, defense rehearsal, and explaining what was implemented versus what came from the literature.

---

## 1. Core Thesis Questions

### Q1. What is the thesis in one sentence?

This thesis builds a **video-based, subject-invariant micro-expression recognition pipeline** that converts face clips into motion tensors using EVM, optical flow, and optical strain, then classifies emotions with a compact STSTNet-SimAM-Transformer model trained with GRL, SupCon, XBM, and Focal Loss.

### Q2. What exactly did you implement?

I implemented a four-stage pipeline:

1. **Stage 1 — Data pipeline:** CASME II and CAS(ME)^2 metadata unification, EVM motion magnification, temporal sampling to 33 frames, Farneback optical flow, optical strain, and `.npy` tensor saving.
2. **Stage 2 — Architecture:** three-stream STSTNet-style 3D-CNN, 3D SimAM attention, adaptive pooling, SLSTT-style Transformer, and a compact classifier.
3. **Stage 3 — Subject-invariant training:** triple-head wrapper with emotion head, projection head, and identity head behind GRL; trained with Focal Loss, SupCon, XBM, and LOSO.
4. **Stage 4 — Transfer learning:** macro-expression pre-training followed by micro-expression fine-tuning.

### Q3. What is the main novelty?

No single paper in the reviewed set combines **EVM + 32-step u/v/strain motion tensors + STSTNet-style modality streams + SimAM + Transformer temporal modelling + GRL subject-invariance + SupCon/XBM + Focal Loss + macro-to-micro transfer** in one MER pipeline.

The novelty is the **integration and thesis-specific training design**, not claiming that each individual block was invented from scratch.

### Q4. What problem are you solving?

The thesis targets two major MER problems:

- **Data scarcity:** public spontaneous MER datasets are tiny, often only a few hundred samples.
- **Subject identity bias:** models can memorize who the subject is instead of learning subtle expression motion.

### Q5. Why is MER difficult?

Micro-expressions are brief, low-intensity, localized, and often appear for less than 500 ms. Their motion can be sub-pixel, and datasets require expert frame-level annotation. This makes deep models prone to overfitting and identity memorization.

---

## 2. Micro vs Macro Expressions

### Q6. Are macro-expressions always voluntary?

No. This was an important professor correction. Macro-expressions can also be spontaneous or involuntary. The cleaner distinction is:

- **Macro-expression:** longer, larger, easier to see, often spread across more of the face.
- **Micro-expression:** shorter, weaker, more localized, often linked to concealment or suppression.

### Q7. Why use macro-expressions if the thesis is about micro-expressions?

Macro and micro expressions often use related **facial action units**. Macro motion is larger and easier to learn, so Stage 4 uses macro-expression clips as a stronger initial signal before adapting to micro-expressions.

This is supported by **Xia et al., "Learning from Macro-expression"**, which argues that macro-expression samples can guide micro-expression recognition because of shared facial muscle movements and texture changes.

### Q8. Does Stage 4 mean macro and micro are the same?

No. Stage 4 assumes they share some facial motion structure, not that they are identical. Micro-expressions still have lower intensity, shorter duration, and weaker temporal evidence.

---

## 3. Datasets

### Q9. Which datasets did you use?

The implemented thesis uses:

- **CASME II:** high-speed micro-expression dataset, 200 fps, 26 subjects.
- **CAS(ME)^2:** 30 fps dataset containing both macro- and micro-expressions.

The pipeline unifies them into a single metadata schema and maps labels to four emotion classes.

### Q10. Why CASME II?

CASME II is a standard spontaneous MER benchmark with high-speed 200 fps capture, FACS-based annotation, onset/apex/offset frames, and 26 subjects. It is widely used in STSTNet, OFF-ApexNet, SLSTT, and many survey benchmarks.

### Q11. Why CAS(ME)^2?

CAS(ME)^2 contains both macro- and micro-expressions from the same general data source, making it useful for Stage 4 macro-to-micro transfer learning. It also exposes a hard limitation: many clips are only 30 fps, so temporal motion evidence is weaker than in high-speed datasets.

### Q12. What labels did you use?

The final thesis taxonomy is:

- **Positive:** happiness
- **Negative:** disgust, sadness, fear, anger, repression, contempt-like negative labels
- **Surprise:** surprise
- **Others:** others, helpless, pain, confused, sympathy, and ambiguous labels

### Q13. Why collapse labels?

MER datasets are too small for fine-grained emotion classification. Many original labels have only a few examples. Collapsing labels follows the logic used in MEGC composite evaluation: reduce sparse classes into broader, more learnable groups.

### Q14. Why not use SMIC and SAMM too?

SMIC and SAMM are important literature datasets, but the implemented pipeline focused on CASME II and CAS(ME)^2. SMIC/SAMM remain natural future extensions, especially for comparing against MEGC 2019 composite evaluation.

### Q15. What is DFME and why did the professor mention it?

The professor’s "Douglas dataset" note likely refers to **DFME: Dynamic Facial Micro-Expressions**. DFME is much larger than CASME II:

- **DFME:** 7,526 ME videos, 656+ participants, 200/300/500 fps.
- **CASME II:** 247 ME samples, 26 participants, 200 fps.

DFME is a strong future validation target because it reduces the small-data bottleneck and better supports deep learning.

---

## 4. Video vs Photos

### Q16. Are you using videos or photos?

The implemented thesis uses **videos**. Each clip is converted into a 32-step motion tensor of shape `[3, 32, 224, 224]`.

### Q17. Why do some papers use only onset and apex photos?

Papers such as **Less is More**, **OFF-ApexNet**, **STSTNet**, and **joint local/global apex-frame learning** show that the onset-apex pair is highly informative because the apex frame contains peak expression intensity.

### Q18. Why did you not only use apex frames?

Apex methods reduce redundancy but discard temporal evolution. My thesis keeps a 32-step video tensor so the Transformer can model the full motion path: **onset -> apex -> offset**.

### Q19. What is the advantage of video tensors?

Video tensors preserve motion order. A micro-expression is not only a peak image; it is a short dynamic event. The model can learn whether a movement rises, peaks, fades, or changes direction.

---

## 5. Stage 1 — Preprocessing

### Q20. What does Stage 1 output?

Stage 1 outputs one `.npy` tensor per clip:

```text
[3, 32, 224, 224]
```

The channels are:

1. horizontal optical flow `u`
2. vertical optical flow `v`
3. optical strain `os`

### Q21. Why not feed raw RGB?

Raw RGB contains identity cues: skin tone, face shape, lighting, background, and texture. Motion tensors reduce this identity bias by focusing on displacement and deformation.

### Q22. What is EVM?

**Eulerian Video Magnification** amplifies subtle motion by treating pixel intensity changes over time as signals. The implemented EVM uses Laplacian pyramids, temporal bandpass filtering, and amplification.

### Q23. Why use EVM?

Micro-expression motion can be smaller than a pixel. EVM makes subtle facial motion easier for optical flow and the neural network to detect. This is supported by Wu et al. and later MER work such as Bai et al. and Li et al.

### Q24. What EVM parameters did you use?

The guide documents:

- amplification `alpha = 20.0`
- low cutoff `0.4 Hz`
- high cutoff `3.0 Hz`
- pyramid levels `4`

These settings aim to amplify facial muscle motion while suppressing slow head sway and fast noise.

### Q25. What is optical flow?

Optical flow estimates a motion vector at each pixel between two frames:

- `u`: horizontal movement
- `v`: vertical movement

The implementation uses **Farneback dense optical flow**.

### Q26. What is optical strain?

Optical strain measures local deformation from spatial gradients of optical flow:

```text
eps_xx = du/dx
eps_yy = dv/dy
eps_xy = 0.5 * (du/dy + dv/dx)
os = sqrt(eps_xx^2 + eps_yy^2 + eps_xy^2)
```

It captures how much the skin stretches or compresses, not just where it moves.

### Q27. Which papers justify optical strain?

Key papers:

- **Shreve et al.** used spatio-temporal strain for spotting macro/micro expressions.
- **Liong et al.** used optical strain for subtle emotion recognition.
- **Subtle Expression Recognition using Optical Strain Weighted Features** used strain as region weights.
- **STSTNet** used optical strain as one of three input streams.

### Q28. What is the edge-detection analogy?

Like precomputing Canny edges before image classification, I precompute optical flow and optical strain before training. This trades disk space for training time:

- more storage for `.npy` tensors
- less repeated optical flow computation during every epoch

### Q29. Why save tensors to disk?

Optical flow and strain are expensive. Saving tensors makes training repeatable and faster. It also separates preprocessing errors from model-training errors.

---

## 6. Temporal Interpolation and FPS

### Q30. Why do you sample to 33 frames?

The model needs a fixed tensor shape. Sampling 33 frames produces 32 consecutive frame pairs for optical flow, matching the `[3, 32, 224, 224]` input format.

### Q31. Is temporal interpolation a weakness?

Yes. It standardizes length but cannot create missing motion information.

- Downsampling high-FPS clips can drop subtle frames.
- Upsampling short or low-FPS clips can duplicate frames.
- Duplicate frames reduce optical flow magnitude.

### Q32. Why is 10 fps a problem?

At 10 fps, one frame is 100 ms. A 100-200 ms micro-expression may produce only 1-2 useful frames. That is not enough for reliable onset/apex/offset motion modelling.

### Q33. Why is 30 fps still a concern?

At 30 fps, one frame is about 33 ms. A 0.13-0.33 s micro-expression may contain only 4-10 frames. CAS(ME)^2 is useful, but its 30 fps clips are weaker than CASME II or DFME high-speed recordings.

### Q34. How could temporal interpolation be improved?

Future work:

- apex-aware sampling
- onset-apex-offset weighted sampling
- keep more high-FPS frames near apex
- avoid duplicating very short clips
- train variable-length models or temporal masking

---

## 7. Stage 2 — Architecture

### Q35. What is the Stage 2 model?

The model is a compact **STSTNet-SimAM-SLSTT hybrid**:

```text
[B,3,32,224,224]
 -> three STSTNet-style 3D-CNN streams
 -> SimAM attention
 -> [B,96,32,112,112]
 -> adaptive spatial pooling
 -> [B,32,96]
 -> Transformer
 -> [B,96]
```

### Q36. Why three streams?

Horizontal flow, vertical flow, and optical strain represent different physical signals. Separate streams let the model learn modality-specific filters.

### Q37. What did you take from STSTNet?

From STSTNet:

- shallow model philosophy
- three-stream u/v/strain representation
- optical strain as a distinct motion modality
- small-data awareness

What I changed:

- STSTNet uses onset-apex maps; my code uses 32 consecutive motion pairs.
- STSTNet uses small 28x28 input; my tensors are 224x224.
- STSTNet uses a shallow CNN classifier; my code adds SimAM and a Transformer.

### Q38. Why Conv3D kernel `(1,3,3)`?

The kernel looks spatially but not temporally. It extracts local face-region motion at each time step while preserving all 32 time steps for the Transformer.

### Q39. What is SimAM?

SimAM is a parameter-free attention module. It computes how different each neuron is from its neighborhood and uses that as an attention score. It adds no trainable weights, which is useful for small MER datasets.

### Q40. Why use a Transformer?

The Transformer models temporal relations between all 32 time steps. This helps capture onset-apex-offset dependencies and longer-range motion patterns.

### Q41. Is your Transformer a full ViT?

No. ViT splits images into patch tokens and usually needs massive pretraining. My model uses CNN-produced temporal tokens `[B,32,96]`, so it is much smaller and more suitable for limited MER data.

### Q42. How many layers and heads are used?

The current implementation uses:

- `d_model = 96`
- `num_layers = 4`
- `nhead = 8`
- `dim_feedforward = 256`
- dropout `0.1`

This matches `SLSTTTransformer` and `HybridMERModel`.

### Q43. How many parameters are in the model?

The Stage 3 wrapped model has about **382,094 trainable parameters**, which is intentionally small for a 598-clip dataset.

---

## 8. Stage 3 — Training

### Q44. Is Stage 3 adversarial training?

Not in the GAN or adversarial-example sense. The precise term is **GRL-based subject-invariant representation learning**.

### Q45. What is GRL?

The Gradient Reversal Layer passes features unchanged during the forward pass, but multiplies identity gradients by `-lambda` during backpropagation. The identity head tries to recognize the person; the backbone learns to remove identity cues.

### Q46. Why do you treat people as domains?

In domain-adversarial learning, the model learns features that do not reveal domain identity. Here, each subject identity is treated like a domain. The goal is to classify emotion without encoding who the subject is.

### Q47. What are the three heads in Stage 3?

1. **Emotion head:** predicts four emotion classes.
2. **Projection head:** creates 64-d embeddings for SupCon.
3. **Identity head:** predicts subject identity through GRL.

### Q48. What is SupCon?

Supervised Contrastive Loss pulls embeddings of the same emotion together and pushes different emotions apart. It helps create emotion clusters in the latent space.

### Q49. Why use XBM?

The RTX 4060 GPU only allowed a physical batch size of 2. SupCon needs many positives and negatives, so Cross-Batch Memory stores 64 previous embeddings and increases the effective comparison set.

### Q50. Why use Focal Loss?

The dataset is class-imbalanced. Focal Loss down-weights easy majority examples and focuses learning on hard or rare examples such as Positive and Surprise.

### Q51. What is the Stage 3 total loss?

The guide documents:

```text
L_total = 1.0 * L_SupCon + L_Focal(emotion) + 0.5 * L_Identity
```

### Q52. Why LOSO?

Leave-One-Subject-Out is the strict standard in MER because it tests on a person never seen during training. This directly tests subject generalization and identity bias.

### Q53. What were the Stage 3 results?

Stage 3 achieved:

- grand-mean LOSO accuracy: **58.11%**
- macro-F1: **33.31%**
- random baseline: 25%
- best fold: 85.71%
- worst fold: 35.00%

---

## 9. Stage 4 — Transfer Learning

### Q54. What is Stage 4?

Stage 4 pre-trains the model on macro-expression clips, then fine-tunes on micro-expression clips.

### Q55. Why turn the identity objective off during macro pre-training?

During macro pre-training, the goal is to learn general facial motion. The identity-invariance constraint is applied during micro fine-tuning, where generalization to unseen subjects is more important.

### Q56. What weights transferred?

The guide reports **101 of 103 parameter tensors** transferred. The identity head is re-initialized because the number of subjects changes.

### Q57. What were Stage 4 results?

Reported results:

- macro pre-training best validation accuracy: **70.37%**
- micro fine-tuning best validation accuracy: **37.78%**
- fine-tuning macro-F1: **18.41%**

### Q58. Why did Stage 4 micro performance drop below Stage 3?

Stage 4 showed that macro transfer alone did not solve rare-class scarcity and temporal mismatch. CAS(ME)^2 macro clips are 30 fps and have different distributions from CASME II micro clips. The fine-tuned model still struggled with Positive and Surprise.

---

## 10. Results and Limitations

### Q59. Why is accuracy higher than macro-F1?

Accuracy is dominated by majority classes. Macro-F1 averages each class equally, so failing on Positive and Surprise hurts strongly.

### Q60. Why did Positive and Surprise get 0% recall?

The micro dataset contains very few samples:

- Positive: about 45
- Surprise: about 34

After validation splits, there are too few examples for robust class-specific learning.

### Q61. Did GRL work?

The t-SNE evidence suggests yes. Emotion-colored embeddings show clusters, while subject-colored embeddings are mixed. This supports the claim that identity information was reduced.

### Q62. What remains unsolved?

Main unresolved issues:

- rare-class scarcity
- temporal interpolation loss
- low-FPS clips
- dataset domain mismatch
- lack of external validation on DFME/SMIC/SAMM
- no Grad-CAM++ visualization yet

### Q63. What is the strongest future work?

Run the pipeline on **DFME** using a stronger GPU, revise temporal sampling, and add interpretability such as Grad-CAM++ on u/v/strain streams.

---

## 11. Paper-Specific Defense Questions

### Q64. What did STSTNet contribute?

STSTNet contributed the shallow triple-stream idea using horizontal flow, vertical flow, and optical strain. My thesis adapts this idea to full 32-step video tensors and adds SimAM plus a Transformer.

### Q65. What did Less is More contribute?

It showed that onset-apex pairs are highly informative and that apex frames carry peak expression intensity. My work uses that insight but keeps the temporal sequence instead of reducing the clip to two frames.

### Q66. What did SLSTT contribute?

SLSTT showed that Transformer attention can model short- and long-range spatiotemporal relations in MER. My model uses an SLSTT-style Transformer on CNN-extracted temporal tokens.

### Q67. What did SimAM contribute?

SimAM contributes parameter-free attention. It is useful because my dataset is small, so adding attention without extra weights reduces overfitting risk.

### Q68. What did Bai et al. contribute?

Bai et al. showed that motion magnification can improve MER before temporal deep learning. This supports the use of EVM in Stage 1.

### Q69. What did Xia et al. macro-to-micro learning contribute?

It supports the idea that macro-expression data can guide micro-expression learning because both share facial muscle patterns. My Stage 4 implements a simpler two-phase pretrain/fine-tune version of that idea.

### Q70. What did Grad-CAM++ contribute?

Grad-CAM++ is not currently implemented in the thesis, but it is a strong future interpretability method. It could show whether the model focuses on brows, mouth, cheeks, or strain-heavy areas.

---

## 12. Hardware and Practical Questions

### Q71. What hardware did you use?

The results report lists an **NVIDIA GeForce RTX 4060 Laptop GPU with 8.6 GB VRAM**.

### Q72. Why was batch size only 2?

The input tensor is large: `[B,3,32,224,224]`. With model activations and gradients, only batch size 2 fit in available VRAM.

### Q73. Why would an RTX 5090 help?

A 5090-class GPU would allow:

- larger batch size
- less dependence on XBM
- longer temporal sequences
- higher resolution
- larger datasets such as DFME
- more robust ablation studies

### Q74. Would a larger GPU automatically solve the thesis?

No. It helps scale experiments, but the scientific problems remain: temporal sampling, class imbalance, subject/domain shift, and external validation.

---

## 13. Short Answers For Viva

### Q75. What is your strongest claim?

The strongest claim is that the pipeline can learn motion-based, subject-reduced representations using EVM, flow/strain, compact temporal modelling, GRL, and contrastive learning under LOSO evaluation.

### Q76. What is your most honest limitation?

Temporal interpolation and data scarcity. Standardizing to 33 frames made training possible, but it may drop high-FPS detail and duplicate low-FPS frames.

### Q77. What would you improve first?

I would test on DFME using stronger hardware and replace uniform temporal sampling with apex-aware or variable-length temporal modelling.

### Q78. What should you avoid saying?

Avoid saying:

- "Macro is always voluntary."
- "Stage 3 is adversarial training" without clarifying GRL.
- "Interpolation creates frames."
- "The model solved MER."
- "Stage 4 improved everything."

### Q79. What should you say instead?

Say:

- "Macro is usually longer/larger, but can be spontaneous."
- "Stage 3 uses GRL-based subject-invariant learning."
- "Temporal interpolation standardizes shape but cannot create missing motion."
- "The model addresses identity bias better than rare-class scarcity."
- "DFME and stronger hardware are the next validation step."

---

## 14. Final Defense Summary

My thesis is a motion-first MER system. It does not classify raw face photos. It turns each clip into a 32-step video motion tensor using EVM, dense optical flow, and optical strain. The architecture extends STSTNet with SimAM and Transformer temporal modelling. The training uses GRL to reduce subject identity, SupCon to cluster emotion embeddings, XBM to make contrastive learning possible at batch size 2, and Focal Loss to address imbalance. The main contribution is the integration of these components under strict subject-independent evaluation. The main limitation is that small, imbalanced, mixed-FPS datasets still restrict rare-class and temporal generalization.
