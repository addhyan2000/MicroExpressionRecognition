<!-- markdownlint-disable MD024 MD025 MD060 -->

# Detailed Research Paper Summary — Micro-Expression Recognition

**Scope:** All PDFs under `Papers/` and `Papers/additional papers/` were reviewed and synthesized for the thesis *Improving Micro-Expression Recognition through Temporal and Motion Modelling*.  
**Purpose:** Use this as the deep related-work companion to `PRESENTATION_GUIDE_FULL.md` and `THESIS_FAQ.md`.

---

## How To Read This Document

Each paper is summarized using the same structure:

- **What the paper is about**
- **Introduction / motivation**
- **Methodology**
- **Important parameters**
- **Experiments and results**
- **Figures / tables**
- **Conclusion and limitations**
- **Connection to my thesis**

The most important papers for the thesis defense are:

1. **CASME II**, **CAS(ME)^2**, **DFME**, and **MEGC 2019** for datasets/protocols.
2. **Optical strain**, **STSTNet**, **Less is More**, **OFF-ApexNet**, and **SLSTT** for representation and architecture.
3. **Learning from Macro-expression**, **A Neural Micro-Expression Recognizer**, and **Feature Refinement** for training and transfer ideas.
4. **SimAM**, **ViT**, and **Grad-CAM++** for general deep-learning components used or proposed.

---

# Part A — Dataset And Benchmark Papers

## 1. SMIC: A Spontaneous Micro-Expression Database — Inducement, Collection and Baseline

**Paper:** `Papers/additional papers/SMIC A_Spontaneous_Micro-expression_Database_Inducement_collection_and_baseline.pdf`

### What the paper is about

SMIC introduced one of the first public spontaneous micro-expression datasets. It was designed to solve the absence of genuine spontaneous ME video data for machine learning.

### Introduction / motivation

Earlier ME work relied on posed samples or very small datasets. SMIC uses an inhibited emotion paradigm where participants watch emotional videos while trying to keep a neutral face. This better matches the psychological idea of micro-expressions as involuntary leakage under concealment.

### Methodology

- Participants watched 16 emotion-inducing video clips.
- They were told to suppress facial expressions.
- Recordings were made with high-speed and standard cameras.
- Clips were annotated frame by frame by experts.
- Labels were collapsed into Positive, Negative, and Surprise.
- Baselines used LBP-TOP, TIM, and SVM.

### Important parameters

- High-speed subset: **100 fps**
- VIS/NIR subsets: **25 fps**
- Duration threshold: **<= 500 ms**
- Evaluation: **LOSO**
- Baseline features: LBP-TOP with TIM to fixed frame count

### Experiments and results

The paper evaluates both detection and recognition. Best high-speed recognition was around **48.78%**, showing that spontaneous MER is much harder than normal facial expression recognition.

### Figures / tables

- Recording setup figure shows the controlled elicitation environment.
- Dataset composition tables show Positive/Negative/Surprise distribution.
- Baseline result tables compare HS, VIS, and NIR subsets.

### Conclusion and limitations

SMIC proved spontaneous MER benchmarking was possible, but it was still small and had limited labels. Low-FPS VIS/NIR subsets performed worse, supporting the importance of high frame rates.

### Connection to my thesis

SMIC is not used in the implemented pipeline, but it is important background. It supports the thesis argument that high-speed video is necessary and that LOSO is the proper evaluation protocol. It is also part of the MEGC composite benchmark that includes STSTNet results.

---

## 2. CASME II: An Improved Spontaneous Micro-Expression Database and the Baseline Evaluation

**Paper:** `Papers/additional papers/CASME II An Improved Spontaneous Micro-Expression.pdf`

### What the paper is about

CASME II is a high-speed spontaneous micro-expression dataset and one of the main datasets used in the thesis.

### Introduction / motivation

CASME II improves on SMIC and CASME by increasing temporal resolution, face resolution, sample quality, and FACS annotation. It was created because earlier datasets were too small or too low resolution for reliable MER.

### Methodology

- Participants watched emotion-inducing videos and suppressed facial expressions.
- Videos were recorded at **200 fps**.
- Facial action units, onset, apex, and offset frames were annotated.
- Emotion labels were assigned using AUs, self-reports, and stimulus context.
- LBP-TOP + SVM was used as a baseline.

### Important parameters

- 26 valid subjects
- 247 micro-expression samples
- 200 fps recording
- Face area approximately 280 x 340 pixels
- 5-class setup: Happiness, Disgust, Surprise, Repression, Others
- Baseline: LBP-TOP with LOSO

### Experiments and results

The LBP-TOP baseline achieved approximately **63.41%** accuracy in the 5-class setting.

### Figures / tables

- Acquisition setup figure
- Example micro-expression sequence with onset/apex/offset
- Emotion label distribution table
- LBP-TOP parameter sweep table

### Conclusion and limitations

CASME II became a central benchmark because of its high fps, FACS labels, and reliable annotations. Limitations include small scale, lab setting, and class imbalance.

### Connection to my thesis

CASME II is a primary dataset in `Stage1_DataPipeline/config.py`. My pipeline uses its high-speed micro-expression clips and maps its labels into the four thesis classes. It is also the reason LOSO uses 26 folds.

---

## 3. CAS(ME)^2: A Database of Spontaneous Macro-Expressions and Micro-Expressions

**Paper:** `Papers/additional papers/CASME2_ADatabaseofSpontaneousMacro-expressionsandMicro-expressions.pdf`

### What the paper is about

CAS(ME)^2 provides both macro- and micro-expressions, making it important for macro-to-micro transfer learning.

### Introduction / motivation

Most earlier ME datasets only contained micro-expression clips. CAS(ME)^2 was created to support both recognition and spotting, and to allow comparison between macro and micro expressions from related recording conditions.

### Methodology

- Participants watched emotion-inducing videos and suppressed expressions.
- Videos were recorded at **30 fps**.
- Dataset contains macro-expression samples and micro-expression samples.
- Labels were assigned using AUs, self-report, and elicitation video content.
- Baseline used LBP-TOP + SVM.

### Important parameters

- 250 macro-expression samples
- 53 micro-expression samples in original paper
- 30 fps
- Emotion categories: Positive, Negative, Surprise, Other
- Macro duration: roughly 0.5-4 s
- Micro duration: <= 0.5 s

### Experiments and results

The paper reports LBP-TOP recognition baselines. The key value for this thesis is not the exact baseline accuracy, but the presence of both macro and micro expressions in one dataset family.

### Figures / tables

- Macro vs micro comparison figure
- Duration statistics table
- Emotion distribution table
- 36 facial ROI figure for baseline features

### Conclusion and limitations

CAS(ME)^2 is valuable because it contains macro and micro expressions. Its main limitation is **30 fps**, which is weak for very short micro-expressions.

### Connection to my thesis

CAS(ME)^2 is the source for Stage 4 macro pre-training and also contributes micro samples. The professor’s concern about low FPS is especially relevant here: at 30 fps, a micro-expression may contain only a few true frames.

---

## 4. SAMM: A Spontaneous Micro-Facial Movement Dataset

**Paper:** `Papers/additional papers/SAMM_A_Spontaneous_Micro-Facial_Movement_Dataset.pdf`

### What the paper is about

SAMM is a high-resolution, diverse spontaneous micro-facial movement dataset.

### Introduction / motivation

Earlier datasets were limited by ethnicity, age, resolution, and stimulus design. SAMM aims to improve diversity and recording quality.

### Methodology

- Participants from multiple ethnicities and age groups.
- High-speed grayscale camera at **200 fps**.
- High-resolution recordings around 2040 x 1088.
- FACS-certified coders annotated facial action units.
- Personalized stimuli were selected using pre-experiment questionnaires.

### Important parameters

- 32 participants
- Around 159 micro-expression samples used in many MER comparisons
- 200 fps
- 400 x 400 face crops
- FACS AU labels

### Experiments and results

The paper evaluates spotting and recognition baselines using LBP-TOP, HOG3D, DPM, and related methods. It also proposes improved spotting via adaptive baseline thresholding.

### Figures / tables

- Recording setup
- FACS-coded micro-movement examples
- AU frequency tables
- Spotting and recognition result tables

### Conclusion and limitations

SAMM improves diversity and resolution but still has limited sample size. It is more diverse than CASME II and should be considered for future external validation.

### Connection to my thesis

SAMM is not used in the implemented training but is important in STSTNet, OFF-ApexNet, SLSTT, and MEGC comparisons. It is a natural next dataset after CASME II/CAS(ME)^2.

---

## 5. DFME: A New Benchmark for Dynamic Facial Micro-Expression Recognition

**Paper:** `Papers/additional papers/DFME_A_New_Benchmark_for_Dynamic_Facial_Micro-Expression_Recognition.pdf`

### What the paper is about

DFME is a large-scale dynamic facial micro-expression benchmark and likely the dataset your professor referred to as "Douglas".

### Introduction / motivation

Small datasets like CASME II, SMIC, and SAMM restrict deep MER. DFME directly attacks data scarcity by collecting thousands of authentic high-speed ME videos.

### Methodology

- 671 participants recruited; 656+ valid participants with MEs.
- 7,526 well-labeled ME videos.
- Multiple high frame rates: **200, 300, 500 fps**.
- Neutralization paradigm with emotion-inducing videos.
- Annotation by more than 20 professional annotators over three years.
- Labels include emotion and facial action units.

### Important parameters

- 7,526 ME videos
- 200/300/500 fps
- 7 emotion labels plus others in annotation process
- 24 AU labels
- Subject-independent 10-fold cross-validation
- Baselines include R3D, I3D, LBP-TOP, MDMO, OffApexNet, STSTNet, RCN-A, MERSiam, FR

### Experiments and results

DFME shows that larger data allows general video models such as I3D to become competitive. It also shows that class imbalance and negative-emotion confusion remain difficult even at scale.

### Figures / tables

- Macro vs micro same-emotion figure showing smaller micro motion arrows
- Dataset comparison table showing DFME is much larger than older databases
- Recording environment figure
- Emotion and AU distribution figures
- Baseline result tables
- Confusion matrices

### Conclusion and limitations

DFME is the strongest current benchmark for large-scale MER, but it remains lab-based and demographically limited. It does not remove all imbalance problems.

### Connection to my thesis

DFME is the best future validation target. It would test whether my EVM + flow/strain + STSTNet-SimAM + Transformer + GRL/SupCon pipeline scales beyond small CASME-style datasets. It also justifies using a 5090-class GPU.

---

## 6. MEGC 2019: The Second Facial Micro-Expressions Grand Challenge

**Paper:** `Papers/additional papers/megc-2019-the-second-facial-micro-expressions-grand-challenge.pdf`

### What the paper is about

MEGC 2019 defines benchmark tasks for micro-expression spotting and recognition, especially composite-database evaluation.

### Introduction / motivation

MER needed standardized cross-dataset evaluation. Many methods overfit one dataset. MEGC 2019 combined SMIC, CASME II, and SAMM into a composite benchmark.

### Methodology

- Recognition task: composite dataset with 442 clips.
- Datasets: SMIC, CASME II, SAMM.
- Labels mapped to Positive, Negative, Surprise.
- Evaluation: LOSO across 68 subjects.
- Metrics: UF1 and UAR.

### Important parameters

- 442 samples
- 68 subjects
- 3-class setup
- LOSO
- UF1/UAR rather than raw accuracy

### Experiments and results

Top results included:

- STSTNet: UF1 around **0.7353**, UAR **0.7605**
- Liu et al. neural recognizer with adversarial/domain adaptation: UF1 around **0.7885**, UAR **0.7824**

Spotting results were very low, showing spotting remains much harder than recognition on cropped clips.

### Figures / tables

- Composite class distribution table
- Recognition leaderboard table
- Spotting evaluation table

### Conclusion and limitations

Composite recognition improved, but spotting remained unsolved. Optical flow and apex-frame methods dominated strong submissions.

### Connection to my thesis

MEGC validates three design choices:

- LOSO subject-independent evaluation
- optical-flow-based MER
- STSTNet lineage

My thesis differs by using four classes, CASME II + CAS(ME)^2, and full 32-step motion tensors.

---

# Part B — Motion, Optical Flow, Optical Strain, EVM

## 7. Optical Strain Based Recognition of Subtle Emotions

**Paper:** `Papers/additional papers/Optical strain based recognition of subtle emotions.pdf`

### What the paper is about

This paper is one of the earliest works using optical strain for micro-expression recognition, not just spotting.

### Introduction / motivation

Micro-expressions are subtle deformations. Optical strain can measure local deformation intensity, making it suitable for recognizing subtle facial motion.

### Methodology

- Compute optical flow between frames.
- Derive optical strain maps from flow gradients.
- Filter strain maps using Gaussian or Wiener filters.
- Temporally pool strain maps into a single representation.
- Resize and classify with SVM.

### Important parameters

- Optical strain tensor based on normal and shear components.
- Temporal sum pooling.
- Strain map resized to 50 x 50.
- Dataset: SMIC.
- Evaluation: LOSO.

### Experiments and results

The best result was approximately **53.56%** on SMIC, outperforming the original LBP-TOP baseline by about 5%.

### Figures / tables

- Pipeline figure
- Example surprise sequence
- Strain map comparison before/after filtering

### Conclusion and limitations

Optical strain is useful for subtle deformation recognition, but the method collapses temporal information into one map and uses classical SVM.

### Connection to my thesis

My `FlowStrainExtractor` uses the same principle but keeps optical strain as a full temporal channel across 32 frame pairs.

---

## 8. Subtle Expression Recognition Using Optical Strain Weighted Features

**Paper:** `Papers/additional papers/Subtle expression recognition using optical strain weighted features.pdf`

### What the paper is about

This paper uses optical strain as a weighting mechanism for LBP-TOP features.

### Introduction / motivation

LBP-TOP captures dynamic texture but does not automatically know which facial blocks contain important micro-motion. Optical strain can weight the important blocks.

### Methodology

- Compute strain maps.
- Divide the face into blocks.
- Pool strain magnitude per block.
- Use block strain as weights for LBP-TOP histograms.
- Classify with SVM.

### Important parameters

- Block-based strain pooling.
- LBP-TOP with XY, XT, YT planes.
- CASME II and SMIC evaluation.

### Experiments and results

The method improved over LBP-TOP baselines, reaching about **66.40%** on CASME II and about **57.71%** on SMIC in reported settings.

### Figures / tables

- Strain map figures
- Block pooling diagram
- LBP-TOP weighting diagram
- Result tables for CASME II and SMIC

### Conclusion and limitations

Optical strain helps identify informative facial regions. However, the method remains hand-crafted and block-dependent.

### Connection to my thesis

Instead of using strain as a hand-crafted weight, my model gives optical strain its own 3D-CNN stream so the network learns how to use it.

---

## 9. Spontaneous Subtle Expression Detection and Recognition Based on Facial Strain

**Paper:** `Papers/additional papers/Spontaneous subtle expression detection and recognition based on facial strain.pdf`

### What the paper is about

This paper combines direct optical strain features with optical-strain-weighted LBP-TOP for detection and recognition.

### Introduction / motivation

The authors aim to handle both spotting and recognition using strain-based features. It builds on earlier optical strain work and tries to combine deformation magnitude with dynamic texture.

### Methodology

- Optical strain features are computed from flow.
- Noise is reduced using thresholding and region segmentation.
- Strain maps are temporally pooled.
- OSW-LBP-TOP features are extracted.
- Concatenated features are classified with SVM.

### Important parameters

- OSF map resized to 50 x 50.
- LBP-TOP temporal radius tuned.
- SMIC and CASME II evaluation.
- LOSO / subject-independent evaluation.

### Experiments and results

The method performs well on SMIC detection and recognition, but gains are smaller on CASME II. The paper notes that high frame rate can create redundancy when temporal pooling is used.

### Figures / tables

- Optical flow vs strain visualization
- OSF pipeline
- OSW weighting pipeline
- Parameter sensitivity plots
- Comparison tables

### Conclusion and limitations

Strain helps subtle expression recognition, but hand-crafted pooling can lose temporal detail. CASME II high-speed redundancy can reduce the value of pooled strain.

### Connection to my thesis

This supports using optical strain but also warns against collapsing all frames into one pooled map. My thesis keeps strain as a 32-step tensor.

---

## 10. Macro- and Micro-Expression Spotting in Long Videos Using Spatio-Temporal Strain

**Paper:** `Papers/additional papers/Macro-and micro-expression spotting in long videos using spatio-temporal strain.pdf`

### What the paper is about

This paper uses strain signals to spot macro- and micro-expressions in long videos.

### Introduction / motivation

Before classifying expressions, a system must find when they happen. This paper focuses on spotting intervals in long videos using facial strain.

### Methodology

- Align face and segment into regions.
- Compute optical flow and strain.
- Sum strain over facial regions.
- Detect macro expressions using global strain peaks.
- Detect micro expressions using local region peaks and duration rules.

### Important parameters

- 8 facial regions
- Micro duration threshold around 1/5 s in their setup
- Region peak thresholds
- Long-video spotting task

### Experiments and results

Results were stronger on controlled videos and weaker on in-the-wild debate footage. Micro-expression spotting had many false positives.

### Figures / tables

- Facial region mask
- Strain maps
- Timeline peak detection figures
- Spotting result tables

### Conclusion and limitations

Spatio-temporal strain can spot expression intervals, but robust real-world spotting remains difficult.

### Connection to my thesis

My thesis performs recognition on pre-segmented clips, not long-video spotting. This paper supports the physical meaning of strain and motivates future spotting extensions.

---

## 11. Micro-Expression Recognition Based on Video Motion Magnification and Pre-Trained Neural Network

**Paper:** `Papers/Micro-expression recognition based on video motion magnification and pre-trained neural network.pdf`

### What the paper is about

This paper studies whether motion magnification improves deep MER.

### Introduction / motivation

Micro-expression motion is too small to see. Motion magnification may reveal hidden motion before classification.

### Methodology

- Apply amplitude-based or phase-based motion magnification.
- Use VGGFace2/ResNet-50 for spatial encoding.
- Use Bi-LSTM for temporal decoding.
- Use Grad-CAM for visual explanation.

### Important parameters

- Dataset: SMIC
- Motion magnification frequency band tuned for micro-expression duration
- Sliding temporal windows
- Bi-LSTM with 512 hidden units

### Experiments and results

The paper reports strong improvement with motion magnification, reaching about **75.76%** accuracy using phase-based magnification.

### Figures / tables

- Pipeline diagram
- Motion magnification comparison
- Grad-CAM visualizations
- Accuracy comparison table

### Conclusion and limitations

Motion magnification helps MER, especially phase-based magnification. Limitations include dataset scope and hand-tuned magnification parameters.

### Connection to my thesis

This supports my use of EVM in Stage 1. My pipeline differs by using EVM before optical flow/strain rather than RGB CNN + LSTM only.

---

## 12. Dynamic Texture Recognition Using Local Binary Patterns with an Application to Facial Expressions

**Paper:** `Papers/additional papers/Dynamic_Texture_Recognition_Using_Local_Binary_Patterns_with_an_Application_to_Facial_Expressions.pdf`

### What the paper is about

This is the foundational LBP-TOP paper used by many early MER baselines.

### Introduction / motivation

Dynamic textures require spatial and temporal representation. LBP-TOP extends local binary patterns to XY, XT, and YT planes.

### Methodology

- Compute LBP on spatial plane XY.
- Compute LBP on temporal planes XT and YT.
- Concatenate histograms.
- Use block-based face representation.
- Classify with SVM.

### Important parameters

- LBP neighbors and radii
- XY/XT/YT planes
- Block partitions
- Dynamic texture and facial expression datasets

### Experiments and results

LBP-TOP achieved strong results on dynamic textures and macro facial expression datasets, becoming a standard baseline for MER.

### Figures / tables

- VLBP and LBP-TOP diagrams
- XY/XT/YT plane figures
- Block face representation figures
- Accuracy comparison tables

### Conclusion and limitations

LBP-TOP is efficient and robust but hand-crafted. It can encode temporal texture but not explicit physics-based deformation.

### Connection to my thesis

My thesis replaces LBP-TOP with learned u/v/strain tensors and a neural temporal model.

---

# Part C — Architecture And Deep Model Papers

## 13. STSTNet: Shallow Triple Stream Three-Dimensional CNN for Micro-Expression Recognition

**Paper:** `Papers/additional papers/shallow-triple-stream-three-dimensional-cnn-ststnet-for-micro-expression-recognition.pdf`

### What the paper is about

STSTNet proposes a shallow 3D-CNN using three optical-flow-derived inputs: horizontal flow, vertical flow, and optical strain.

### Introduction / motivation

Deep networks overfit on tiny MER datasets. A shallow model using physically meaningful motion features can perform better than large models.

### Methodology

- Spot apex frame if needed.
- Compute optical flow between onset and apex.
- Derive optical strain.
- Build three-stream 3D-CNN on u, v, and strain.
- Fuse streams and classify.

### Important parameters

- Input resized to 28 x 28 x 3
- Shallow 3D convolution streams
- Learning rate 5e-5
- LOSO evaluation
- Composite dataset: SMIC + CASME II + SAMM

### Experiments and results

STSTNet achieved around:

- Composite UF1: **0.7353**
- Composite UAR: **0.7605**
- CASME II UAR: around **0.8686**

### Figures / tables

- Pipeline diagram
- STSTNet layer configuration
- Benchmark comparison table
- Confusion matrices

### Conclusion and limitations

STSTNet proves shallow flow/strain CNNs are strong for MER. However, it uses mainly onset-apex representation and does not model full temporal sequence.

### Connection to my thesis

This is the direct architectural ancestor of my Stage 2 backbone. My thesis extends it to 32-frame video tensors, SimAM attention, and Transformer temporal modelling.

---

## 14. Less is More: Micro-Expression Recognition from Video Using Apex Frame

**Papers:**  
`Papers/Less is more Micro-expression recognition from video using apex frame.pdf`  
`Papers/additional papers/less-is-more-micro-expression-recognition-from-video-using-apex-frame.pdf`

### What the paper is about

This paper argues that onset and apex frames can be sufficient for strong MER because the apex contains peak expression intensity.

### Introduction / motivation

High-speed videos have many redundant frames. The paper asks whether all frames are necessary.

### Methodology

- Use onset as neutral reference.
- Use apex as peak motion frame.
- Compute optical flow and strain-based Bi-WOOF descriptors.
- Classify using conventional classifiers.

### Important parameters

- Datasets: CAS(ME)^2, CASME II, SMIC-HS, SMIC-VIS, SMIC-NIR
- Apex-frame detection when ground truth unavailable
- Block sizes tuned for descriptors
- LOSO / subject-independent evaluation

### Experiments and results

The method achieved strong F-measure on CASME II and SMIC-HS and showed apex frames outperform random frame selection.

### Figures / tables

- Onset/apex/offset figure
- Bi-WOOF feature diagrams
- Random vs apex comparison tables
- Dataset statistics

### Conclusion and limitations

Apex frame is highly informative, but apex-only methods depend on accurate apex detection and lose temporal evolution.

### Connection to my thesis

My thesis accepts the importance of apex but does not reduce the input to one or two frames. I preserve 32 motion pairs so the Transformer can learn temporal order.

---

## 15. OFF-ApexNet on Micro-Expression Recognition System

**Paper:** `Papers/additional papers/off-apexnet-on-micro-expression-recognition-system.pdf`

### What the paper is about

OFF-ApexNet uses optical flow from onset to apex as input to CNNs for MER.

### Introduction / motivation

Since apex contains peak motion, optical flow between onset and apex can represent the expression efficiently.

### Methodology

- Detect or use apex frame.
- Compute horizontal and vertical optical flow.
- Feed u and v into CNN streams.
- Fuse and classify.

### Important parameters

- TV-L1 optical flow
- 28 x 28 flow inputs
- dual CNN branches
- LOSO on composite datasets

### Experiments and results

OFF-ApexNet is competitive on CASME II and composite datasets, with composite UF1 around **0.7196** in MEGC comparisons.

### Figures / tables

- Network architecture diagram
- Layer configuration
- Accuracy and F-measure tables

### Conclusion and limitations

Onset-apex optical flow plus CNN works well. Limitation: no strain, no full temporal modelling.

### Connection to my thesis

My model extends this direction by using u, v, and strain across 32 temporal pairs and by adding Transformer modelling.

---

## 16. Can Micro-Expression Be Recognized Based on Single Apex Frame?

**Paper:** `Papers/additional papers/can-micro-expression-be-recognized-based-on-single-apex-frame.pdf`

### What the paper is about

This paper belongs to the apex-frame research line asking whether one key frame can support MER.

### Introduction / motivation

The motivation is efficiency: if apex contains most emotion evidence, a model can avoid processing full videos.

### Methodology

The exact local PDF text was limited, but the paper aligns with apex-based methods using apex or onset-apex input for recognition.

### Important parameters

- Apex-frame selection
- Subject-independent evaluation
- Comparison with sequence-based baselines

### Experiments and results

Apex-frame methods generally show that apex carries strong discriminative information, especially in high-speed datasets with reliable apex annotations.

### Figures / tables

Likely includes apex-frame examples and comparison to sequence methods.

### Conclusion and limitations

Apex frames are useful, but apex-only pipelines can miss temporal dynamics and depend heavily on accurate apex spotting.

### Connection to my thesis

This paper is a counterpoint. My thesis uses video motion tensors because temporal evolution still matters, especially for a Transformer.

---

## 17. Short and Long Range Relation Based Spatio-Temporal Transformer for MER

**Paper:** `Papers/Short and Long Range Relation Based Spatio-Temporal Transformer for Micro-Expression Recognition.pdf`

### What the paper is about

SLSTT uses Transformer attention for MER, focusing on short- and long-range spatiotemporal relations.

### Introduction / motivation

CNNs capture local relations well but struggle with global spatial/temporal dependencies. Transformers can model relations across all tokens.

### Methodology

- Use optical-flow-based representation.
- Spatial Transformer encoder models spatial relations.
- Temporal aggregation models sequence relations.
- Classification head predicts emotion.

### Important parameters

- Transformer encoder
- self-attention
- temporal aggregation
- LOSO and composite evaluation

### Experiments and results

SLSTT reports strong results on SMIC, CASME II, SAMM, and composite evaluation, with CDE UF1 around **0.816** and UAR around **0.790** in reported settings.

### Figures / tables

- CNN vs Transformer relation figure
- SLSTT framework
- Long-term optical flow visualization
- Benchmark comparison tables

### Conclusion and limitations

Transformers improve global relation modelling in MER, but pure Transformer methods can be heavier and still rely on careful preprocessing.

### Connection to my thesis

My `SLSTTTransformer` is inspired by this paper, but I use CNN-generated temporal tokens rather than pure patch tokens.

---

## 18. Spatiotemporal Recurrent Convolutional Networks for Recognizing Spontaneous Micro-Expressions

**Papers:**  
`Papers/Spatiotemporal recurrent convolutional networks for recognizing spontaneous microexpressions.pdf`  
`Papers/additional papers/spatiotemporal-recurrent-convolutional-networks-for-recognizing-spontaneous-micro-expressions.pdf`

### What the paper is about

This paper proposes recurrent convolutional networks for MER using appearance and geometric streams.

### Introduction / motivation

Hand-crafted features are insufficient, but ordinary deep models overfit. Recurrent convolution can model temporal facial changes while controlling complexity.

### Methodology

- Preprocess videos with face alignment and EVM.
- Use appearance and geometry/flow streams.
- Use recurrent convolutional layers.
- Apply temporal augmentation and balanced loss.

### Important parameters

- EVM magnification
- 30-frame input sequences
- 64 x 48 or related spatial resolutions
- recurrent convolution layers
- LOSO / LOVO evaluation

### Experiments and results

STRCN variants outperform many hand-crafted baselines on SMIC, CASME II, and SAMM.

### Figures / tables

- Overall pipeline
- Temporal connectivity diagrams
- RCN architecture
- Dataset comparison tables

### Conclusion and limitations

Recurrent convolution helps model ME dynamics, but the pipeline is complex and still dataset-limited.

### Connection to my thesis

This supports EVM and temporal deep modelling. My thesis uses Transformer attention instead of recurrent convolution.

---

## 19. A Two-Stage 3D CNN Based Learning Method for Spontaneous MER

**Paper:** `Papers/additional papers/a-two-stage-3d-cnn-based-learning-method-for-spontaneous-micro-expression-recognition.pdf`

### What the paper is about

This paper proposes a two-stage 3D-CNN learning approach for spontaneous MER.

### Introduction / motivation

3D-CNNs can learn spatiotemporal features but overfit on small MER datasets. A staged design tries to make 3D learning more stable.

### Methodology

- Use 3D convolution for spatiotemporal learning.
- Separate learning into stages, usually feature learning followed by target recognition/fusion.
- Use key frames or normalized sequences depending on dataset.

### Important parameters

- 3D convolution layers
- key-frame or sequence preprocessing
- public MER datasets such as SMIC/CASME II/SAMM

### Experiments and results

The paper is cited as a strong 3D-CNN baseline in later DFME and survey papers.

### Figures / tables

Expected figures include two-stage framework and 3D-CNN architecture diagrams.

### Conclusion and limitations

3D-CNNs are promising for MER but must be controlled carefully due to sample scarcity.

### Connection to my thesis

My model also uses 3D convolutions, but only shallow spatial-temporal feature extraction before Transformer sequence modelling.

---

## 20. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

**Paper:** `Papers/additional papers/an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale.pdf`

### What the paper is about

This is the Vision Transformer paper. It shows that images can be treated as patch sequences and processed by Transformers.

### Introduction / motivation

Transformers had dominated NLP. The paper asks whether the same architecture can replace CNNs for images when trained at large scale.

### Methodology

- Split image into 16x16 patches.
- Linearly embed patches.
- Add position embeddings and class token.
- Process with Transformer encoder.
- Fine-tune on downstream image classification.

### Important parameters

- ViT-B/16: 12 layers, 768 dim, 12 heads
- ViT-L/16: 24 layers, 1024 dim, 16 heads
- Very large pretraining datasets such as JFT-300M

### Experiments and results

ViT performs extremely well when pretrained at scale, but underperforms CNNs when data is small.

### Figures / tables

- Patch embedding architecture figure
- Data scale vs performance plots
- Model size tables

### Conclusion and limitations

Transformers can work for vision, but they need large-scale pretraining or strong regularization.

### Connection to my thesis

This explains why I use a small Transformer over CNN tokens rather than a full ViT. My dataset has only hundreds of clips, so a large ViT would overfit.

---

## 21. SimAM: A Simple, Parameter-Free Attention Module for CNNs

**Paper:** `Papers/additional papers/SimAM A Simple, Parameter Free Attention Module for Convolutional Neural Networks.pdf`

### What the paper is about

SimAM proposes attention without adding trainable parameters.

### Introduction / motivation

Most attention modules such as SE or CBAM add parameters. SimAM derives attention from a neuroscience-inspired energy function.

### Methodology

- Compute neuron importance from local statistics.
- Use an energy function involving mean, variance, and a regularization term.
- Apply sigmoid attention to feature maps.

### Important parameters

- Regularization lambda, commonly around `1e-4`
- No trainable parameters

### Experiments and results

SimAM improves CNN performance on classification, detection, and segmentation with zero added parameters.

### Figures / tables

- Attention visualization
- Classification result tables
- Comparison with SE/CBAM/ECA

### Conclusion and limitations

SimAM is lightweight and effective, but the original paper focuses on 2D image CNNs rather than video MER.

### Connection to my thesis

My `SimAM3D` adapts this idea to 3D feature maps in each STSTNet stream.

---

# Part D — Transfer, Domain Adaptation, Graphs, Apex, Interpretability

## 22. Learning from Macro-Expression: A Micro-Expression Recognition Framework

**Paper:** `Papers/additional papers/Learning from Macro-expression - a Micro-expression Recognition Framework.pdf`

### What the paper is about

This paper proposes learning MER from macro-expression guidance.

### Introduction / motivation

Micro-expression samples are scarce, while macro-expression datasets are larger. Since they share facial muscle movements, macro data can guide micro recognition.

### Methodology

- Build MicroNet and MacroNet.
- Use expression-identity disentanglement.
- Guide micro features using macro feature and label space.
- Use triplet loss, adversarial loss, and loss inequality regularization.

### Important parameters

- Macro datasets such as CK+, MMI, Oulu-CASIA
- Micro datasets such as SMIC, CASME II, SAMM
- Apex-centered frames
- ResNet-style feature extractors

### Experiments and results

The framework improves MER over baseline and supports the macro-to-micro learning hypothesis.

### Figures / tables

- Macro vs micro examples
- Architecture diagram
- Ablation tables for losses

### Conclusion and limitations

Macro-expression data can help MER, but macro/micro domain differences remain.

### Connection to my thesis

This is the main paper supporting Stage 4. My implementation uses direct pretraining/fine-tuning rather than the paper’s full disentanglement framework.

---

## 23. Revealing the Invisible with Model and Data Shrinking for Composite-Database MER

**Paper:** `Papers/additional papers/revealing-the-invisible-with-model-and-data-shrinking-for-composite-database-micro-expression-recognition.pdf`

### What the paper is about

This paper argues that smaller models and smaller inputs can generalize better in composite MER.

### Introduction / motivation

Deep models overfit and domain shift is severe across SMIC, CASME II, and SAMM.

### Methodology

- Use optical flow maps.
- Reduce input resolution.
- Use recurrent convolutional networks and parameter-free modules.
- Evaluate on composite database.

### Important parameters

- Low-resolution flow maps
- composite database
- LOSO
- UF1/UAR

### Experiments and results

RCN variants achieve strong composite UF1/UAR and outperform heavier CNNs. The paper also notes that motion magnification can sometimes amplify domain noise in composite evaluation.

### Figures / tables

- Model diagrams
- Input-size comparisons
- CAM visualizations
- Composite benchmark tables

### Conclusion and limitations

For tiny MER datasets, reducing complexity can outperform larger models.

### Connection to my thesis

This supports my compact 382K-parameter design. It also cautions that EVM may not always help under domain shift.

---

## 24. A Neural Micro-Expression Recognizer

**Paper:** `Papers/additional papers/a-neural-micro-expression-recognizer.pdf`

### What the paper is about

This is a MEGC-style neural recognizer using optical flow, part-based CNNs, expression magnification/reduction, and domain adaptation.

### Introduction / motivation

Composite MER needs methods that generalize across datasets and subjects. Optical-flow-based local features are strong for subtle expression.

### Methodology

- Detect face and landmarks.
- Compute onset-apex optical flow.
- Use part-based ResNet-style CNN.
- Use expression magnification/reduction.
- Add adversarial/domain adaptation with macro data.

### Important parameters

- Composite dataset
- 3-class setup
- LOSO
- UF1/UAR

### Experiments and results

The method reaches around **0.7885 UF1** and **0.7824 UAR** on the MEGC composite benchmark, outperforming STSTNet in that setting.

### Figures / tables

- Network architecture
- EMR examples
- Domain adaptation diagram
- Benchmark result table

### Conclusion and limitations

Part-based optical flow and domain adaptation are strong. The method still uses onset-apex flow rather than full video.

### Connection to my thesis

This is close in spirit to my GRL training, but I apply GRL for subject identity invariance and combine it with SupCon/XBM and full video motion tensors.

---

## 25. Joint Local and Global Information Learning with Single Apex Frame Detection for MER

**Paper:** `Papers/additional papers/joint-local-and-global-information-learning-with-single-apex-frame-detection-for-micro-expression-recognition.pdf`

### What the paper is about

This paper detects apex frames and uses local-global learning for apex-based MER.

### Introduction / motivation

If apex contains peak expression information, good apex detection plus local/global modelling may be enough for recognition.

### Methodology

- Detect apex using frequency-domain pixel change.
- Use local multi-instance learning to find informative regions.
- Use global context branch.
- Combine with center loss for better embedding separation.

### Important parameters

- Apex detection
- local/global branches
- center loss
- CASME II, SAMM, SMIC, composite evaluation

### Experiments and results

The method shows apex-only recognition can be competitive and that local regions improve recognition.

### Figures / tables

- Apex detection pipeline
- Local/global network diagram
- Benchmark result tables

### Conclusion and limitations

Apex frames are powerful, but apex-only methods miss temporal progression.

### Connection to my thesis

This supports using apex-aware future sampling. My current method uses uniform sampling and full 32-step temporal modelling.

---

## 26. Micro-Expression Recognition Based on Facial Graph Representation Learning and AU Fusion

**Paper:** `Papers/additional papers/micro-expression-recognition-based-on-facial-graph-representation-learning-and-facial-action-unit-fusion.pdf`

### What the paper is about

This paper uses graph learning over facial regions and fuses facial action unit information.

### Introduction / motivation

Micro-expressions are localized and related to AUs. Graph structures can model relations between facial regions.

### Methodology

- Use onset and apex features.
- Extract landmark-based patches.
- Build graph nodes from facial patches.
- Use attention/Transformer-like edge learning.
- Fuse AU information with graph convolution.

### Important parameters

- Landmark patches
- graph nodes
- AU labels
- GCN / Transformer edge learning
- LOSO

### Experiments and results

The method reports strong results on CASME II, SAMM, and composite MER, with performance competitive with STSTNet and neural recognizer methods.

### Figures / tables

- Dual graph/AU architecture
- patch/node diagrams
- benchmark tables

### Conclusion and limitations

Graph and AU fusion improves MER but requires reliable landmarks and AU labels.

### Connection to my thesis

My thesis does not use AU labels. This paper is a future direction: combine my motion tensors with AU/graph priors.

---

## 27. A Delaunay-Based Temporal Coding Model for MER

**Paper:** `Papers/additional papers/A Delaunay-Based Temporal Coding Model for Micro-expression Recognition-703-716.pdf`

### What the paper is about

This paper uses Delaunay triangulation over facial landmarks to capture temporal changes.

### Introduction / motivation

Micro-expressions are local and subtle. Landmark-based triangulation can adapt to facial structure better than fixed grids.

### Methodology

- Detect facial landmarks.
- Build Delaunay triangles.
- Warp/normalize face.
- Code temporal intensity changes inside triangles.
- Select salient subregions.
- Classify with SVM or Random Forest.

### Important parameters

- AAM landmarks
- Delaunay triangulation
- TIM
- salient region threshold

### Experiments and results

The method improves over some LBP-TOP baselines on CASME and CASME II, with reported CASME II performance around the low 70% range in some settings.

### Figures / tables

- Delaunay triangulation diagrams
- temporal coding pipeline
- parameter sensitivity figures

### Conclusion and limitations

Adaptive geometry helps, but the pipeline is hand-crafted and landmark-dependent.

### Connection to my thesis

My dense optical flow/strain representation is a learned alternative to landmark triangulation.

---

## 28. Microexpression Identification and Categorization Using a Facial Dynamics Map

**Paper:** `Papers/additional papers/microexpression-identification-and-categorization-using-a-facial-dynamics-map.pdf`

### What the paper is about

This paper proposes Facial Dynamics Map (FDM), an optical-flow-based representation for identifying and categorizing MEs.

### Introduction / motivation

Dynamic motion is more informative for micro-expressions than static texture.

### Methodology

- Align faces.
- Compute optical flow.
- Divide face into spatiotemporal cuboids.
- Estimate principal flow direction per cuboid.
- Build Facial Dynamics Map.
- Classify with SVM.

### Important parameters

- cuboid scales
- temporal interpolation
- optical flow
- identification and categorization tasks

### Experiments and results

FDM outperforms LBP-TOP and some earlier methods on SMIC and CASME datasets.

### Figures / tables

- FDM pipeline
- flow visualizations
- comparison charts

### Conclusion and limitations

Motion direction maps are interpretable and useful, but still hand-crafted.

### Connection to my thesis

FDM supports the thesis decision to model motion explicitly instead of raw RGB appearance.

---

## 29. Towards Reading Hidden Emotions: A Comparative Study of Spontaneous ME Spotting and Recognition

**Paper:** `Papers/additional papers/towards-reading-hidden-emotions-a-comparative-study-of-spontaneous-micro-expression-spotting-and-recognition-methods.pdf`

### What the paper is about

This paper compares spotting and recognition methods and builds a pipeline using EVM, TIM, and descriptors.

### Introduction / motivation

Micro-expression systems must both detect and recognize hidden emotions. The paper compares handcrafted descriptors and evaluates a full pipeline.

### Methodology

- Spot candidate expressions using feature difference peaks.
- Use EVM to magnify subtle motion.
- Apply TIM.
- Extract descriptors such as LBP, HOG, HIGO on TOP planes.
- Classify with SVM.

### Important parameters

- EVM alpha around 4 in reported experiments
- TIM to 10 frames
- LOSO
- 3-class recognition

### Experiments and results

EVM + TIM + gradient features improved recognition. The end-to-end spotting-plus-recognition pipeline remained much harder than recognition on cropped clips.

### Figures / tables

- spotting pipeline
- MESR system diagram
- descriptor comparison tables

### Conclusion and limitations

EVM and TIM are useful, but spotting remains difficult and false positives are common.

### Connection to my thesis

This paper strongly supports the EVM step and the use of temporal normalization, while also exposing the weakness of interpolation and spotting.

---

## 30. Feature Refinement: Expression-Specific Feature Learning and Fusion for MER

**Paper:** `Papers/additional papers/feature-refinement-an-expression-specific-feature-learning-and-fusion-method-for-micro-expression-recognition.pdf`

### What the paper is about

This paper learns expression-specific features instead of relying only on shared features.

### Introduction / motivation

Different micro-expression classes can share similar facial movements. A single shared representation may confuse classes.

### Methodology

- Use optical flow input.
- Learn shared features with Inception-style network.
- Add expression proposal/refinement modules.
- Fuse expression-specific features for classification.

### Important parameters

- optical flow maps
- expression proposal loss
- CDE/composite evaluation
- UF1/UAR

### Experiments and results

The method improves over a basic Inception baseline and reaches composite UF1/UAR close to strong MEGC methods.

### Figures / tables

- network architecture
- confusion matrices
- t-SNE feature visualizations
- ablation tables

### Conclusion and limitations

Expression-specific refinement improves class separation but adds complexity and parameters.

### Connection to my thesis

My SupCon objective tries to create emotion-specific clusters more simply. The paper’s t-SNE validation is similar to my t-SNE evidence.

---

## 31. Grad-CAM++: Generalized Gradient-Based Visual Explanations

**Paper:** `Papers/additional papers/GRAD CAM ++Grad-CAM_Generalized_Gradient-Based_Visual_Explanations_for_Deep_Convolutional_Networks.pdf`

### What the paper is about

Grad-CAM++ improves class-discriminative visual explanations for CNNs.

### Introduction / motivation

Deep models need interpretability. Grad-CAM sometimes fails for multiple object instances or small discriminative regions.

### Methodology

- Use gradients of class score with respect to convolution feature maps.
- Compute weighted channel importance.
- Produce heatmaps using ReLU over weighted feature maps.
- Combine with guided backpropagation for high-resolution maps.

### Important parameters

- last convolution layer
- positive gradients
- second-order gradient weighting

### Experiments and results

Grad-CAM++ improves localization and human preference metrics over Grad-CAM.

### Figures / tables

- Grad-CAM vs Grad-CAM++ examples
- localization metric tables
- multi-instance visual comparisons

### Conclusion and limitations

Grad-CAM++ gives better post-hoc explanations but is still not a built-in model explanation.

### Connection to my thesis

Not implemented yet. It is a strong future method to show whether my model focuses on brows, cheeks, mouth, or optical-strain-heavy regions.

---

# Part E — Survey And Overview Papers

## 32. Deep Learning for Micro-Expression Recognition: A Survey

**Paper:** `Papers/Deep learning for micro-expression recognition A survey.pdf`

### What the paper is about

This survey categorizes deep learning methods for MER.

### Introduction / motivation

MER has moved from hand-crafted features to deep learning, but small data and identity/domain bias remain major barriers.

### Methodology / taxonomy

The survey covers:

- datasets
- preprocessing
- input modalities
- CNNs, 3D-CNNs, RNNs, GCNs, Transformers
- transfer learning
- domain adaptation
- losses and evaluation metrics

### Important parameters

The survey emphasizes:

- LOSO
- UF1/UAR
- optical flow
- apex-frame methods
- macro-to-micro transfer
- imbalanced learning

### Experiments and results

The survey compares many published results across SMIC, CASME II, SAMM, and composite datasets.

### Figures / tables

- MER taxonomy figure
- dataset comparison tables
- method performance tables

### Conclusion and limitations

Deep MER has improved but lacks data, generalization, standardized protocols, and ethical clarity.

### Connection to my thesis

This is the broad literature map that justifies almost every major thesis component.

---

## 33. An Overview of Facial Micro-Expression Analysis: Data, Methodology and Challenge

**Paper:** `Papers/An_Overview_of_Facial_Micro-Expression_Analysis_Data_Methodology_and_Challenge.pdf`

### What the paper is about

This overview reviews ME datasets, methodologies, challenges, and future directions.

### Introduction / motivation

The field has grown rapidly but remains fragmented across datasets, metrics, and protocols.

### Methodology

The paper reviews:

- dataset construction
- spotting
- recognition
- apex methods
- AU methods
- macro-to-micro transfer
- synthetic data
- ethical and bias concerns

### Important parameters

The overview emphasizes duration thresholds, fps, FACS, LOSO, UF1/UAR, and benchmark consistency.

### Experiments and results

It summarizes reported results across many papers rather than running one new experiment.

### Figures / tables

- publication trend figures
- pipeline overview
- dataset sample figures
- benchmark tables

### Conclusion and limitations

Main challenges include data scarcity, imbalance, in-the-wild generalization, spotting, multimodal analysis, and bias.

### Connection to my thesis

It supports the thesis framing: data scarcity, identity bias, low-FPS limitations, and need for robust evaluation.

---

## 34. Video-Based Facial Micro-Expression Analysis: A Survey of Datasets, Features and Algorithms

**Paper:** `Papers/additional papers/video-based-facial-micro-expression-analysis-a-survey-of-datasets-features-and-algorithms.pdf`

### What the paper is about

This survey focuses on video-based ME analysis, including datasets, features, and algorithms.

### Introduction / motivation

Micro-expressions are dynamic events, so video analysis is essential.

### Methodology

The survey reviews:

- handcrafted features
- motion and geometry features
- optical flow and optical strain
- spotting methods
- recognition methods
- deep learning and transfer learning

### Important parameters

Dataset fps, resolution, sample count, emotion labels, spotting/recognition protocols.

### Experiments and results

The paper provides comparative analysis of dataset and method performance.

### Figures / tables

- macro vs micro examples
- neural pathway explanation
- method taxonomy
- dataset comparison tables

### Conclusion and limitations

Video-based MER needs better datasets, better generalization, and stronger temporal modelling.

### Connection to my thesis

This paper supports the decision to use video tensors instead of static photos.

---

## 35. Facial Micro-Expression Recognition: A Machine Learning Approach

**Paper:** `Papers/Facial micro-expression recognition A Machine learning approach.pdf`

### What the paper is about

This paper compares classical machine learning approaches using LBP and LBP-TOP on CASME II.

### Introduction / motivation

Before deep MER, LBP and LBP-TOP were standard feature descriptors. The paper evaluates SVM and ELM classifiers.

### Methodology

- Extract LBP from apex/static frames.
- Extract LBP-TOP from video sequences.
- Train SVM and ELM classifiers.

### Important parameters

- CASME II
- 5-fold cross-validation
- LBP and LBP-TOP descriptors
- ELM hidden units

### Experiments and results

The paper reports very high accuracies, especially with LBP-TOP + ELM, but the protocol is weaker than strict LOSO.

### Figures / tables

- LBP-TOP planes
- confusion matrices
- classifier comparison tables

### Conclusion and limitations

Classical methods can perform well under easier protocols, but results may not generalize to unseen subjects.

### Connection to my thesis

This is a baseline-era paper. My work uses stricter LOSO and deep motion representations.

---

# Part F — Remaining Related Methods

## 36. A Neural Micro-Expression Recognizer, Feature Refinement, Graph Learning, FDM, DTCM, and Apex Papers as a Group

The remaining method papers collectively show the field's movement from hand-crafted geometry and texture to flow-based, graph-based, apex-based, and domain-adaptive deep learning.

### Shared introductions

They all start from the same constraints:

- micro-expressions are brief and low intensity
- datasets are small
- subject-independent evaluation is difficult
- local facial regions are important
- optical flow and apex frames are strong signals

### Shared methodologies

Across these papers, the common strategies are:

- apex detection
- onset-apex optical flow
- local facial region modelling
- graph or landmark structure
- expression-specific feature refinement
- domain adaptation or macro-expression transfer
- t-SNE or CAM-style feature visualization

### Shared parameters

Common settings include:

- LOSO
- UF1/UAR
- 3-class composite mapping
- CASME II, SMIC, SAMM
- optical flow maps resized to small spatial dimensions
- shallow or specialized networks

### Shared results

The strongest methods on MEGC-style composite evaluation generally achieve UF1/UAR in the 0.70-0.80 range. They are often based on optical flow, apex frames, part-based models, graph/AU fusion, or expression-specific branches.

### Shared limitations

- heavy dependence on apex accuracy
- no full video temporal modelling in many methods
- small datasets
- imbalanced classes
- limited in-the-wild validation
- often no explicit subject-invariance objective

### Connection to my thesis

My thesis combines several of these lessons:

- flow and strain are more robust than raw RGB
- shallow models are safer on small datasets
- temporal modelling matters
- subject identity must be controlled
- macro-expression transfer is promising but not sufficient
- class imbalance remains a major bottleneck

---

# Part G — Direct Paper-To-Thesis Mapping

| Thesis Component | Main Supporting Papers | How The Thesis Uses It |
|---|---|---|
| Dataset motivation | SMIC, CASME II, CAS(ME)^2, SAMM, DFME, surveys | Shows small data, fps issues, FACS labels, benchmark protocols |
| EVM | Wu et al. via EVM literature, Bai et al., Li et al. | Amplifies sub-pixel motion before flow/strain |
| Optical flow | Farneback, OFF-ApexNet, FDM, STSTNet | Uses horizontal and vertical dense flow channels |
| Optical strain | Shreve, Liong optical strain papers, STSTNet | Uses deformation magnitude as third channel |
| Apex insight | Less is More, OFF-ApexNet, single apex papers | Recognizes apex importance but keeps full 32-step sequence |
| STSTNet backbone | STSTNet | Three unshared streams for u/v/strain |
| SimAM | SimAM | Parameter-free 3D attention per stream |
| Transformer | SLSTT, ViT | Temporal self-attention over CNN tokens |
| Macro transfer | Learning from Macro-expression, A Neural MER | Stage 4 macro pre-train then micro fine-tune |
| Subject invariance | DANN/GRL literature, MEGC neural recognizer | GRL identity head for subject-invariant features |
| Contrastive learning | SupCon, XBM | Emotion clustering under small batch |
| Imbalance | Focal Loss, surveys, DFME | Rare-class handling |
| Evaluation | MEGC 2019, CASME II, surveys | LOSO, macro-F1/UF1/UAR awareness |
| Future validation | DFME | Larger high-fps benchmark requiring stronger GPU |
| Interpretability | Grad-CAM++, FDM, RCN CAM | Future saliency over u/v/strain streams |

---

# Part H — Final Synthesis

The reviewed literature shows a clear development path:

1. **LBP-TOP and classical descriptors** proved that temporal texture matters.
2. **Optical flow and optical strain** showed that explicit motion/deformation is more meaningful than raw appearance.
3. **Apex-frame papers** showed that the peak frame is highly informative, but also introduced dependence on apex detection.
4. **STSTNet and OFF-ApexNet** demonstrated that shallow flow-based CNNs work well on small MER data.
5. **SLSTT and Transformer papers** showed that attention can capture longer-range spatiotemporal relations.
6. **Macro-transfer and domain-adaptation papers** showed that small ME datasets need external guidance and invariance constraints.
7. **Surveys and DFME** show that data scale, fps, imbalance, and subject/domain shift remain the core open problems.

My thesis is best understood as a carefully scoped integration of these lessons:

```text
Raw video
 -> EVM
 -> temporal sampling
 -> Farneback u/v + optical strain
 -> [3,32,224,224] motion tensor
 -> STSTNet-style 3-stream 3D-CNN
 -> SimAM
 -> Transformer
 -> GRL + SupCon/XBM + Focal
 -> LOSO evaluation
 -> macro-to-micro transfer study
```

The strongest defense position is:

> The thesis does not claim each component is new. It claims that combining motion magnification, physics-based motion channels, compact temporal modelling, subject-invariant training, contrastive clustering, and macro-to-micro transfer creates a clear and defensible MER pipeline. The results show progress on identity-invariant representation learning, while the failure modes show that rare-class scarcity, temporal interpolation, and low-FPS data remain the next problems to solve.
