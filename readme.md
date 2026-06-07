This is a substantial refactoring task. Before writing any code, I need to understand the existing codebase in depth — the current model definitions, dataset loaders, training loops, and where the GRL/SupCon/XBM and hardcoded logic live. Let me explore thoroughly.

Composer 2.5 Fast

Composer 2.5 Fast

I now have a thorough map. Let me read the key model and loss files directly so I preserve exact implementations when building the new modular structure.

Now let me read the dataset, focal loss, EVM, and flow/strain extractor files to preserve their behavior in the refactored code.

Let me read the Stage 1 config and the models __init__.py to capture hardcoded items and exports before designing the refactor.

I now have a complete, precise understanding of every component. Let me set up the plan and build the isolated ablation study module.

Create Ablation_Study/ablation_config.py
Create Ablation_Study/models.py
Create Ablation_Study/dataset.py
Create Ablation_Study/losses.py
Create Ablation_Study/metrics.py
Create Ablation_Study/trainer.py
Create Ablation_Study/run_ablation_experiments.py
Create Ablation_Study/__init__.py and verify imports / lint
Now I'll build the isolated Ablation_Study/ module. Starting with the configuration matrix.

"""
ablation_config.py — Experiment Configuration for the Stage 1 + Stage 2 Ablation
=================================================================================
Now the modular model definitions with conditional forward paths.

"""
models.py — Modular, Toggleable MER Architecture for the Ablation Study
========================================================================
Now the generalized dataset loader.

"""
dataset.py — Generalised Single-Dataset Loader for the Ablation Study
======================================================================
Now the decoupled loss module.

"""
losses.py — Decoupled Classification Loss for the Ablation Study
=================================================================
Now the metrics module for Accuracy, Macro-F1, and confusion matrices with per-config saving.

"""
metrics.py — Evaluation Metrics & Result Persistence for the Ablation Study
============================================================================
Now the clean trainer (no GRL/SupCon/XBM).

"""
trainer.py — Clean Single-Task Trainer for the Ablation Study
==============================================================
Now the orchestrator that ties everything together.

"""
run_ablation_experiments.py — Ablation Orchestrator (Stage 1 + Stage 2)
========================================================================
Now the package __init__.py, then I'll lint and run the standalone smoke tests.

"""
Ablation_Study — Isolated Stage 1 + Stage 2 Ablation for MER
=============================================================
Now let me verify the modules actually work — run the standalone smoke tests for the config matrix, the model (all 8 toggle combos), losses, and metrics.

(eval):1: command not found: python
(eval):1: command not found: python
Let me find the right Python interpreter.

/opt/homebrew/bin/python3
python3.11 not found
python3.12 not found
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import torch; print('torch', torch.__version__)
    ^^^^^^^^^^^^
ModuleNotFoundError: No module named 'torch'
-rw-r--r--@  1 addhyanpant  staff  6148 May 31 15:22 .DS_Store
drwxr-xr-x@ 24 addhyanpant  staff   768 Jun  7 21:14 MicroExpressionRecognition
---find venvs---
---conda---
conda not found
Torch isn't installed in this environment. Let me at least verify the torch-free modules (config, and metrics if numpy is available).

Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import numpy; print('numpy', numpy.__version__)
    ^^^^^^^^^^^^
ModuleNotFoundError: No module named 'numpy'
/opt/homebrew/Cellar/python@3.14/3.14.4_1/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python: can't open file '/Users/addhyanpant/Documents/Thesis/MicroExpressionRecognition/ablation_config.py': [Errno 2] No such file or directory

num_classes   : 3
class_names   : ['Negative', 'Positive', 'Surprise']
EVM tensors   : /Users/addhyanpant/Documents/Thesis/MicroExpressionRecognition/Processed_Data/tensors
RAW tensors   : /Users/addhyanpant/Documents/Thesis/MicroExpressionRecognition/Processed_Data/tensors_raw
The configuration matrix runs correctly and exactly matches the thesis brief (all 8 phases). Torch/NumPy aren't installed in this shell's Python (a bare Homebrew 3.14), so I can't execute the torch-dependent smoke tests here — but lints are clean and the logic mirrors the existing Stage 2 modules I read.

Let me harden one subtle coupling in the orchestrator: the train/val views are two dataset instances that must share identical sample ordering. I'll add a guard.

        # A separate augmented "view" of the same dataset for the train split.
        # Both instances read the same CSV/tensor_dir with identical filters, so
        # their internal ``samples`` ordering is deterministic and aligned — the
        # assertion below makes that contract explicit (indices must map 1:1).
Let me do a syntax/compile check on all the torch-dependent files (this doesn't import torch, just validates the Python parses).

ALL FILES COMPILE OK
All modules compile cleanly. Let me mark the final todo complete.

Done. I built a self-contained Ablation_Study/ package that isolates Stage 1 + Stage 2 and strips out all Stage 3/4 machinery (GRL, identity head, SupCon, XBM). It reuses the exact math from your existing Stage 2 modules but rewires everything around toggleable components.

What was created
Ablation_Study/
├── ablation_config.py            # AblationConfig + ExperimentConfig + the 8-cell matrix
├── models.py                     # AblationMERModel (conditional forward) + build_model
├── dataset.py                    # MERAblationDataset (generalized, single-dataset)
├── losses.py                     # FocalLoss + build_loss (decoupled, no SupCon/XBM)
├── metrics.py                    # MetricsComputer + ResultWriter (Acc/Macro-F1/CM)
├── trainer.py                    # AblationTrainer (clean single-task loop)
├── run_ablation_experiments.py   # Orchestrator over the matrix
└── __init__.py
How the three required deliverables map
1. models.py — AblationMERModel decides its graph from three model-level booleans. The forward pass routes through:

Stage A (spatial stem) → [B, 32, 96]: ThreeStreamCNNBackbone (with optional per-stream SimAM) when use_cnn=True, otherwise RawPatchEmbedding (adaptive-pool raw flow → flatten → linear projection).
Stage B (temporal) → [B, 96]: SLSTTTransformer when use_transformer=True, otherwise parameter-free TemporalPooling (mean/max).
Stage C → [B, num_classes]. Input stays [B, 3, 32, 224, 224]; Conv3D kernels stay (1,3,3); the spatial-pool-to-sequence step is preserved. Every toggle-off path is documented in the module docstring and inline.
2. dataset.py — MERAblationDataset returns (tensor, label) only (no identity leakage to the network), driven by explicit emotion_map (→ num_classes), dataset_filter (single dataset, e.g. CASME_II), expression_filter, and sequence_length. It retains subject ids solely to build strict subject-disjoint splits (subject_disjoint_split and loso_folds).

3. run_ablation_experiments.py — loops the matrix, selects the EVM/raw tensor dir per Variable A, builds the conditional model, trains/validates on a subject-disjoint split (holdout or LOSO), and writes per-config metrics.json, confusion_matrix.npy/.png, plus an aggregated summary.csv.

Verified
ablation_config.py runs and prints all 8 configs matching Phases I–IV exactly.
All modules byte-compile; lints are clean.
Two things to note
Variable A (EVM) is data-level, not a network flag. It selects between Processed_Data/tensors (EVM-magnified) and Processed_Data/tensors_raw (raw). You'll need to generate the raw tensor set by running the Stage 1 Step-2 pipeline with EVM disabled; configs needing a missing directory are skipped with a warning rather than crashing the sweep.
Torch/NumPy aren't installed in the shell's default Python (bare Homebrew 3.14), so I couldn't execute the torch-dependent smoke tests here. In your training environment you can validate quickly with:
python Ablation_Study/models.py                                   # all 8 toggle combos → [2,3]
python Ablation_Study/run_ablation_experiments.py --epochs 1 --max_samples 16   # plumbing smoke test
Per your instruction I kept this strictly to the refactor; I didn't touch the existing Stage 3/4 code.