# STSTNet: Shallow Triple Stream Three-Dimensional CNN for Micro-Expression Recognition

This repository contains the official implementation of the paper **["Shallow Triple Stream Three-dimensional CNN (STSTNet) for Micro-expression Recognition"](https://arxiv.org/abs/1902.04352)** by Sze-Teng Liong, Y.S. Gan, John See, Huai-Qian Khor, and Yen-Chang Huang.

## Project Overview

Micro-expressions (MEs) are involuntary, brief facial expressions that occur when people try to conceal their genuine emotions. Recognizing them is a challenging task due to their short duration and subtle intensity. 

**STSTNet** introduces a lightweight and highly efficient mechanism for micro-expression recognition. Instead of relying on deep complex architectures, the network leverages three distinct input channels computed from video clips:
1. **Horizontal Optical Flow (u-component)**
2. **Vertical Optical Flow (v-component)**
3. **Optical Strain (OS)**

By feeding these dynamic features from the onset and apex frames into three parallel lightweight convolutional neural network streams, the features are rich enough to attain state-of-the-art results across composite datasets composed of 3 emotion classes (**Positive**, **Negative**, and **Surprise**). 

The experimental methodology rigorously employs the **Leave-One-Subject-Out Cross-Validation (LOSOCV)** protocol, ensuring the network is evaluated impartially by unseen subject faces from the combined CASME II, SMIC, and SAMM databases (442 target videos, 68 subjects).

## File Architecture and Description

Beneath is the detailed breakdown of every component and file in this codebase, clarifying its function and necessity.

### Source Code
- **`main.py`**: The fully independent Python implementation using PyTorch framework. 
  - *Purpose*: Handles deep learning routines including defining the three-stream `STSTNet` module, preparing training and test dataloaders from the local dataset folder `norm_u_v_os`, executing Leave-One-Subject-Out Cross Validation, training the model using the Adam optimizer and CrossEntropyLoss, calculating unweighted F1 (UF1) and unweighted average recall (UAR) metrics, and saving out intermediate weight matrices.
- **`main.m`**: The original MATLAB realization of STSTNet.
  - *Purpose*: Implements the paper's framework leveraging MATLAB's Deep Learning Toolbox. It recursively iterates over the 68 subjects, applies `imageDatastore` to load precomputed inputs (`u_train`, `u_test`), trains the network parsed from `STSTNet.mat`, and registers accuracy.

### Configuration & Data Structures
- **`STSTNet.mat`**: MATLAB Binary Data Model. 
  - *Purpose*: Pre-bundles the actual layer definitions, parameters, and architecture layout required by Deep Learning Toolbox to construct the model natively inside `main.m`.
- **`video442subName.txt`**: Identifier Listing.
  - *Purpose*: Contains precisely 68 subjects (`sub01` ... `sub26`, `s01` ... `s20`, `006` ... `037`). Its purpose is to guide the LOSOCV loops in both MATLAB and Python environments defining which subject corresponds to the "Leave-One-Out" test set dynamically.
- **`requirements.txt`**: Python dependencies list.
  - *Purpose*: Declares the exact environment libraries needed to reliably map the PyTorch execution such as `torch`, `opencv-contrib-python-headless`, `scikit-learn`, etc.

### Documentation & Reporting
- **`STSTNet_arXiv Paper.pdf`**: The authored academic manuscript.
  - *Purpose*: Provides the underlying theory, mathematical formulas (optical strain calculations), architectural reasoning, and evaluation datasets (CASME II, SMIC, SAMM) validation. 
- **`STSTNet_logfile.txt`**: Ground Truth & Legacy Mapping.
  - *Purpose*: A comprehensive dataset log spanning subject names alongside particular video IDs and internal class targets. Provides traceability into precisely what inputs and outputs mapped structurally for reproducibility audits.
- **`STSTNet_python_logfile.txt`**: The Output Output Evaluation Matrix.
  - *Purpose*: Shows logging output straight from a continuous run of `main.py`. Validates the PyTorch reproduction scores reaching an aggregate Unweighted F1 of ~0.7209 and Unweighted Average Recall (UAR) of ~0.7250.
- **`README.md`**: Guide document.
  - *Purpose*: Instructs users contextually on how to orchestrate the pipeline from downloading data, structuring folders, and training the code.

### Directories
- **`STSTNet_Weights/`**: Stored Weight Directory.
  - *Purpose*: During the execution of `main.py`, as each fold of the LOSOCV completes, it dumps the subject-specific optimized neural network weights into this directory (`.pth` extensions) preventing model training loss and enabling re-import evaluation via `--train False`.

## Mathematical & Architectural Mechanics
STSTNet employs **Shallow Layers** explicitly to resist overfitting, a common plague in domains heavily starved for data like Micro-Expression.
`Conv2d (In=3, Out=3)` -> `BatchNorm2D` -> `ReLU` -> `MaxPool2D` -> `Dropout(0.5)`
`Conv2d (In=3, Out=5)` -> `BatchNorm2D` -> `ReLU` -> `MaxPool2D` -> `Dropout(0.5)`
`Conv2d (In=3, Out=8)` -> `BatchNorm2D` -> `ReLU` -> `MaxPool2D` -> `Dropout(0.5)`
The three tensors are concatenated dimensionally into `5x5x16`, flattened via Average Pooling, and fed to a dense layer with an output of size 3 (Negative, Positive, Surprise).

## Running the Code

### PyTorch Setup
1. Download the composite inputs (Optical flows & strain images) [from the provided repository link](https://bit.ly/2S35u05) and place it under a directory named `norm_u_v_os`.
2. Construct your Python Environment:
   ```bash
   pip install -r requirements.txt
   ```
3. Begin execution tracking metrics across all 68 subjects:
   ```bash
   python main.py --train True
   ```
   If testing inference solely on pre-generated models saved in `STSTNet_Weights/`:
   ```bash
   python main.py --train False
   ```

### MATLAB Setup
1. Verify presence of **Deep Learning Toolbox**, **Parallel Computing Toolbox**, and **Computer Vision System Toolbox**.
2. Organize inputs identically to Python, however placed within an `input` directory formatted explicitly for `imageDatastore`.
3. Open and run `main.m`, tuning parameters directly in script variables.

## Citations
If making use of STSTNet components or modifications, cite the paper:
```bibtex
@inproceedings{liong2019shallow,
  title={Shallow triple stream three-dimensional cnn (ststnet) for micro-expression recognition},
  author={Liong, Sze-Teng and Gan, YS and See, John and Khor, Huai-Qian and Huang, Yen-Chang},
  booktitle={2019 14th IEEE International Conference on Automatic Face \& Gesture Recognition (FG 2019)},
  pages={1--5},
  year={2019},
  organization={IEEE}
}
```
