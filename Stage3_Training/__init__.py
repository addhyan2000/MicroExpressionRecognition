"""
Stage3_Training — Adversarial Domain Adaptation & Training Scaffolding
=======================================================================

This package implements Stage 3 of the Micro-Expression Recognition thesis:

    Adversarial Training with Gradient Reversal to destroy Identity Bias.

The Stage 2 HybridMERModel produces a [B, 96] feature vector.  Stage 3
wraps this in an adversarial framework with two heads:

    1. Emotion Head  — Classifies micro-expression emotion (Focal Loss)
    2. Identity Head — Predicts subject ID via GRL (Cross-Entropy Loss)

The Gradient Reversal Layer (GRL) reverses gradients during backprop,
forcing the representation to become identity-invariant.

Author  : Addhyan
Stage   : 3 — Adversarial Domain Adaptation
"""
