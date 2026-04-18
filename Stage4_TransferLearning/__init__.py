"""
Stage4_TransferLearning — Macro → Micro Transfer Learning Pipeline
====================================================================

This package implements Stage 4 of the Micro-Expression Recognition thesis:

    Two-phase transfer learning to overcome the data starvation wall.

    Phase 1: Pre-train on Macro-expressions (adversary OFF) to learn
             general facial motion dynamics and expression physics.

    Phase 2: Fine-tune on Micro-expressions (adversary ON) to achieve
             identity-invariant, SOTA micro-expression recognition.

All model, trainer, loss, and dataset classes are imported from Stage 3
to enforce strict zero-duplication of core logic.

Author  : Addhyan
Stage   : 4 — Macro-to-Micro Transfer Learning
"""
