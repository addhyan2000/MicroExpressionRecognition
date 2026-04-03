"""
core — Training loop and orchestration for Stage 3.

Exports:
    AdversarialTrainer : Full training loop with checkpointing and logging.
"""

from .trainer import AdversarialTrainer

__all__ = ["AdversarialTrainer"]
