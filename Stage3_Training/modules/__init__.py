"""
modules — Neural network components for Stage 3.

Exports:
    GradientReversalFunction : Custom autograd function for GRL.
    GradientReversalLayer    : nn.Module wrapper around the GRL function.
    AdversarialMERWrapper    : Full adversarial wrapper around HybridMERModel.
"""

from .grl import GradientReversalFunction, GradientReversalLayer
from .adversarial_wrapper import AdversarialMERWrapper

__all__ = [
    "GradientReversalFunction",
    "GradientReversalLayer",
    "AdversarialMERWrapper",
]
