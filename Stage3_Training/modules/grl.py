"""
grl.py — Gradient Reversal Layer (GRL)
=======================================

Implements the Gradient Reversal Layer from:

    Ganin, Y., & Lempitsky, V. (2015).
    "Unsupervised Domain Adaptation by Backpropagation."
    International Conference on Machine Learning (ICML).

Theory:
    During the **forward pass**, the GRL acts as an identity function:

        GRL(x) = x

    During the **backward pass**, it multiplies the incoming gradient
    by a negative scalar ``-λ`` (lambda):

        ∂L/∂x = -λ · ∂L/∂GRL(x)

    This reversal penalizes the upstream encoder (Transformer) for
    retaining features that help the Identity Head.  Over training,
    the encoder learns to produce representations that are rich in
    emotion information but invariant to subject identity.

Scheduling:
    λ is typically annealed from 0 → 1 over training using a sigmoid
    schedule (Ganin et al., 2016):

        λ(p) = 2 / (1 + exp(-γ · p)) - 1

    where p ∈ [0, 1] is the training progress and γ = 10 is a common
    choice.  The ``GradientReversalLayer`` module supports dynamic
    λ updates via ``set_lambda()``.

Author  : Addhyan
Stage   : 3 — Adversarial Domain Adaptation
"""

from __future__ import annotations

import math
from typing import Tuple, Any

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    Custom autograd Function implementing gradient reversal.

    This is the core mathematical primitive.  The forward pass is a
    no-op (identity), while the backward pass multiplies the gradient
    by ``-lambda_val``.

    Usage:
        This should NOT be called directly.  Use the
        ``GradientReversalLayer`` nn.Module wrapper instead.

    Notes:
        We use ``@staticmethod`` and ``ctx.save_for_backward`` per
        PyTorch best practices for custom autograd Functions.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        lambda_val: float,
    ) -> torch.Tensor:
        """
        Forward: Identity function.

        Parameters
        ----------
        ctx : autograd context
            Used to stash lambda_val for the backward pass.
        x : torch.Tensor
            Input tensor of arbitrary shape.
        lambda_val : float
            The reversal coefficient λ.

        Returns
        -------
        torch.Tensor
            ``x`` unchanged (identity, same storage).
        """
        # Store lambda as a Python float via ctx (not saved as tensor
        # since we don't need gradients w.r.t. lambda itself).
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        """
        Backward: Negate and scale the gradient.

        The returned gradient is:  ``-lambda_val * grad_output``

        Parameters
        ----------
        ctx : autograd context
            Contains ``lambda_val`` stashed during forward.
        grad_output : torch.Tensor
            Incoming gradient from downstream layers (Identity Head).

        Returns
        -------
        tuple of (torch.Tensor, None)
            Gradient w.r.t. ``x`` (reversed) and ``None`` for ``lambda_val``
            (non-differentiable scalar).
        """
        lambda_val = ctx.lambda_val
        # Multiply by negative lambda — this is the gradient reversal
        grad_input = -lambda_val * grad_output
        return grad_input, None


class GradientReversalLayer(nn.Module):
    """
    nn.Module wrapper around ``GradientReversalFunction``.

    This provides a clean interface for use in ``nn.Sequential`` or
    as a standalone layer in the adversarial wrapper.

    Parameters
    ----------
    initial_lambda : float
        Starting value for the reversal coefficient λ.
        Default: 1.0 (full reversal from the start).

    Attributes
    ----------
    lambda_val : float
        Current reversal coefficient, modifiable via ``set_lambda()``
        or ``schedule_lambda()``.

    Example
    -------
    >>> grl = GradientReversalLayer(initial_lambda=0.5)
    >>> x = torch.randn(4, 96, requires_grad=True)
    >>> y = grl(x)
    >>> y.sum().backward()
    >>> # x.grad ≈ -0.5 * ones
    """

    def __init__(self, initial_lambda: float = 1.0) -> None:
        super().__init__()
        self.lambda_val: float = initial_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply gradient reversal.

        During forward: identity.
        During backward: gradient is multiplied by ``-λ``.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape ``[B, D]``.

        Returns
        -------
        torch.Tensor
            Same as input (forward pass is identity).
        """
        return GradientReversalFunction.apply(x, self.lambda_val)

    def set_lambda(self, new_lambda: float) -> None:
        """
        Manually set the reversal coefficient.

        Parameters
        ----------
        new_lambda : float
            New value for λ.
        """
        self.lambda_val = new_lambda

    def schedule_lambda(
        self,
        progress: float,
        gamma: float = 10.0,
    ) -> float:
        """
        Compute and apply the sigmoid-annealed λ schedule.

        From Ganin et al. (2016)::

            λ(p) = 2 / (1 + exp(-γ · p)) - 1

        where ``p ∈ [0, 1]`` is the training progress (current_epoch / total_epochs).

        This starts λ near 0 early in training (letting the encoder
        stabilize) and ramps to ~1.0 by convergence.

        Parameters
        ----------
        progress : float
            Training progress in ``[0, 1]``.
        gamma : float
            Steepness of the sigmoid curve.  Default: 10.0.

        Returns
        -------
        float
            The new lambda value.
        """
        new_lambda = 2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0
        self.lambda_val = new_lambda
        return new_lambda

    def extra_repr(self) -> str:
        """String representation for print(model)."""
        return f"lambda_val={self.lambda_val:.4f}"


# ─────────────────────────────────────────────────────────────────────
# Standalone Verification
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  Gradient Reversal Layer — Standalone Verification")
    print("=" * 70)

    # ── Test 1: Forward pass is identity ────────────────────────────
    grl = GradientReversalLayer(initial_lambda=1.0)
    x = torch.randn(4, 96, requires_grad=True)
    y = grl(x)

    assert torch.allclose(x, y), "Forward pass should be identity!"
    print("✓ Forward pass is identity")

    # ── Test 2: Backward pass reverses gradient ─────────────────────
    loss = y.sum()
    loss.backward()

    expected_grad = -1.0 * torch.ones_like(x)
    assert torch.allclose(x.grad, expected_grad, atol=1e-6), (
        f"Gradient should be -1.0, got {x.grad.mean():.4f}"
    )
    print("✓ Backward pass reverses gradient (λ=1.0)")

    # ── Test 3: Lambda scheduling ───────────────────────────────────
    grl2 = GradientReversalLayer(initial_lambda=0.0)
    schedule_values = []
    for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
        lam = grl2.schedule_lambda(p, gamma=10.0)
        schedule_values.append((p, lam))
        print(f"  progress={p:.2f}  →  λ={lam:.4f}")

    # λ should be monotonically increasing
    for i in range(1, len(schedule_values)):
        assert schedule_values[i][1] >= schedule_values[i - 1][1], (
            "Lambda schedule should be monotonically increasing!"
        )
    print("✓ Lambda schedule is monotonically increasing")

    # ── Test 4: Lambda = 0 means no gradient ────────────────────────
    grl3 = GradientReversalLayer(initial_lambda=0.0)
    x3 = torch.randn(2, 96, requires_grad=True)
    y3 = grl3(x3)
    y3.sum().backward()
    assert torch.allclose(x3.grad, torch.zeros_like(x3), atol=1e-6), (
        "λ=0 should zero out gradients!"
    )
    print("✓ λ=0 correctly zeros gradients")

    print(f"\n{'=' * 70}")
    print(f"  ✓ ALL GRL TESTS PASSED")
    print(f"{'=' * 70}")
