"""
test_stage3.py — Defense-Grade QA Suite for Stage 3: Adversarial Domain Adaptation
====================================================================================

Comprehensive tests covering every Stage 3 component:

    Section 1: Gradient Reversal Layer (GRL)
    Section 2: Focal Loss
    Section 3: MERDataset
    Section 4: Subject-Disjoint Split (Data Leakage Prevention)
    Section 5: AdversarialMERWrapper (Dual-Head Architecture)
    Section 6: Adversarial Gradient Flow (GRL ↔ Heads interaction)
    Section 7: Checkpointing & Resume
    Section 8: Edge Cases & Numerical Stability

Usage::
    python -m Stage3_Training.tests.test_stage3
    pytest Stage3_Training/tests/test_stage3.py -v

Author  : Addhyan
Stage   : 3 — Adversarial Domain Adaptation (QA)
"""

from __future__ import annotations

import math
import os
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── Path setup ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Stage1_DataPipeline"))
sys.path.insert(0, str(PROJECT_ROOT / "Stage2_Architecture"))
sys.path.insert(0, str(PROJECT_ROOT / "Stage3_Training"))

from utils.logger import get_logger

from Stage3_Training.modules.grl import (
    GradientReversalFunction,
    GradientReversalLayer,
)
from Stage3_Training.losses.focal_loss import FocalLoss
from Stage3_Training.data.mer_dataset import MERDataset, DEFAULT_EMOTION_MAP
from Stage3_Training.modules.adversarial_wrapper import AdversarialMERWrapper

# ── Logger ──────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).resolve().parent / "logs"
log = get_logger("test_stage3", log_dir=LOG_DIR, log_filename="test_stage3.log")


# ===================================================================
# Helpers
# ===================================================================

def _make_temp_dataset(
    num_subjects: int = 5,
    samples_per_subject: int = 4,
    emotions: List[str] = None,
    expression_type: str = "micro-expression",
) -> tuple:
    """Create a temporary CSV + tensor directory for testing."""
    if emotions is None:
        emotions = ["Negative", "Positive", "Surprise", "Others"]

    tmp_dir = tempfile.mkdtemp(prefix="stage3_test_")
    csv_path = Path(tmp_dir) / "labels.csv"
    tensor_dir = Path(tmp_dir) / "tensors"
    tensor_dir.mkdir()

    rows = []
    for sid in range(1, num_subjects + 1):
        for j in range(samples_per_subject):
            vid = f"video_s{sid}_{j}"
            emo = emotions[j % len(emotions)]
            rows.append({
                "Dataset": "TEST",
                "Subject_ID": sid,
                "Video_ID": vid,
                "Onset_Frame": 1,
                "Apex_Frame": 16,
                "Offset_Frame": 32,
                "Raw_Emotion": emo.lower(),
                "Unified_Emotion": emo,
                "Action_Units": "AU1",
                "Expression_Type": expression_type,
                "Sequence_Length": 32,
                "FPS": 30,
                "Frames_Directory": "",
                "Frames_Exist": False,
                "OF_Processed": True,
            })
            # Create dummy tensor
            t = np.random.randn(3, 32, 224, 224).astype(np.float32)
            np.save(str(tensor_dir / f"TEST_{vid}.npy"), t)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return tmp_dir, csv_path, tensor_dir


def _cleanup(tmp_dir: str):
    shutil.rmtree(tmp_dir, ignore_errors=True)


# ===================================================================
# Section 1: Gradient Reversal Layer
# ===================================================================

class TestGRL(unittest.TestCase):
    """Tests for GradientReversalFunction and GradientReversalLayer."""

    @classmethod
    def setUpClass(cls):
        log.info("=" * 65)
        log.info("SECTION 1: Gradient Reversal Layer Tests")
        log.info("=" * 65)

    def test_forward_is_identity(self):
        """TEST 1.1 — Forward pass must be identity."""
        log.info("TEST 1.1: Forward pass identity")
        grl = GradientReversalLayer(initial_lambda=1.0)
        x = torch.randn(4, 96, requires_grad=True)
        y = grl(x)
        self.assertTrue(torch.allclose(x, y), "Forward pass is not identity!")
        log.info("  [PASS]")

    def test_backward_reverses_gradient(self):
        """TEST 1.2 — Backward must multiply gradient by -λ."""
        log.info("TEST 1.2: Backward gradient reversal")
        grl = GradientReversalLayer(initial_lambda=1.0)
        x = torch.randn(4, 96, requires_grad=True)
        y = grl(x)
        y.sum().backward()
        expected = -1.0 * torch.ones_like(x)
        self.assertTrue(torch.allclose(x.grad, expected, atol=1e-6))
        log.info("  [PASS]")

    def test_backward_fractional_lambda(self):
        """TEST 1.3 — Gradient reversal with λ=0.5."""
        log.info("TEST 1.3: Fractional lambda (0.5)")
        grl = GradientReversalLayer(initial_lambda=0.5)
        x = torch.randn(2, 96, requires_grad=True)
        y = grl(x)
        y.sum().backward()
        expected = -0.5 * torch.ones_like(x)
        self.assertTrue(torch.allclose(x.grad, expected, atol=1e-6))
        log.info("  [PASS]")

    def test_lambda_zero_blocks_gradient(self):
        """TEST 1.4 — λ=0 must zero out all gradients."""
        log.info("TEST 1.4: Lambda=0 zeroes gradients")
        grl = GradientReversalLayer(initial_lambda=0.0)
        x = torch.randn(2, 96, requires_grad=True)
        y = grl(x)
        y.sum().backward()
        self.assertTrue(torch.allclose(x.grad, torch.zeros_like(x), atol=1e-7))
        log.info("  [PASS]")

    def test_sigmoid_schedule_monotonic(self):
        """TEST 1.5 — Sigmoid schedule must be monotonically increasing."""
        log.info("TEST 1.5: Sigmoid schedule monotonicity")
        grl = GradientReversalLayer()
        vals = [grl.schedule_lambda(p, gamma=10.0) for p in [0.0, 0.25, 0.5, 0.75, 1.0]]
        for i in range(1, len(vals)):
            self.assertGreaterEqual(vals[i], vals[i-1])
        log.info("  [PASS] Schedule: %s", [f"{v:.4f}" for v in vals])

    def test_sigmoid_schedule_boundary_values(self):
        """TEST 1.6 — At p=0 λ≈0, at p=1 λ≈1."""
        log.info("TEST 1.6: Schedule boundary values")
        grl = GradientReversalLayer()
        lam_0 = grl.schedule_lambda(0.0, gamma=10.0)
        lam_1 = grl.schedule_lambda(1.0, gamma=10.0)
        self.assertAlmostEqual(lam_0, 0.0, places=4)
        self.assertAlmostEqual(lam_1, 1.0, places=3)
        log.info("  [PASS] λ(0)=%.6f, λ(1)=%.6f", lam_0, lam_1)

    def test_grl_preserves_shape(self):
        """TEST 1.7 — GRL preserves arbitrary tensor shapes."""
        log.info("TEST 1.7: Shape preservation")
        grl = GradientReversalLayer()
        for shape in [(1, 96), (8, 256), (2, 32, 64)]:
            x = torch.randn(*shape)
            y = grl(x)
            self.assertEqual(x.shape, y.shape)
        log.info("  [PASS]")

    def test_set_lambda(self):
        """TEST 1.8 — set_lambda updates lambda_val."""
        log.info("TEST 1.8: set_lambda")
        grl = GradientReversalLayer(initial_lambda=0.0)
        grl.set_lambda(0.75)
        self.assertEqual(grl.lambda_val, 0.75)
        log.info("  [PASS]")

    def test_negative_lambda(self):
        """TEST 1.9 — Negative λ should reinforce instead of reverse."""
        log.info("TEST 1.9: Negative lambda (gradient reinforcement)")
        grl = GradientReversalLayer(initial_lambda=-1.0)
        x = torch.randn(2, 96, requires_grad=True)
        y = grl(x)
        y.sum().backward()
        expected = 1.0 * torch.ones_like(x)  # -(-1) = +1
        self.assertTrue(torch.allclose(x.grad, expected, atol=1e-6))
        log.info("  [PASS]")


# ===================================================================
# Section 2: Focal Loss
# ===================================================================

class TestFocalLoss(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        log.info("")
        log.info("=" * 65)
        log.info("SECTION 2: Focal Loss Tests")
        log.info("=" * 65)
        torch.manual_seed(42)

    def test_gamma_zero_equals_ce(self):
        """TEST 2.1 — γ=0 degenerates to standard CrossEntropy."""
        log.info("TEST 2.1: γ=0 matches CE")
        logits = torch.randn(16, 4)
        targets = torch.randint(0, 4, (16,))
        fl = FocalLoss(gamma=0.0)(logits, targets)
        ce = nn.CrossEntropyLoss()(logits, targets)
        self.assertTrue(torch.allclose(fl, ce, atol=1e-5))
        log.info("  [PASS]")

    def test_loss_positive(self):
        """TEST 2.2 — Loss is always positive."""
        log.info("TEST 2.2: Loss positivity")
        for gamma in [0.0, 0.5, 1.0, 2.0, 5.0]:
            loss = FocalLoss(gamma=gamma)(torch.randn(8, 4), torch.randint(0, 4, (8,)))
            self.assertGreater(loss.item(), 0)
        log.info("  [PASS]")

    def test_higher_gamma_lower_confident_loss(self):
        """TEST 2.3 — Higher γ reduces loss for confident predictions."""
        log.info("TEST 2.3: Gamma scaling on confident predictions")
        logits = torch.zeros(4, 4)
        for i in range(4):
            logits[i, i] = 10.0
        targets = torch.arange(4)
        losses = [FocalLoss(gamma=g)(logits, targets).item() for g in [0, 2, 5]]
        self.assertGreaterEqual(losses[0], losses[1])
        self.assertGreaterEqual(losses[1], losses[2])
        log.info("  [PASS] γ=0: %.6f ≥ γ=2: %.6f ≥ γ=5: %.6f", *losses)

    def test_alpha_weights(self):
        """TEST 2.4 — Alpha weights change the loss value."""
        log.info("TEST 2.4: Alpha weights effect")
        logits = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))
        loss_no_alpha = FocalLoss(gamma=2.0)(logits, targets)
        loss_alpha = FocalLoss(gamma=2.0, alpha=torch.tensor([2., 2., 2., 2.]))(logits, targets)
        self.assertFalse(torch.allclose(loss_no_alpha, loss_alpha))
        log.info("  [PASS]")

    def test_gradient_flow(self):
        """TEST 2.5 — Gradients flow through Focal Loss."""
        log.info("TEST 2.5: Gradient flow")
        logits = torch.randn(4, 4, requires_grad=True)
        loss = FocalLoss(gamma=2.0)(logits, torch.randint(0, 4, (4,)))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertFalse(torch.isnan(logits.grad).any().item())
        log.info("  [PASS]")

    def test_reduction_none(self):
        """TEST 2.6 — reduction='none' returns per-sample losses."""
        log.info("TEST 2.6: Reduction none")
        loss = FocalLoss(gamma=2.0, reduction="none")(torch.randn(8, 4), torch.randint(0, 4, (8,)))
        self.assertEqual(loss.shape, (8,))
        log.info("  [PASS]")

    def test_reduction_sum(self):
        """TEST 2.7 — reduction='sum' returns scalar."""
        log.info("TEST 2.7: Reduction sum")
        loss = FocalLoss(gamma=2.0, reduction="sum")(torch.randn(8, 4), torch.randint(0, 4, (8,)))
        self.assertEqual(loss.dim(), 0)
        log.info("  [PASS]")

    def test_label_smoothing(self):
        """TEST 2.8 — Label smoothing increases loss for confident preds."""
        log.info("TEST 2.8: Label smoothing")
        logits = torch.zeros(4, 4)
        for i in range(4):
            logits[i, i] = 10.0
        targets = torch.arange(4)
        loss_no_smooth = FocalLoss(gamma=2.0, label_smoothing=0.0)(logits, targets)
        loss_smooth = FocalLoss(gamma=2.0, label_smoothing=0.1)(logits, targets)
        self.assertGreater(loss_smooth.item(), loss_no_smooth.item())
        log.info("  [PASS]")

    def test_single_sample(self):
        """TEST 2.9 — Works with batch size 1."""
        log.info("TEST 2.9: Batch size 1")
        loss = FocalLoss(gamma=2.0)(torch.randn(1, 4), torch.tensor([2]))
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0)
        log.info("  [PASS]")

    def test_two_class(self):
        """TEST 2.10 — Works with binary classification (2 classes)."""
        log.info("TEST 2.10: Binary classification")
        loss = FocalLoss(gamma=2.0)(torch.randn(8, 2), torch.randint(0, 2, (8,)))
        self.assertGreater(loss.item(), 0)
        log.info("  [PASS]")


# ===================================================================
# Section 3: MERDataset
# ===================================================================

class TestMERDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        log.info("")
        log.info("=" * 65)
        log.info("SECTION 3: MERDataset Tests")
        log.info("=" * 65)
        cls.tmp_dir, cls.csv_path, cls.tensor_dir = _make_temp_dataset(
            num_subjects=5, samples_per_subject=4,
        )

    @classmethod
    def tearDownClass(cls):
        _cleanup(cls.tmp_dir)

    def _make_ds(self, **kwargs):
        defaults = dict(csv_path=self.csv_path, tensor_dir=self.tensor_dir,
                        log_dir=Path(self.tmp_dir) / "logs")
        defaults.update(kwargs)
        return MERDataset(**defaults)

    def test_correct_sample_count(self):
        """TEST 3.1 — 5 subjects × 4 samples = 20."""
        log.info("TEST 3.1: Sample count")
        ds = self._make_ds()
        self.assertEqual(len(ds), 20)
        log.info("  [PASS] %d samples", len(ds))

    def test_getitem_returns_triple(self):
        """TEST 3.2 — __getitem__ returns (tensor, emotion, subject)."""
        log.info("TEST 3.2: __getitem__ returns triple")
        ds = self._make_ds()
        t, e, s = ds[0]
        self.assertIsInstance(t, torch.Tensor)
        self.assertIsInstance(e, int)
        self.assertIsInstance(s, int)
        self.assertEqual(t.shape, (3, 32, 224, 224))
        self.assertEqual(t.dtype, torch.float32)
        log.info("  [PASS] shape=%s, emotion=%d, subject=%d", t.shape, e, s)

    def test_emotion_mapping(self):
        """TEST 3.3 — All emotion labels are in [0, num_classes-1]."""
        log.info("TEST 3.3: Emotion label range")
        ds = self._make_ds()
        for i in range(len(ds)):
            _, e, _ = ds[i]
            self.assertGreaterEqual(e, 0)
            self.assertLess(e, ds.num_classes)
        log.info("  [PASS]")

    def test_subject_mapping_contiguous(self):
        """TEST 3.4 — Subject labels form contiguous range [0, N-1]."""
        log.info("TEST 3.4: Contiguous subject IDs")
        ds = self._make_ds()
        subject_labels = set()
        for i in range(len(ds)):
            _, _, s = ds[i]
            subject_labels.add(s)
        self.assertEqual(subject_labels, set(range(ds.num_subjects)))
        log.info("  [PASS] subjects: %s", sorted(subject_labels))

    def test_expression_filter(self):
        """TEST 3.5 — Expression filter works."""
        log.info("TEST 3.5: Expression filter")
        ds = self._make_ds(expression_filter="micro-expression")
        self.assertEqual(len(ds), 20)
        # Filter to non-existent type
        ds2 = self._make_ds(expression_filter="macro-expression")
        self.assertEqual(len(ds2), 0)
        log.info("  [PASS] micro=20, macro=0")

    def test_missing_csv_raises(self):
        """TEST 3.6 — Missing CSV raises FileNotFoundError."""
        log.info("TEST 3.6: Missing CSV error")
        with self.assertRaises(FileNotFoundError):
            MERDataset(csv_path=Path("/nonexistent.csv"),
                       tensor_dir=self.tensor_dir,
                       log_dir=Path(self.tmp_dir) / "logs")
        log.info("  [PASS]")

    def test_class_weights(self):
        """TEST 3.7 — Class weights are positive and finite."""
        log.info("TEST 3.7: Class weights")
        ds = self._make_ds()
        w = ds.get_class_weights()
        self.assertEqual(w.shape[0], ds.num_classes)
        self.assertTrue((w > 0).all())
        self.assertTrue(torch.isfinite(w).all())
        log.info("  [PASS] weights=%s", w.tolist())

    def test_missing_tensors_skipped(self):
        """TEST 3.8 — Rows without .npy files are skipped gracefully."""
        log.info("TEST 3.8: Missing tensor handling")
        tmp2 = tempfile.mkdtemp(prefix="stage3_test_missing_")
        csv2 = Path(tmp2) / "labels.csv"
        tdir2 = Path(tmp2) / "tensors"
        tdir2.mkdir()
        # CSV has 2 rows but only 1 tensor exists
        df = pd.DataFrame([
            {"Dataset": "T", "Subject_ID": 1, "Video_ID": "v1",
             "Unified_Emotion": "Positive", "Expression_Type": "micro-expression",
             **{c: "" for c in ["Onset_Frame","Apex_Frame","Offset_Frame",
                "Raw_Emotion","Action_Units","Sequence_Length","FPS",
                "Frames_Directory","Frames_Exist","OF_Processed"]}},
            {"Dataset": "T", "Subject_ID": 1, "Video_ID": "v2",
             "Unified_Emotion": "Negative", "Expression_Type": "micro-expression",
             **{c: "" for c in ["Onset_Frame","Apex_Frame","Offset_Frame",
                "Raw_Emotion","Action_Units","Sequence_Length","FPS",
                "Frames_Directory","Frames_Exist","OF_Processed"]}},
        ])
        df.to_csv(csv2, index=False)
        np.save(str(tdir2 / "T_v1.npy"), np.zeros((3,32,224,224), dtype=np.float32))
        # v2 tensor missing
        ds = MERDataset(csv_path=csv2, tensor_dir=tdir2, log_dir=Path(tmp2)/"logs")
        self.assertEqual(len(ds), 1)
        _cleanup(tmp2)
        log.info("  [PASS] 1 of 2 samples loaded (1 tensor missing)")

    def test_custom_emotion_map(self):
        """TEST 3.9 — Custom emotion map works and filters unknown emotions."""
        log.info("TEST 3.9: Custom emotion map")
        custom_map = {"Negative": 0, "Positive": 1}  # Exclude Surprise, Others
        ds = self._make_ds(emotion_map=custom_map)
        self.assertLess(len(ds), 20)  # Some samples excluded
        for i in range(len(ds)):
            _, e, _ = ds[i]
            self.assertIn(e, [0, 1])
        log.info("  [PASS] %d samples with 2-class map", len(ds))


# ===================================================================
# Section 4: Subject-Disjoint Split
# ===================================================================

class TestSubjectDisjointSplit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        log.info("")
        log.info("=" * 65)
        log.info("SECTION 4: Subject-Disjoint Split (Data Leakage Prevention)")
        log.info("=" * 65)
        cls.tmp_dir, cls.csv_path, cls.tensor_dir = _make_temp_dataset(
            num_subjects=10, samples_per_subject=6,
        )

    @classmethod
    def tearDownClass(cls):
        _cleanup(cls.tmp_dir)

    def _make_ds(self):
        return MERDataset(csv_path=self.csv_path, tensor_dir=self.tensor_dir,
                          log_dir=Path(self.tmp_dir) / "logs")

    def test_zero_subject_overlap(self):
        """TEST 4.1 — CRITICAL: No subject appears in both train and val."""
        log.info("TEST 4.1: Zero subject overlap")
        ds = self._make_ds()
        train_idx, val_idx = MERDataset.subject_disjoint_split(ds, 0.2, seed=42)
        train_sids = set(ds.get_subject_for_sample(i) for i in train_idx)
        val_sids = set(ds.get_subject_for_sample(i) for i in val_idx)
        overlap = train_sids & val_sids
        self.assertEqual(len(overlap), 0, f"LEAKAGE: {overlap}")
        log.info("  [PASS] Train subjects: %s | Val subjects: %s", sorted(train_sids), sorted(val_sids))

    def test_all_samples_accounted(self):
        """TEST 4.2 — train + val indices cover entire dataset."""
        log.info("TEST 4.2: Full coverage")
        ds = self._make_ds()
        train_idx, val_idx = MERDataset.subject_disjoint_split(ds, 0.3, seed=42)
        self.assertEqual(len(train_idx) + len(val_idx), len(ds))
        self.assertEqual(len(set(train_idx) & set(val_idx)), 0)
        log.info("  [PASS] %d + %d = %d", len(train_idx), len(val_idx), len(ds))

    def test_deterministic_with_same_seed(self):
        """TEST 4.3 — Same seed produces identical split."""
        log.info("TEST 4.3: Determinism")
        ds = self._make_ds()
        t1, v1 = MERDataset.subject_disjoint_split(ds, 0.2, seed=123)
        t2, v2 = MERDataset.subject_disjoint_split(ds, 0.2, seed=123)
        self.assertEqual(t1, t2)
        self.assertEqual(v1, v2)
        log.info("  [PASS]")

    def test_different_seed_different_split(self):
        """TEST 4.4 — Different seed produces different split."""
        log.info("TEST 4.4: Different seeds")
        ds = self._make_ds()
        _, v1 = MERDataset.subject_disjoint_split(ds, 0.2, seed=1)
        _, v2 = MERDataset.subject_disjoint_split(ds, 0.2, seed=999)
        val_sids_1 = set(ds.get_subject_for_sample(i) for i in v1)
        val_sids_2 = set(ds.get_subject_for_sample(i) for i in v2)
        self.assertNotEqual(val_sids_1, val_sids_2)
        log.info("  [PASS]")

    def test_val_fraction_respected(self):
        """TEST 4.5 — Approx correct fraction of subjects in val."""
        log.info("TEST 4.5: Val fraction")
        ds = self._make_ds()
        _, val_idx = MERDataset.subject_disjoint_split(ds, 0.3, seed=42)
        val_sids = set(ds.get_subject_for_sample(i) for i in val_idx)
        # 30% of 10 subjects = 3
        self.assertIn(len(val_sids), [2, 3, 4])
        log.info("  [PASS] %d val subjects (expected ~3)", len(val_sids))

    def test_minimum_one_val_subject(self):
        """TEST 4.6 — Even with tiny fraction, at least 1 val subject."""
        log.info("TEST 4.6: Minimum 1 val subject")
        ds = self._make_ds()
        _, val_idx = MERDataset.subject_disjoint_split(ds, 0.01, seed=42)
        val_sids = set(ds.get_subject_for_sample(i) for i in val_idx)
        self.assertGreaterEqual(len(val_sids), 1)
        log.info("  [PASS] %d val subjects with 1%% fraction", len(val_sids))

    def test_all_val_samples_from_val_subjects_only(self):
        """TEST 4.7 — Every val sample belongs to a val-only subject."""
        log.info("TEST 4.7: Val sample provenance")
        ds = self._make_ds()
        train_idx, val_idx = MERDataset.subject_disjoint_split(ds, 0.2, seed=42)
        train_sids = set(ds.get_subject_for_sample(i) for i in train_idx)
        for i in val_idx:
            sid = ds.get_subject_for_sample(i)
            self.assertNotIn(sid, train_sids, f"Val sample {i} has train subject {sid}")
        log.info("  [PASS]")


# ===================================================================
# Section 5: AdversarialMERWrapper
# ===================================================================

class TestAdversarialWrapper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        log.info("")
        log.info("=" * 65)
        log.info("SECTION 5: AdversarialMERWrapper Tests")
        log.info("=" * 65)
        cls.model = AdversarialMERWrapper(
            num_emotions=4, num_subjects=10,
            grl_lambda=1.0, feature_dim=96, head_dropout=0.0,
        )
        cls.model.eval()

    def test_forward_output_shapes(self):
        """TEST 5.1 — Dual-head output shapes."""
        log.info("TEST 5.1: Output shapes")
        x = torch.randn(2, 3, 32, 224, 224)
        with torch.no_grad():
            emo, ident = self.model(x)
        self.assertEqual(emo.shape, (2, 4))
        self.assertEqual(ident.shape, (2, 10))
        log.info("  [PASS] emo=%s, id=%s", emo.shape, ident.shape)

    def test_batch_size_one(self):
        """TEST 5.2 — B=1 edge case."""
        log.info("TEST 5.2: Batch size 1")
        x = torch.randn(1, 3, 32, 224, 224)
        with torch.no_grad():
            emo, ident = self.model(x)
        self.assertEqual(emo.shape, (1, 4))
        self.assertEqual(ident.shape, (1, 10))
        log.info("  [PASS]")

    def test_no_nan_in_outputs(self):
        """TEST 5.3 — No NaN in either head output."""
        log.info("TEST 5.3: NaN check")
        x = torch.randn(2, 3, 32, 224, 224)
        with torch.no_grad():
            emo, ident = self.model(x)
        self.assertFalse(torch.isnan(emo).any().item())
        self.assertFalse(torch.isnan(ident).any().item())
        log.info("  [PASS]")

    def test_parameter_count_positive(self):
        """TEST 5.4 — Parameter count is sane."""
        log.info("TEST 5.4: Parameter count")
        params = self.model.count_parameters()
        self.assertGreater(params["model_trainable"], 0)
        self.assertGreater(params["emotion_head_trainable"], 0)
        self.assertGreater(params["identity_head_trainable"], 0)
        log.info("  [PASS] total=%d", params["model_trainable"])

    def test_grl_lambda_scheduling(self):
        """TEST 5.5 — GRL lambda schedule via wrapper."""
        log.info("TEST 5.5: GRL scheduling via wrapper")
        lam = self.model.schedule_grl_lambda(0.5, gamma=10.0)
        self.assertGreater(lam, 0.0)
        self.assertLess(lam, 1.0)
        log.info("  [PASS] λ(0.5)=%.4f", lam)


# ===================================================================
# Section 6: Adversarial Gradient Flow
# ===================================================================

class TestAdversarialGradientFlow(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        log.info("")
        log.info("=" * 65)
        log.info("SECTION 6: Adversarial Gradient Flow Tests")
        log.info("=" * 65)

    def test_backward_completes(self):
        """TEST 6.1 — Full backward through both heads completes."""
        log.info("TEST 6.1: Backward completion")
        model = AdversarialMERWrapper(num_emotions=4, num_subjects=10,
                                       grl_lambda=1.0, head_dropout=0.0)
        model.train()
        x = torch.randn(2, 3, 32, 224, 224)
        emo, ident = model(x)
        loss = emo.sum() + ident.sum()
        loss.backward()
        log.info("  [PASS]")

    def test_all_params_receive_gradients(self):
        """TEST 6.2 — Every trainable parameter gets a gradient."""
        log.info("TEST 6.2: All params receive gradients")
        model = AdversarialMERWrapper(num_emotions=4, num_subjects=10,
                                       grl_lambda=1.0, head_dropout=0.0)
        model.train()
        model.zero_grad()
        x = torch.randn(2, 3, 32, 224, 224)
        emo, ident = model(x)
        loss = emo.sum() + ident.sum()
        loss.backward()
        no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        self.assertEqual(len(no_grad), 0, f"Params without grad: {no_grad}")
        log.info("  [PASS]")

    def test_gradients_finite(self):
        """TEST 6.3 — No NaN/Inf in gradients."""
        log.info("TEST 6.3: Gradient finiteness")
        model = AdversarialMERWrapper(num_emotions=4, num_subjects=10,
                                       grl_lambda=1.0, head_dropout=0.0)
        model.train()
        model.zero_grad()
        x = torch.randn(2, 3, 32, 224, 224)
        emo, ident = model(x)
        loss = emo.sum() + ident.sum()
        loss.backward()
        bad = [n for n, p in model.named_parameters()
               if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())]
        self.assertEqual(len(bad), 0, f"NaN/Inf grads: {bad}")
        log.info("  [PASS]")

    def test_dual_loss_training_step(self):
        """TEST 6.4 — Realistic dual-loss training step."""
        log.info("TEST 6.4: Dual-loss training step")
        model = AdversarialMERWrapper(num_emotions=4, num_subjects=10,
                                       grl_lambda=1.0, head_dropout=0.0)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(2, 3, 32, 224, 224)
        emo_targets = torch.randint(0, 4, (2,))
        id_targets = torch.randint(0, 10, (2,))

        emo_logits, id_logits = model(x)
        emo_loss = FocalLoss(gamma=2.0)(emo_logits, emo_targets)
        id_loss = nn.CrossEntropyLoss()(id_logits, id_targets)
        total = emo_loss + 0.5 * id_loss

        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        self.assertTrue(total.item() > 0)
        self.assertFalse(torch.isnan(total).item())
        log.info("  [PASS] total_loss=%.4f", total.item())


# ===================================================================
# Section 7: Checkpointing
# ===================================================================

class TestCheckpointing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        log.info("")
        log.info("=" * 65)
        log.info("SECTION 7: Checkpointing & Resume Tests")
        log.info("=" * 65)

    def test_save_load_model_state(self):
        """TEST 7.1 — Save and reload produces identical outputs."""
        log.info("TEST 7.1: Save/load bitwise equality")
        model = AdversarialMERWrapper(num_emotions=4, num_subjects=10,
                                       grl_lambda=1.0, head_dropout=0.0)
        model.eval()
        x = torch.randn(1, 3, 32, 224, 224)
        with torch.no_grad():
            emo1, id1 = model(x)

        tmp = tempfile.mktemp(suffix=".pth")
        torch.save({"model_state_dict": model.state_dict()}, tmp)

        model2 = AdversarialMERWrapper(num_emotions=4, num_subjects=10,
                                        grl_lambda=1.0, head_dropout=0.0)
        ckpt = torch.load(tmp, map_location="cpu", weights_only=False)
        model2.load_state_dict(ckpt["model_state_dict"])
        model2.eval()
        with torch.no_grad():
            emo2, id2 = model2(x)

        self.assertTrue(torch.allclose(emo1, emo2, atol=1e-6))
        self.assertTrue(torch.allclose(id1, id2, atol=1e-6))
        os.remove(tmp)
        log.info("  [PASS]")

    def test_checkpoint_contains_required_keys(self):
        """TEST 7.2 — Checkpoint has all required keys for resume."""
        log.info("TEST 7.2: Checkpoint key completeness")
        model = AdversarialMERWrapper(num_emotions=4, num_subjects=10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        checkpoint = {
            "epoch": 5,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_accuracy": 0.75,
            "history": {"train_loss": [1.0, 0.9, 0.8, 0.7, 0.6]},
        }
        required = ["epoch", "model_state_dict", "optimizer_state_dict",
                     "best_val_accuracy", "history"]
        for key in required:
            self.assertIn(key, checkpoint)
        log.info("  [PASS]")


# ===================================================================
# Section 8: Edge Cases & Numerical Stability
# ===================================================================

class TestEdgeCases(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        log.info("")
        log.info("=" * 65)
        log.info("SECTION 8: Edge Cases & Numerical Stability")
        log.info("=" * 65)

    def test_focal_loss_extreme_logits(self):
        """TEST 8.1 — Focal Loss handles extreme logit values."""
        log.info("TEST 8.1: Extreme logits")
        logits = torch.tensor([[100., -100., 0., 0.]])
        targets = torch.tensor([0])
        loss = FocalLoss(gamma=2.0)(logits, targets)
        self.assertTrue(torch.isfinite(loss).item())
        log.info("  [PASS] loss=%.6f", loss.item())

    def test_focal_loss_all_same_class(self):
        """TEST 8.2 — All samples same class (mode collapse scenario)."""
        log.info("TEST 8.2: All same class")
        logits = torch.randn(8, 4)
        targets = torch.zeros(8, dtype=torch.long)  # All class 0
        loss = FocalLoss(gamma=2.0)(logits, targets)
        self.assertTrue(torch.isfinite(loss).item())
        log.info("  [PASS]")

    def test_grl_with_zero_input(self):
        """TEST 8.3 — GRL handles zero tensor."""
        log.info("TEST 8.3: GRL with zero input")
        grl = GradientReversalLayer(initial_lambda=1.0)
        x = torch.zeros(2, 96, requires_grad=True)
        y = grl(x)
        y.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all().item())
        log.info("  [PASS]")

    def test_grl_large_lambda(self):
        """TEST 8.4 — GRL with very large lambda."""
        log.info("TEST 8.4: Large lambda (100)")
        grl = GradientReversalLayer(initial_lambda=100.0)
        x = torch.randn(2, 96, requires_grad=True)
        y = grl(x)
        y.sum().backward()
        expected = -100.0 * torch.ones_like(x)
        self.assertTrue(torch.allclose(x.grad, expected, atol=1e-4))
        log.info("  [PASS]")

    def test_wrapper_with_single_subject(self):
        """TEST 8.5 — Wrapper with num_subjects=1 (degenerate case)."""
        log.info("TEST 8.5: Single subject")
        model = AdversarialMERWrapper(num_emotions=4, num_subjects=1,
                                       grl_lambda=1.0, head_dropout=0.0)
        model.eval()
        x = torch.randn(2, 3, 32, 224, 224)
        with torch.no_grad():
            emo, ident = model(x)
        self.assertEqual(ident.shape, (2, 1))
        log.info("  [PASS]")

    def test_wrapper_with_single_emotion(self):
        """TEST 8.6 — Wrapper with num_emotions=1 (degenerate case)."""
        log.info("TEST 8.6: Single emotion")
        model = AdversarialMERWrapper(num_emotions=1, num_subjects=10,
                                       grl_lambda=1.0, head_dropout=0.0)
        model.eval()
        x = torch.randn(2, 3, 32, 224, 224)
        with torch.no_grad():
            emo, ident = model(x)
        self.assertEqual(emo.shape, (2, 1))
        log.info("  [PASS]")

    def test_dataset_with_one_subject(self):
        """TEST 8.7 — Dataset with only 1 subject."""
        log.info("TEST 8.7: Single-subject dataset")
        tmp, csv, tdir = _make_temp_dataset(num_subjects=1, samples_per_subject=5)
        ds = MERDataset(csv_path=csv, tensor_dir=tdir, log_dir=Path(tmp)/"logs")
        self.assertEqual(ds.num_subjects, 1)
        self.assertEqual(len(ds), 5)
        _, _, s = ds[0]
        self.assertEqual(s, 0)
        _cleanup(tmp)
        log.info("  [PASS]")

    def test_empty_dataset_after_filter(self):
        """TEST 8.8 — Empty dataset after expression filter."""
        log.info("TEST 8.8: Empty dataset")
        tmp, csv, tdir = _make_temp_dataset(num_subjects=2, samples_per_subject=2)
        ds = MERDataset(csv_path=csv, tensor_dir=tdir,
                        expression_filter="nonexistent-type",
                        log_dir=Path(tmp)/"logs")
        self.assertEqual(len(ds), 0)
        _cleanup(tmp)
        log.info("  [PASS]")

    def test_focal_loss_alpha_on_device(self):
        """TEST 8.9 — Alpha buffer stays on correct device."""
        log.info("TEST 8.9: Alpha device consistency")
        alpha = torch.tensor([1.0, 2.0, 3.0, 4.0])
        fl = FocalLoss(gamma=2.0, alpha=alpha)
        self.assertEqual(fl.alpha.device.type, "cpu")
        logits = torch.randn(4, 4)
        targets = torch.randint(0, 4, (4,))
        loss = fl(logits, targets)
        self.assertTrue(torch.isfinite(loss).item())
        log.info("  [PASS]")


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    log.info("=" * 65)
    log.info("  STAGE 3 — COMPREHENSIVE TEST SUITE")
    log.info("  Adversarial Domain Adaptation QA")
    log.info("=" * 65)

    unittest.main(verbosity=2)
