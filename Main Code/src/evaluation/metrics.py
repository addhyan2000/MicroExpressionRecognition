"""Metrics Tracker for Micro-Expression Recognition (MER) Evaluation.

Calculates Unweighted F1-score (UF1) and Unweighted Average Recall (UAR) 
using a global, incrementally updated N x N Confusion Matrix to securely 
handle extreme class imbalances and missing instances during LOSO evaluations.
"""
import torch
import numpy as np
from typing import Tuple, Dict, List, Any


class MERMetricTracker:
    """MER Metric Tracker generating mathematically stable UF1 and UAR.
    
    Maintains dual state: a fold-wise matrix for localized evaluation and a 
    global matrix for mathematically rigorous macro metrics (averaging over all
    data, mitigating zero-instance divide-by-zero faults for rare classes).
    
    Parameters
    ----------
    num_classes : int
        The number of underlying affective classes (e.g., 3 or 5).
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        
        # N x N global confusion matrix for composite database evaluation
        self.global_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        
        # N x N localized temporal confusion matrix for the current fold
        self.fold_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        
        # Localized fold-wise tracked UAR for variance calculations
        self.fold_uars: List[float] = []

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
        """Accumulate batch predictions uniformly.
        
        Parameters
        ----------
        y_true : torch.Tensor
            Ground truth labels of shape (B,).
        y_pred : torch.Tensor
            Predicted class indices of shape (B,).
        """
        y_true_np = y_true.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()
        
        # Ensure inputs are within bounds
        valid_mask = (y_true_np >= 0) & (y_true_np < self.num_classes) & (y_pred_np >= 0) & (y_pred_np < self.num_classes)
        y_true_np = y_true_np[valid_mask]
        y_pred_np = y_pred_np[valid_mask]
        
        batch_cm = np.bincount(
            y_true_np * self.num_classes + y_pred_np, 
            minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)
        
        self.global_cm += batch_cm
        self.fold_cm += batch_cm

    def _calculate_metrics_from_cm(self, cm: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Core engine calculating Macro metrics reliably."""
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        
        precision_denom = tp + fp
        recall_denom = tp + fn
        
        # A class must be included in the Macro-Averaging if it is present 
        # in the ground truth OR if the model predicted (hallucinated) it.
        valid_classes = (recall_denom > 0) | (precision_denom > 0)
        
        if not np.any(valid_classes):
            return 0.0, 0.0, np.zeros(self.num_classes, dtype=np.float64), np.zeros(self.num_classes, dtype=np.float64)
            
        # Calculate isolated per-class Recall (Accuracy for each class)
        recalls = np.zeros(self.num_classes, dtype=np.float64)
        valid_recall_mask = recall_denom > 0
        recalls[valid_recall_mask] = tp[valid_recall_mask] / recall_denom[valid_recall_mask]
        
        # Calculate isolated per-class Precision
        precisions = np.zeros(self.num_classes, dtype=np.float64)
        valid_precision_mask = precision_denom > 0
        precisions[valid_precision_mask] = tp[valid_precision_mask] / precision_denom[valid_precision_mask]
        
        # Calculate isolated per-class F1
        f1s = np.zeros(self.num_classes, dtype=np.float64)
        f1_denom = precisions + recalls
        valid_f1_mask = f1_denom > 0
        f1s[valid_f1_mask] = 2 * (precisions[valid_f1_mask] * recalls[valid_f1_mask]) / f1_denom[valid_f1_mask]
        
        # Macro-Averaging over active participating classes directly
        uar = float(np.mean(recalls[valid_classes]))
        uf1 = float(np.mean(f1s[valid_classes]))
        
        return uf1, uar, f1s, recalls

    def compute_fold_metrics(self) -> Tuple[float, float]:
        """Calculates localized metrics, logs the fold UAR, and zeros local state."""
        uf1, uar, _, _ = self._calculate_metrics_from_cm(self.fold_cm)
        self.fold_uars.append(uar)
        
        # Zero internal state directly for the impending incoming fold
        self.fold_cm.fill(0)
        
        return uf1, uar

    def compute_global_metrics(self) -> Dict[str, Any]:
        """Calculates overarching metrics directly from the monolithic global matrix."""
        uf1, uar, f1s, recalls = self._calculate_metrics_from_cm(self.global_cm)
        
        # Compute distribution standard deviation representing fold variance (σ)
        uar_std = float(np.std(self.fold_uars)) if len(self.fold_uars) > 0 else 0.0
        
        return {
            "Global_UF1": uf1,
            "Global_UAR": uar,
            "UAR_std": uar_std,
            "Per_Class_F1": f1s.tolist(),
            "Per_Class_Recall": recalls.tolist()
        }

    def get_global_confusion_matrix(self) -> np.ndarray:
        """Exposes a secure copy of the global confusion matrix for downstream visualization."""
        return self.global_cm.copy()
