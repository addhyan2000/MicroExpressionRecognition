"""
eval_visualizations.py — Academic Visualization Suite for Master's Thesis
==========================================================================

Provides publication-quality plotting functions for the Master's Thesis on
Micro-Expression Recognition (MER).  Every function is self-contained and
returns a ``matplotlib.figure.Figure`` so it can be embedded directly in
``Master_Thesis_Results.ipynb`` or saved for LaTeX inclusion.

Design conventions:
    • ``sns.set_context('paper')`` with custom rcParams for consistency.
    • 300 DPI output; axes styled with spines and grid for academic look.
    • Color palettes are ColorBrewer-safe (colourblind-accessible).
    • t-SNE computed with ``sklearn``; falls back gracefully if unavailable.

Functions
---------
plot_class_distribution(emotion_dict)
    Stage 1 EDA: bar chart of emotion class counts.
plot_training_history(history_dict)
    Stage 3/4: four-panel loss + accuracy curves over epochs.
plot_confusion_matrix(y_true, y_pred, classes)
    Per-class confusion heatmap with accuracy annotations.
plot_tsne_embeddings(features, emotion_labels, subject_labels)
    Dual t-SNE scatter: coloured by emotion AND by subject.
plot_stage3_vs_stage4(stage3_acc, stage3_f1, stage4_acc, stage4_f1)
    Grouped bar chart comparing Stage 3 baseline vs Stage 4 transfer.

Author  : Addhyan
Thesis  : Subject-Invariant Spatiotemporal Contrastive Network for MER
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ─────────────────────────────────────────────────────────────────────
# Global Academic Style Configuration
# ─────────────────────────────────────────────────────────────────────
# Apply once at import time so every downstream plot inherits the style.

sns.set_context("paper", font_scale=1.3)
sns.set_style("whitegrid")

_PAPER_RC = {
    "font.family":          "DejaVu Sans",
    "axes.titlesize":       13,
    "axes.labelsize":       11,
    "xtick.labelsize":      9,
    "ytick.labelsize":      9,
    "legend.fontsize":      9,
    "legend.framealpha":    0.85,
    "figure.dpi":           150,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.grid":            True,
    "grid.alpha":           0.4,
    "grid.linestyle":       "--",
}
matplotlib.rcParams.update(_PAPER_RC)

# ── Shared palettes ──────────────────────────────────────────────────
# 4 emotion classes: Negative / Positive / Surprise / Others
_EMOTION_PALETTE = ["#E63946", "#457B9D", "#F4A261", "#2A9D8F"]
_STAGE_PALETTE   = ["#4E8098", "#C05746"]   # Stage 3 = teal, Stage 4 = rust
_CMAP_CONF       = "Blues"                   # confusion matrix cmap


# ═════════════════════════════════════════════════════════════════════
# 1. Stage 1 — Class Distribution
# ═════════════════════════════════════════════════════════════════════

def plot_class_distribution(
    emotion_dict: Dict[str, int],
    title: str = "Emotion Class Distribution — Combined CASME II & CAS(ME)²",
    figsize: tuple = (7, 4),
) -> plt.Figure:
    """
    Render a bar chart of emotion class sample counts (Stage 1 EDA).

    Parameters
    ----------
    emotion_dict : dict
        Mapping of ``{emotion_name: sample_count}``.
        Example: ``{"Negative": 320, "Positive": 74, ...}``
    title : str
        Figure title.
    figsize : tuple
        Width × height in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    labels  = list(emotion_dict.keys())
    counts  = list(emotion_dict.values())
    total   = sum(counts)
    colors  = _EMOTION_PALETTE[: len(labels)]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(labels, counts, color=colors, edgecolor="white",
                  linewidth=0.8, zorder=3, width=0.55)

    # Annotate each bar with count and percentage
    for bar, count in zip(bars, counts):
        pct = 100.0 * count / max(total, 1)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.008,
            f"{count}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold",
        )

    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("Unified Emotion Category", labelpad=8)
    ax.set_ylabel("Number of Samples", labelpad=8)
    ax.set_ylim(0, max(counts) * 1.22)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Subtle horizontal reference lines
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)

    # Annotation: total
    ax.text(
        0.98, 0.96, f"Total samples: {total}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8.5, color="#555555",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.8),
    )

    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════
# 2. Training History (Loss + Accuracy Curves)
# ═════════════════════════════════════════════════════════════════════

def plot_training_history(
    history_dict: Dict[str, List[float]],
    phase_label: str = "Stage 4 — Transfer Learning",
    figsize: tuple = (13, 9),
) -> plt.Figure:
    """
    Four-panel training history: SupCon Loss, Emotion Loss,
    Identity Loss, and Emotion Accuracy (train vs val).

    Parameters
    ----------
    history_dict : dict
        Keys expected (subset used gracefully if missing):
            ``train_supcon_loss``, ``train_emotion_loss``,
            ``train_identity_loss``, ``train_emotion_acc``,
            ``val_supcon_loss``,   ``val_loss``,
            ``val_emotion_acc``,   ``grl_lambda``
    phase_label : str
        Shown as the figure super-title.
    figsize : tuple
        Width × height in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    def _get(key: str) -> list:
        return history_dict.get(key, [])

    epochs = range(1, max(
        len(_get("train_emotion_loss")),
        len(_get("train_supcon_loss")), 1
    ) + 1)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(phase_label, fontsize=14, fontweight="bold", y=1.01)

    # ── Panel 1: SupCon Loss ─────────────────────────────────────────
    ax = axes[0, 0]
    tr_sc = _get("train_supcon_loss")
    vl_sc = _get("val_supcon_loss")
    if tr_sc:
        ax.plot(range(1, len(tr_sc)+1), tr_sc, color="#457B9D",
                lw=1.8, label="Train SupCon")
    if vl_sc:
        ax.plot(range(1, len(vl_sc)+1), vl_sc, color="#457B9D",
                lw=1.8, ls="--", alpha=0.7, label="Val SupCon")
    ax.set_title("Supervised Contrastive Loss (SupCon)", fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    if tr_sc or vl_sc: ax.legend()

    # ── Panel 2: Emotion (Focal) Loss ───────────────────────────────
    ax = axes[0, 1]
    tr_em = _get("train_emotion_loss")
    if tr_em:
        ax.plot(range(1, len(tr_em)+1), tr_em, color="#E63946",
                lw=1.8, label="Train Focal Loss")
    ax.set_title("Emotion Classification Loss (Focal)", fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    if tr_em: ax.legend()

    # ── Panel 3: Identity (Adversarial) Loss ────────────────────────
    ax = axes[1, 0]
    tr_id = _get("train_identity_loss")
    if tr_id:
        ax.plot(range(1, len(tr_id)+1), tr_id, color="#F4A261",
                lw=1.8, label="Train Identity Loss (GRL)")
    # Overlay GRL lambda on secondary axis
    grl_lam = _get("grl_lambda")
    if grl_lam:
        ax2 = ax.twinx()
        ax2.plot(range(1, len(grl_lam)+1), grl_lam, color="#2A9D8F",
                 lw=1.2, ls=":", alpha=0.7, label="GRL λ")
        ax2.set_ylabel("GRL λ", color="#2A9D8F", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="#2A9D8F")
        ax2.set_ylim(-0.05, 1.1)
    ax.set_title("Identity Loss + GRL λ Schedule", fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    if tr_id: ax.legend(loc="upper right")

    # ── Panel 4: Emotion Accuracy ────────────────────────────────────
    ax = axes[1, 1]
    tr_acc = _get("train_emotion_acc")
    vl_acc = _get("val_emotion_acc")
    if tr_acc:
        ax.plot(range(1, len(tr_acc)+1), [v*100 for v in tr_acc],
                color="#2A9D8F", lw=1.8, label="Train Acc (%)")
    if vl_acc:
        ax.plot(range(1, len(vl_acc)+1), [v*100 for v in vl_acc],
                color="#2A9D8F", lw=1.8, ls="--", alpha=0.7, label="Val Acc (%)")
    ax.set_title("Emotion Classification Accuracy", fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    if tr_acc or vl_acc: ax.legend()

    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════
# 3. Confusion Matrix
# ═════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    classes: List[str],
    normalize: bool = True,
    title: str = "Confusion Matrix",
    figsize: tuple = (6, 5),
) -> plt.Figure:
    """
    Generate a publication-quality confusion matrix heatmap.

    Parameters
    ----------
    y_true : array-like
        Ground-truth integer emotion labels.
    y_pred : array-like
        Predicted integer emotion labels.
    classes : list of str
        Class names in label-index order (e.g. ["Negative","Positive",...]).
    normalize : bool
        If ``True``, normalise each row to [0, 1] (recall per class).
    title : str
        Plot title.
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    if normalize:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        fmt = ".2f"
        cbar_label = "Recall (row-normalised)"
    else:
        cm_plot = cm
        fmt = "d"
        cbar_label = "Count"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_plot,
        annot=True, fmt=fmt,
        cmap=_CMAP_CONF,
        xticklabels=classes,
        yticklabels=classes,
        linewidths=0.5,
        linecolor="#dddddd",
        cbar_kws={"label": cbar_label, "shrink": 0.8},
        ax=ax,
    )

    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", labelpad=8)
    ax.set_ylabel("True Label", labelpad=8)
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    # Overall accuracy annotation
    overall_acc = np.mean(np.array(y_true) == np.array(y_pred)) * 100
    ax.text(
        0.98, -0.12, f"Overall Accuracy: {overall_acc:.2f}%",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, style="italic", color="#333333",
    )

    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════
# 4. Dual t-SNE Embeddings (by Emotion & by Subject)
# ═════════════════════════════════════════════════════════════════════

def plot_tsne_embeddings(
    features: np.ndarray,
    emotion_labels: Sequence[int],
    subject_labels: Sequence[int],
    class_names: Optional[List[str]] = None,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    random_state: int = 42,
    figsize: tuple = (13, 5.5),
    title_prefix: str = "Stage 4 Projection Head Embeddings",
) -> plt.Figure:
    """
    Compute 2-D t-SNE of 64-d projection head features and render
    **two side-by-side scatter plots**:

    * **Plot A** — coloured by Emotion (validates SupCon clustering).
    * **Plot B** — coloured by Subject ID (validates GRL invariance).

    Parameters
    ----------
    features : np.ndarray, shape [N, proj_dim]
        L2-normalised projection embeddings (float32).
    emotion_labels : array-like, length N
        Integer emotion class indices.
    subject_labels : array-like, length N
        Integer (remapped) subject ID indices.
    class_names : list of str, optional
        Emotion class names in label order.
        Defaults to ``["Negative", "Positive", "Surprise", "Others"]``.
    perplexity : float
        t-SNE perplexity parameter.
    n_iter : int
        t-SNE optimisation iterations.
    random_state : int
        Seed for reproducibility.
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    from sklearn.manifold import TSNE

    if class_names is None:
        class_names = ["Negative", "Positive", "Surprise", "Others"]

    emotion_labels = np.array(emotion_labels)
    subject_labels = np.array(subject_labels)

    # ── Compute t-SNE ────────────────────────────────────────────────
    print(f"  Computing t-SNE on {features.shape[0]} × {features.shape[1]} "
          f"feature matrix …", flush=True)
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, features.shape[0] - 1),
        max_iter=n_iter,
        random_state=random_state,
        metric="cosine",       # cosine is appropriate for L2-normalised vectors
        init="pca",
        learning_rate="auto",
    )
    emb = tsne.fit_transform(features.astype(np.float64))
    print(f"  t-SNE done (KL divergence: {tsne.kl_divergence_:.4f})", flush=True)

    fig, (ax_emo, ax_sub) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title_prefix, fontsize=13, fontweight="bold", y=1.02)

    # ── Plot A: Coloured by Emotion ──────────────────────────────────
    n_emotions = len(np.unique(emotion_labels))
    emo_colors = _EMOTION_PALETTE[:n_emotions]
    for cls_idx, (cls_name, color) in enumerate(
        zip(class_names[:n_emotions], emo_colors)
    ):
        mask = emotion_labels == cls_idx
        ax_emo.scatter(
            emb[mask, 0], emb[mask, 1],
            c=color, label=cls_name,
            s=22, alpha=0.75, edgecolors="none",
        )
    ax_emo.set_title("(A) Coloured by Emotion Class\n"
                     "← Tight clusters prove SupCon anchoring →",
                     fontweight="bold", fontsize=10)
    ax_emo.set_xlabel("t-SNE Dim 1"); ax_emo.set_ylabel("t-SNE Dim 2")
    ax_emo.legend(markerscale=1.6, framealpha=0.85,
                  loc="upper right", fontsize=8)
    ax_emo.grid(True, alpha=0.25)
    ax_emo.set_aspect("equal", adjustable="datalim")

    # ── Plot B: Coloured by Subject ID ───────────────────────────────
    unique_subs = np.unique(subject_labels)
    sub_cmap = plt.get_cmap("tab20", len(unique_subs))
    for i, sid in enumerate(unique_subs):
        mask = subject_labels == sid
        ax_sub.scatter(
            emb[mask, 0], emb[mask, 1],
            c=[sub_cmap(i)], label=f"Sub {sid}",
            s=22, alpha=0.70, edgecolors="none",
        )
    ax_sub.set_title("(B) Coloured by Subject ID\n"
                     "← Mixing proves GRL identity-invariance →",
                     fontweight="bold", fontsize=10)
    ax_sub.set_xlabel("t-SNE Dim 1"); ax_sub.set_ylabel("t-SNE Dim 2")
    # Legend only if ≤ 15 subjects (otherwise too cluttered)
    if len(unique_subs) <= 15:
        ax_sub.legend(markerscale=1.4, framealpha=0.8,
                      loc="upper right", fontsize=7,
                      ncol=2)
    ax_sub.grid(True, alpha=0.25)
    ax_sub.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════
# 5. Stage 3 vs Stage 4 Comparison
# ═════════════════════════════════════════════════════════════════════

def plot_stage3_vs_stage4(
    stage3_acc: float,
    stage3_f1:  float,
    stage4_acc: float,
    stage4_f1:  float,
    figsize: tuple = (7, 5),
) -> plt.Figure:
    """
    Grouped bar chart comparing Stage 3 (LOSO baseline) against
    Stage 4 (Macro→Micro Transfer Learning) on Accuracy and Macro F1.

    Parameters
    ----------
    stage3_acc : float
        Stage 3 grand-mean LOSO accuracy (e.g. 0.5811 for 58.11%).
    stage3_f1 : float
        Stage 3 grand-mean LOSO macro F1 (e.g. 0.3331).
    stage4_acc : float
        Stage 4 best validation accuracy.
    stage4_f1 : float
        Stage 4 macro F1 on validation set.
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    metrics     = ["Accuracy (%)", "Macro F1-Score (%)"]
    stage3_vals = [stage3_acc * 100, stage3_f1 * 100]
    stage4_vals = [stage4_acc * 100, stage4_f1 * 100]

    x      = np.arange(len(metrics))
    width  = 0.32

    fig, ax = plt.subplots(figsize=figsize)

    bars3 = ax.bar(x - width / 2, stage3_vals, width,
                   color=_STAGE_PALETTE[0], label="Stage 3 — LOSO Baseline",
                   edgecolor="white", linewidth=0.8, zorder=3)
    bars4 = ax.bar(x + width / 2, stage4_vals, width,
                   color=_STAGE_PALETTE[1], label="Stage 4 — Transfer Learning",
                   edgecolor="white", linewidth=0.8, zorder=3)

    # Value labels above bars
    for bar in list(bars3) + list(bars4):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.8,
            f"{h:.2f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    # Delta annotations between the two bar pairs
    for i, (s3, s4) in enumerate(zip(stage3_vals, stage4_vals)):
        delta  = s4 - s3
        sign   = "▲" if delta >= 0 else "▼"
        color  = "#27AE60" if delta >= 0 else "#C0392B"
        ax.annotate(
            f"{sign} {abs(delta):.2f}%",
            xy=(x[i], max(s3, s4) + 3.5),
            ha="center", va="bottom",
            fontsize=9, color=color, fontweight="bold",
        )

    ax.set_title(
        "Stage 3 Baseline vs Stage 4 Transfer Learning\n"
        "(Macro-to-Micro Adversarial Fine-Tuning)",
        fontweight="bold", pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel("Score (%)", labelpad=8)
    ax.set_ylim(0, max(max(stage3_vals), max(stage4_vals)) + 14)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.88)

    # Footnote
    ax.text(
        0.99, 0.01,
        "Stage 3: 26-Fold LOSO CV  |  Stage 4: Subject-Disjoint Val Split",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7.5, color="#888888", style="italic",
    )

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────
# Quick self-test (run as __main__ to preview all plots)
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random

    print("=" * 60)
    print("  eval_visualizations.py — Self-Test Preview")
    print("=" * 60)

    # 1. Class distribution
    sample_dist = {"Negative": 320, "Positive": 74, "Surprise": 73, "Others": 59}
    fig1 = plot_class_distribution(sample_dist)
    fig1.suptitle("SELF-TEST: Class Distribution", fontsize=11)

    # 2. Training history (synthetic)
    N = 40
    hist = {
        "train_supcon_loss":   [1.5 * np.exp(-i/20) + 0.05*random.random() for i in range(N)],
        "val_supcon_loss":     [1.6 * np.exp(-i/22) + 0.08*random.random() for i in range(N)],
        "train_emotion_loss":  [2.0 * np.exp(-i/18) + 0.06*random.random() for i in range(N)],
        "train_identity_loss": [1.0 * np.exp(-i/25) + 0.04*random.random() for i in range(N)],
        "train_emotion_acc":   [min(0.95, 0.3 + i/N * 0.65 + 0.02*random.random()) for i in range(N)],
        "val_emotion_acc":     [min(0.90, 0.25 + i/N * 0.60 + 0.03*random.random()) for i in range(N)],
        "grl_lambda":          [1/(1+np.exp(-10*(i/N - 0.5))) for i in range(N)],
    }
    fig2 = plot_training_history(hist, phase_label="SELF-TEST: Training History")

    # 3. Confusion matrix (synthetic)
    rng = np.random.default_rng(0)
    y_t = rng.integers(0, 4, 200)
    y_p = np.where(rng.random(200) > 0.45, y_t, rng.integers(0, 4, 200))
    fig3 = plot_confusion_matrix(y_t, y_p, ["Negative","Positive","Surprise","Others"])

    # 4. t-SNE (synthetic 64-d features)
    feats = rng.standard_normal((150, 64)).astype(np.float32)
    # Make them slightly cluster-able by emotion
    emo_l = rng.integers(0, 4, 150)
    for c in range(4):
        feats[emo_l == c] += rng.standard_normal(64) * 0.5
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    sub_l = rng.integers(0, 10, 150)
    fig4 = plot_tsne_embeddings(feats, emo_l, sub_l, n_iter=300)

    # 5. Stage comparison
    fig5 = plot_stage3_vs_stage4(0.5811, 0.3331, 0.6450, 0.4100)

    plt.show()
    print("\n  ✓ All five plots rendered successfully.")
