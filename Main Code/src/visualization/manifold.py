"""Manifold T-SNE Analysis Module.

Projects 128-D abstract representation sequences dynamically into 2D spaces
mathematically proving Identity Diffusion / Identity-Invariance mapping metrics 
combined with distinctive discriminative emotional clustering. 

Outputs are strictly generated in publication-quality High Resolution (300 DPI) 
utilizing LaTeX mathematical fonts.

Author  : Addhyan Pant
Project : Master's Thesis — Micro-Expression Recognition
"""

import os
import re
import numpy as np
import matplotlib as mpl
mpl.use('Agg')  # Headless Stability
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import torch
from torch.utils.data import DataLoader
from typing import Dict
import logging

logger = logging.getLogger(__name__)

__all__ = ["ManifoldProjector"]

class ManifoldProjector:
    """t-SNE Embeddings Projector.
    
    Collects representations from the exact evaluation phase dynamically, caches 
    abstract projections avoiding repetitive inference delays, and writes highly 
    structured Cartesian geometry plots isolating identity vectors and emotion vectors.
    """
    
    def __init__(self, cache_dir: str = "./cache/tsne", output_dir: str = "./plots"):
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Master publication font configurations
        plt.rcParams.update({
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.labelsize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12
        })

    def _sanitize_tag(self, tag: str) -> str:
        """Sanitizer for filesystem-safe identifier tags."""
        return re.sub(r'[\\/*?:"<>| ]', '_', tag)

    def collect_fold_embeddings(
        self, 
        model: torch.nn.Module, 
        dataloader: DataLoader, 
        device: torch.device
    ) -> Dict[str, np.ndarray]:
        """Collect embeddings for a single fold.
        
        Global Aggregation API: Allows the master script to aggregate all test-set 
        embeddings from every LOSO fold prior to running t-SNE.
        """
        model.eval()
        all_embeddings = []
        all_emotions = []
        all_subjects = []
        
        with torch.no_grad():
            for batch_data in dataloader:
                if len(batch_data) == 4:
                    videos = batch_data[0]
                    labels = batch_data[1]
                    subjects = batch_data[2]
                    lengths = batch_data[3]
                else:
                    raise ValueError("Batch data must contain exactly 4 items: videos, labels, subjects, lengths.")
                
                videos = videos.to(device)
                lengths = lengths.to(device)
                
                out = model(videos, lengths=lengths)
                
                # Fetching proj_features explicitly isolates identity bias
                embeddings = out["features_proj"].cpu().numpy()
                all_embeddings.append(embeddings)
                all_emotions.append(labels.cpu().numpy())
                all_subjects.append(subjects.cpu().numpy() if isinstance(subjects, torch.Tensor) else np.array(subjects))
                
        # Cleanliness: Ensure system RAM is freed and cache cleared
        try:
            return {
                'embeddings': np.concatenate(all_embeddings, axis=0),
                'labels': np.concatenate(all_emotions, axis=0),
                'subjects': np.concatenate(all_subjects, axis=0)
            }
        finally:
            del all_embeddings, all_emotions, all_subjects
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def process_and_cache_global_embeddings(
        self, 
        global_data: Dict[str, np.ndarray], 
        identifier_tag: str,
        perplexity: float = 30.0
    ) -> str:
        """Apply dimensionality reduction, compute quantitative metrics, and save.
        
        Parameters
        ----------
        global_data : Dict[str, np.ndarray]
            Aggregated global data containing 'embeddings', 'labels', and 'subjects'.
        identifier_tag : str
            Prefix to save the cache explicitly.
        perplexity : float
            The perplexity parameter for t-SNE optimization.
            
        Returns
        -------
        str
            The final cache file absolute path (.npz).
        """
        safe_tag = self._sanitize_tag(identifier_tag)
        cache_path = os.path.join(self.cache_dir, f"{safe_tag}_embeddings.npz")
        
        if os.path.exists(cache_path):
            logger.info(f"Target memory cache formally detected. Utilizing {cache_path}")
            return cache_path
            
        X = global_data['embeddings']
        y_emo = global_data['labels']
        y_sub = global_data['subjects']
        
        logger.info("Applying Standard Scaler to feature embeddings for dimension hardening.")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Quantitative Validation: Latent (128-D) Features
        unique_emotions = np.unique(y_emo)
        if len(unique_emotions) > 1:
            emo_silhouette_latent = float(silhouette_score(X_scaled, y_emo))
        else:
            emo_silhouette_latent = 0.0
            
        unique_subjects = np.unique(y_sub)
        if len(unique_subjects) > 1:
            if not np.issubdtype(y_sub.dtype, np.number):
                _, sub_labels = np.unique(y_sub, return_inverse=True)
            else:
                sub_labels = y_sub
            id_silhouette_latent = float(silhouette_score(X_scaled, sub_labels))
        else:
            id_silhouette_latent = 0.0
            
        logger.info("Projecting 128-D geometry coordinates down to exactly 2-D spaces.")
        # TSNE Optimization
        tsne = TSNE(
            n_components=2, 
            perplexity=perplexity, 
            early_exaggeration=12.0,
            learning_rate="auto", 
            n_iter=2000, 
            init='pca', 
            random_state=42
        )
        X_2d = tsne.fit_transform(X_scaled)
        
        # Quantitative Validation: 2-D Features
        if len(unique_emotions) > 1:
            emo_silhouette_2d = float(silhouette_score(X_2d, y_emo))
        else:
            emo_silhouette_2d = 0.0
            
        if len(unique_subjects) > 1:
            id_silhouette_2d = float(silhouette_score(X_2d, sub_labels))
        else:
            id_silhouette_2d = 0.0
            
        logger.info(f"Emotion Silhouette (Latent vs 2D): {emo_silhouette_latent:.4f} -> {emo_silhouette_2d:.4f}")
        logger.info(f"Identity Silhouette (Latent vs 2D): {id_silhouette_latent:.4f} -> {id_silhouette_2d:.4f}")
            
        silhouette_metrics = {
            'emotion_latent': emo_silhouette_latent,
            'identity_latent': id_silhouette_latent,
            'emotion_2d': emo_silhouette_2d,
            'identity_2d': id_silhouette_2d
        }
        
        # Robust Data Formatting
        np.savez_compressed(
            cache_path,
            embeddings=X_2d,
            labels=y_emo,
            subject_ids=y_sub,
            **silhouette_metrics
        )
        
        return cache_path

    def block_plot_dual_manifolds(
        self, 
        identifier_tag: str, 
        emotion_map: Dict[int, str]
    ) -> None:
        """Draw and export the thesis-ready dual 2D planes.
        
        Parameters
        ----------
        identifier_tag : str
            The cache tag identifying stored points.
        emotion_map : Dict[int, str]
            Dynamic legend mapping integer labels to formal string semantics.
        """
        safe_tag = self._sanitize_tag(identifier_tag)
        cache_path = os.path.join(self.cache_dir, f"{safe_tag}_embeddings.npz")
        
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Missing cache dependencies: {cache_path}")
            
        data = np.load(cache_path, allow_pickle=True)
        X_2d = data['embeddings']
        y_emo = data['labels']
        y_sub = data['subject_ids']
        
        emo_lat = float(data['emotion_latent'])
        id_lat = float(data['identity_latent'])
        emo_2d = float(data['emotion_2d'])
        id_2d = float(data['identity_2d'])
        
        z1 = X_2d[:, 0]
        z2 = X_2d[:, 1]
        
        # Derive c_val before shuffling to satisfy Z-Order bias prevention identically
        if np.issubdtype(y_sub.dtype, np.number):
            c_val = y_sub
        else:
            _, c_val = np.unique(y_sub, return_inverse=True)
        
        # ELIMINATE Z-ORDER BIAS: Explicitly shuffle c_val alongside z1 and z2
        num_points = len(z1)
        shuffle_idx = np.random.permutation(num_points)
        z1 = z1[shuffle_idx]
        z2 = z2[shuffle_idx]
        y_emo = y_emo[shuffle_idx]
        c_val = c_val[shuffle_idx]
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # --- Plot 1: Emotion Clusters ---
        ax = axes[0]
        # Divergent color palette for Emotion
        scatter = ax.scatter(z1, z2, c=y_emo, cmap='Set1', alpha=0.8, edgecolor='k', s=80)
        
        handles, _ = scatter.legend_elements(prop="colors", alpha=0.8)
        unique_emotions = np.unique(y_emo)
        legend_labels = [emotion_map.get(int(e), str(e)) for e in unique_emotions]
        
        ax.legend(handles, legend_labels, title="Affective Class", loc="best")
        ax.set_title(
            f"Discriminative Embedding Manifold (Emotion)\n"
            f"Latent Space (128-D) Quality: $S_{{Latent}}$ = {emo_lat:.4f}  |  Manifold Space (2-D) Quality: $S_{{2D}}$ = {emo_2d:.4f}", 
            fontweight='bold', pad=15
        )
        ax.set_xlabel(r"t-SNE Dimension 1 ($z_1$)")
        ax.set_ylabel(r"t-SNE Dimension 2 ($z_2$)")
        ax.grid(True, linestyle='--', alpha=0.4)
        
        # --- Plot 2: Identity Diffusion ---
        ax = axes[1]
            
        classes = len(np.unique(c_val))
        
        # High-contrast discrete colormap (`tab20` or `gist_rainbow`)
        cmap = mpl.colormaps['tab20'].resampled(classes) if classes <= 20 else mpl.colormaps['gist_rainbow'].resampled(classes)
        
        scatter2 = ax.scatter(z1, z2, c=c_val, cmap=cmap, alpha=0.7, edgecolor='none', s=50)
        
        # Identity Visualization: Add Colorbar instead of legend
        cbar = fig.colorbar(scatter2, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Subject IDs', rotation=270, labelpad=15)
        
        ax.set_title(
            f"Identity-Invariant Diffusion Matrix (Subject Independence)\n"
            f"Latent Space (128-D) Quality: $S_{{Latent}}$ = {id_lat:.4f}  |  Manifold Space (2-D) Quality: $S_{{2D}}$ = {id_2d:.4f}", 
            fontweight='bold', pad=15
        )
        ax.set_xlabel(r"t-SNE Dimension 1 ($z_1$)")
        ax.set_ylabel(r"t-SNE Dimension 2 ($z_2$)")
        ax.grid(True, linestyle='--', alpha=0.4)
        
        # Mathematical constraint for publication
        plt.tight_layout(pad=3.0)
        save_path = os.path.join(self.output_dir, f"manifold_{safe_tag}.png")
        plt.savefig(save_path, dpi=300, format='png', bbox_inches='tight')
        plt.close(fig)
        logger.info(f"High-Density Manifold strictly outputted correctly to {save_path}")
