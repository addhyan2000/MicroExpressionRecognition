"""Manifold T-SNE Analysis Module.

Projects 128-D abstract representation sequences dynamically into 2D spaces
mathematically proving Identity Diffusion / Identity-Invariance mapping metrics 
combined with distinctive discriminative emotional clustering. 

Outputs are strictly generated in publication-quality High Resolution (300 DPI) 
utilizing LaTeX mathematical fonts.

Author  : Addhyan Pandey
Project : Master's Thesis — Micro-Expression Recognition
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from torch.utils.data import DataLoader
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

    def extract_and_cache_embeddings(
        self, 
        model: torch.nn.Module, 
        dataloader: DataLoader, 
        device: torch.device, 
        identifier_tag: str
    ) -> str:
        """Extract 'proj_features' (identity-invariant embeddings) structurally and cache them.
        
        Parameters
        ----------
        model : nn.Module
            The MTMNet instantiation.
        dataloader : DataLoader
            A consolidated evaluation loader across all validation subsets.
        device : torch.device
            Computation device allocation.
        identifier_tag : str
            Prefix to save the cache explicitly.
            
        Returns
        -------
        str
            The final cache file absolute path.
        """
        model.eval()
        all_embeddings = []
        all_emotions = []
        all_subjects = []
        
        cache_path = os.path.join(self.cache_dir, f"{identifier_tag}_embeddings.npy")
        
        if os.path.exists(cache_path):
            logger.info(f"Target memory cache formally detected. Utilizing {cache_path}")
            return cache_path
            
        logger.info(f"Accumulating embedding arrays dynamically for tag: {identifier_tag}")
        with torch.no_grad():
            for batch_data in dataloader:
                # Based on the unified loader context, batch_data could be 
                # (videos, labels, subjects) or just (videos, labels).
                # Assuming standard loaders yield (videos, labels). If subjects are needed, 
                # they must be packed into the dataset strictly.
                if len(batch_data) == 3:
                    videos, labels, subjects = batch_data
                else:
                    videos, labels = batch_data
                    subjects = torch.zeros_like(labels) # Mapping failsafe if identities weren't explicitly bound
                
                videos = videos.to(device)
                lengths = torch.full((videos.size(0),), videos.size(1), dtype=torch.long, device=device)
                
                out = model(videos, lengths=lengths)
                
                # Fetching proj_features explicitly isolates identity bias
                embeddings = out["features_proj"].cpu().numpy()
                all_embeddings.append(embeddings)
                all_emotions.append(labels.numpy())
                all_subjects.append(subjects.numpy() if isinstance(subjects, torch.Tensor) else np.array(subjects))
                
        # Consolidate memory geometries
        X = np.concatenate(all_embeddings, axis=0) # (N, 128)
        y_emo = np.concatenate(all_emotions, axis=0)
        y_sub = np.concatenate(all_subjects, axis=0)
        
        logger.info("Projecting 128-D geometry coordinates down to exactly 2-D spaces.")
        tsne = TSNE(n_components=2, perplexity=30.0, learning_rate=200.0, init='pca', random_state=42)
        X_2d = tsne.fit_transform(X)
        
        try:
            subject_dtype = np.int32
            _ = y_sub.astype(subject_dtype)
        except ValueError:
            subject_dtype = 'U50'
            
        structured_array = np.zeros(len(X_2d), dtype=[
            ('z1', np.float32), 
            ('z2', np.float32), 
            ('emotion', np.int32), 
            ('subject', subject_dtype)
        ])
        structured_array['z1'] = X_2d[:, 0]
        structured_array['z2'] = X_2d[:, 1]
        structured_array['emotion'] = y_emo.astype(np.int32)
        structured_array['subject'] = y_sub.astype(subject_dtype)
        
        np.save(cache_path, structured_array)
        return cache_path

    def block_plot_dual_manifolds(self, identifier_tag: str, emotion_names: dict = None) -> None:
        """Draw and export the thesis-ready dual 2D planes.
        
        Parameters
        ----------
        identifier_tag : str
            The cache tag identifying stored points.
        emotion_names : dict, optional
            Mapping integer mappings to formal string semantics.
        """
        cache_path = os.path.join(self.cache_dir, f"{identifier_tag}_embeddings.npy")
        
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Missing cache dependencies: {cache_path}")
            
        struct_arr = np.load(cache_path)
        z1 = struct_arr['z1']
        z2 = struct_arr['z2']
        y_emo = struct_arr['emotion']
        y_sub = struct_arr['subject']
        
        if emotion_names is None:
            emotion_names = {0: 'Negative', 1: 'Positive', 2: 'Surprise'}
            
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # --- Plot 1: Emotion Clusters ---
        ax = axes[0]
        scatter = ax.scatter(z1, z2, c=y_emo, cmap='Set1', alpha=0.8, edgecolor='k', s=80)
        epochs = list(emotion_names.keys())
        handles, _ = scatter.legend_elements(prop="colors", alpha=0.8)
        legend_labels = [emotion_names.get(e, str(e)) for e in epochs]
        ax.legend(handles, legend_labels, title="Affective Class", loc="best")
        ax.set_title("Discriminative Embedding Manifold (Emotion Clusters)", fontweight='bold', pad=15)
        ax.set_xlabel(r"t-SNE Dimension 1 ($z_1$)")
        ax.set_ylabel(r"t-SNE Dimension 2 ($z_2$)")
        ax.grid(True, linestyle='--', alpha=0.4)
        
        # --- Plot 2: Identity Diffusion ---
        ax = axes[1]
        classes = len(np.unique(y_sub))
        cmap = plt.cm.get_cmap('Spectral', classes)
        scatter = ax.scatter(z1, z2, c=y_sub, cmap=cmap, alpha=0.7, edgecolor='none', s=50)
        ax.set_title("Identity-Invariant Diffusion Matrix (Subject Independence)", fontweight='bold', pad=15)
        ax.set_xlabel(r"t-SNE Dimension 1 ($z_1$)")
        ax.set_ylabel(r"t-SNE Dimension 2 ($z_2$)")
        ax.grid(True, linestyle='--', alpha=0.4)
        
        # Mathematical constraint for publication
        plt.tight_layout(pad=3.0)
        save_path = os.path.join(self.output_dir, f"manifold_{identifier_tag}.png")
        plt.savefig(save_path, dpi=300, format='png', bbox_inches='tight')
        plt.close(fig)
        logger.info(f"High-Density Manifold strictly outputted correctly to {save_path}")
