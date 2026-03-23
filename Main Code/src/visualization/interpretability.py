"""3D Grad-CAM++ Interpretability Module.

This module implements gradient-based attribution for 5D spacetime tensors 
(Batch, Channels, Time, Height, Width) or (Batch, Time, Channels, Height, Width).
It explicitly targets the multiple parallel streams of the STSTNet/SpatialEncoder 
to produce discriminative spatiotemporal heatmaps, validating that the network 
attends to pertinent Action Units (AUs) during the micro-expression apex.

WARNING:
    The GuidedBackprop module natively hooks into ``nn.Module`` activations 
    (e.g., ``nn.ReLU()``, ``nn.GELU()``, ``nn.Sigmoid()``). It **will not** trigger 
    on PyTorch functional calls (e.g., ``F.relu``) inside your model's forward method.
    You must convert functional activations to modules for interpretability 
    to function correctly.

Reference:
    Chattopadhyay et al., "Grad-CAM++: Generalized Gradient-based Visual 
    Explanations for Deep Convolutional Networks", WACV 2018.
"""

import warnings
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast
from typing import List, Tuple, Dict, Any, Union, Optional

from src.models.components.spatial_encoder import SimAM

__all__ = ["GuidedBackprop", "GradCAM3DPlusPlus"]


class GuidedBackprop(object):
    """Guided Backprop module for pixel-level attribution."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks: List[Any] = []
        self.forward_activations: Dict[int, torch.Tensor] = {}
        
    def __enter__(self):
        def forward_hook(module, input, output):
            # Guided Mask Integrity (CRITICAL): Clone and detach to protect the mask from inplace activations
            self.forward_activations[id(module)] = output.clone().detach()
            
        def backward_hook(module, grad_in, grad_out):
            activation = self.forward_activations.get(id(module))
            if activation is not None and grad_in[0] is not None:
                # True Dual-Masked Guided Backprop (Zeiler & Fergus)
                # Mask negative gradients and zero-activations
                grad = grad_in[0]
                grad = torch.clamp(grad, min=0.0)
                # Where activation > 0 
                grad = grad * (activation > 0).to(grad.dtype)
                return (grad,)
        
        # SimAM-Aware Attribution: Target CNN, Transformer, and SimAM (Sigmoid) activation functions
        for module in self.model.modules():
            if isinstance(module, (nn.ReLU, nn.GELU, nn.Sigmoid)):
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_full_backward_hook(backward_hook))
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        
    def cleanup(self) -> None:
        """Removes Guided Backprop hooks defensively.
        (forward_activations is retained for external verification 
        and will be garbage collected with the object).
        """
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


class GradCAM3DPlusPlus(object):
    """3D Grad-CAM++ for Spatiotemporal Video Models.

    Attaches hooks to the parallel feature extraction streams of the 
    `SpatialEncoder`. Computes the weighted combination of positive 
    partial derivatives (Grad-CAM++) over the temporal and spatial 
    dimensions to produce highly localized heatmaps.

    Implemented as a standard Python Context Manager to guarantee 
    memory safety, decoupling diagnostic footprint from PyTorch state.

    Parameters
    ----------
    model : nn.Module
        The target MTMNet or wrapper model.
    target_layer_name : str
        Usually 'spatial_encoder.streams'. We dynamically hook into 
        the final SimAM layer in each stream.
    spatial_res : int
        The standard input spatial resolution, defaulting to 28 for STSTNet.
        This enables GPU-CPU Bottleneck Optimization by precomputing the 
        spatial-only Gaussian blurring kernel.
    """

    def __init__(self, model: nn.Module, target_layer_name: str = "spatial_encoder.streams", spatial_res: int = 28):
        self.model = model
        self.target_layer_name = target_layer_name
        self.activations: Dict[int, torch.Tensor] = {}
        self.gradients: Dict[int, torch.Tensor] = {}
        self.hooks: List[Any] = []
        
        # Pre-compute the 2D Spatial-Only Gaussian kernel once on CPU
        self.gaussian_kernel = self._get_gaussian_kernel3d(kernel_size=3, spatial_res=spatial_res)

    @staticmethod
    def _get_gaussian_kernel3d(kernel_size: int = 3, spatial_res: int = 28) -> torch.Tensor:
        """
        Generate a strictly 2Spatial + 1Temporal (size=1) Gaussian kernel to 
        prevent "temporal bleeding" of saliency across the micro-expression frames.
        """
        sigma = 1.0 * (float(spatial_res) / 28.0)
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        
        # Generate 2D spatial smoothing
        kernel2d = g.view(kernel_size, 1) * g.view(1, kernel_size)
        kernel2d = kernel2d / kernel2d.sum()
        
        # Broadcast into 3D structure with T=1 (Batch, Channel, Time, Height, Width)
        # 1 Channel in, 1 Channel out, Temporal=1, Height, Width
        return kernel2d.view(1, 1, 1, kernel_size, kernel_size)
        
    @staticmethod
    def apply_jet_colormap(heatmap: np.ndarray) -> np.ndarray:
        """
        Converts a normalized (T, H, W) grayscale heatmap into a 
        (T, H, W, 3) RGB volume using the JET colormap for publication figures.
        """
        # Ensure input is safely constrained to [0, 1] before mapping
        heatmap_safe = np.clip(heatmap, 0.0, 1.0)
        colormap = plt.get_cmap('jet')
        # Returns (T, H, W, 4) RGBA array in floats [0.0, 1.0]
        heatmap_rgba = colormap(heatmap_safe)
        heatmap_rgb = (heatmap_rgba[..., :3] * 255).astype(np.uint8)
        return heatmap_rgb
        
    @staticmethod
    def blend_heatmap_with_video(video: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Blends a grayscale (T, H, W) normalized heatmap with a generic (T, C, H, W) video 
        to produce a (T, H, W, 3) RGB volume containing Action Units overlaid on the face.
        """
        heatmap_rgb = GradCAM3DPlusPlus.apply_jet_colormap(heatmap) # (T, H, W, 3)
        T, C, H, W = video.shape
        if C == 1:
            video_rgb = np.repeat(video.transpose(0, 2, 3, 1), 3, axis=-1)
        elif C == 3:
            video_rgb = video.transpose(0, 2, 3, 1)
        else:
            raise ValueError(f"Video must have 1 or 3 channels. Got {C}")
            
        if video_rgb.max() <= 1.05: # safety tolerance for float normalization
            video_rgb = (np.clip(video_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            video_rgb = video_rgb.astype(np.uint8)
            
        blended = np.zeros_like(video_rgb)
        for t in range(T):
            blended[t] = cv2.addWeighted(video_rgb[t], 1.0 - alpha, heatmap_rgb[t], alpha, 0.0)
        return blended

    def _replace_functional_activations(self, model: nn.Module) -> None:
        """
        The "Functional Trap" Fallback utility: Validates that the model employs 
        legitimate `nn.Module` activations for GuidedBackprop tracking. 
        Provides a strict RuntimeError fallback instructing the user on wrapper mechanics.
        """
        has_supported_act = any(isinstance(m, (nn.ReLU, nn.GELU, nn.Sigmoid)) for m in model.modules())
        if not has_supported_act:
             raise RuntimeError(
                "The 'Functional Trap' Detected: No nn.ReLU, nn.GELU, or nn.Sigmoid modules found. "
                "Guided Backprop cannot natively intercept functional F.relu calls in PyTorch's execution graph. "
                "Please construct a specific model wrapper that overrides the forward pass, explicitly replacing "
                "functional activations with their corresponding nn.Module instances to enable feature attribution."
             )

    def __enter__(self):
        self.model.eval()
        device = next(self.model.parameters()).device
        self.gaussian_kernel = self.gaussian_kernel.to(device)
        self._replace_functional_activations(self.model)
        self._register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self) -> None:
        """Manual cleanup method to purge hooks and references."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations.clear()
        self.gradients.clear()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks into the parallel 3D streams."""
        try:
            streams_module = self.model.spatial_encoder.streams
        except AttributeError:
            raise ValueError(f"Target model must contain `{self.target_layer_name}`.")
            
        num_streams = len(streams_module)
        if num_streams == 0:
            raise ValueError("No streams found in the target spatial encoder.")

        for i, stream in enumerate(streams_module):
            # Target the SimAM layer in the stream via explicit type-safe hooking
            target_layer = None
            for layer in reversed(stream):
                if isinstance(layer, SimAM):
                    target_layer = layer
                    break
            
            if target_layer is None:
                raise TypeError(f"Targeted layer must be SimAM (requires 'e_lambda' attribute), but none found in stream {i}.")
                
            def forward_hook(module, input, output, stream_idx=i):
                self.activations[stream_idx] = output.detach()
                
            def backward_hook(module, grad_input, grad_output, stream_idx=i):
                self.gradients[stream_idx] = grad_output[0].detach()

            self.hooks.append(target_layer.register_forward_hook(forward_hook))
            self.hooks.append(target_layer.register_full_backward_hook(backward_hook))

    @autocast(enabled=False)
    def generate_heatmap(
        self, 
        input_video: torch.Tensor, 
        lengths: torch.Tensor, 
        target_class: int,
        precomputed_outputs: Optional[Dict[str, torch.Tensor]] = None,
        input_format: str = "BCTHW"
    ) -> Dict[str, Union[Dict[str, np.ndarray], np.ndarray, int]]:
        """Generate spatiotemporal Guided Grad-CAM++ heatmaps.

        Parameters
        ----------
        input_video : torch.Tensor
            Optical flow video tensor. Should be batch_size = 1.
        lengths : torch.Tensor
            Sequence length `(1,)`.
        target_class : int
            The class index for which we compute the gradients.
        precomputed_outputs : Optional[Dict[str, torch.Tensor]]
            Model outputs from a prior forward pass to bypass redundant computation.
            If used, the prior forward pass MUST have been executed inside this context manager,
            and `input_video.requires_grad_()` must have been explicitly set to True beforehand.
        input_format : str
            String format indicating the dimension order: "BCTHW" or "BTCHW".

        Returns
        -------
        Dict[str, Union[Dict[str, np.ndarray], np.ndarray, int]]
            Dictionary containing:
            - "Modal Saliency": Dict of individual normalized heatmaps for each stream globally scaled.
            - "Guided Fused": The pixel-level attribution proof required for definitive Action Unit (AU) validation.
            - "Guided Gray": Summed clean single-channel "Glow Map" representation of face.
            - "Temporal Importance": 1D array of length T representing saliency sum per frame matching ground-truth apices.
            - "Apex Frame Index": Integer index representing exactly where the model's attention peaked.
        """
        assert input_video.ndim == 5, "Input video must be a 5D tensor."
        assert input_video.size(0) == 1, "Grad-CAM is exclusively engineered for batch_size=1 visualizations."
        assert input_format in ["BCTHW", "BTCHW"], "input_format must be 'BCTHW' or 'BTCHW'."
        assert not self.model.training, "Model must be in eval() mode."
        
        device = next(self.model.parameters()).device
        
        # Leaf-Safety Hardening: Safely convert input representation into a gradient-tracking leaf
        input_video = input_video.to(device)
        input_video = input_video.detach().requires_grad_(True)
        
        lengths = lengths.to(device)
        
        self.model.zero_grad(set_to_none=True)
        
        try:
            with GuidedBackprop(self.model) as gbp:
                with torch.enable_grad():
                    if precomputed_outputs is not None:
                        logits = precomputed_outputs["class_logits"]
                    else:
                        self.activations.clear()
                        self.gradients.clear()
                        outputs = self.model(input_video, lengths=lengths)
                        logits = outputs["class_logits"]
                        
                # Functional Trap execution checked outside graph tracking 
                if precomputed_outputs is None and not gbp.forward_activations:
                    raise RuntimeError(
                        "The 'Functional Activation' Trap: No module-based activations (e.g., nn.ReLU) "
                        "were triggered during inference. Guided Backprop technically requires explicit "
                        "nn.Module layer activations, not functional calls like F.relu to verify AUs."
                    )
                        
                with torch.enable_grad():
                    # Confidence Guard Validation
                    probs = F.softmax(logits[0], dim=0)
                    target_prob = probs[target_class].item()
                    if target_prob < 0.4:
                        warnings.warn(
                            f"Target class probability is exceptionally low ({target_prob:.1%}). "
                            "Saliency maps for low-confidence predictions are scientifically unreliable "
                            "and may represent random artifacting rather than targeted feature attribution."
                        )
                    
                    target_score = logits[0, target_class]
                    target_score.backward(retain_graph=True)
                
            assert len(self.activations) > 0 and len(self.gradients) > 0, \
                "Activations and gradients are empty! Forward/Backward hooks failed to capture data."

            if input_video.grad is None:
                raise RuntimeError(
                    "Input video gradients are None. Guided Backprop requires gradients flowing back "
                    "to the input. If using `precomputed_outputs`, ensure `input_video.requires_grad_(True)` "
                    "was called BEFORE the forward pass."
                )

            # API Robustness: Explicit indexing
            if input_format == "BCTHW":
                c_dim, t_dim = 1, 2
            elif input_format == "BTCHW":
                t_dim, c_dim = 1, 2
            else:
                raise ValueError("input_format must be 'BCTHW' or 'BTCHW'")
            t_in, h_in, w_in = input_video.size(t_dim), input_video.size(3), input_video.size(4)
            
            raw_stream_heatmaps = {}
            heatmaps_list = []
            
            for stream_idx in sorted(self.activations.keys()):
                A = self.activations[stream_idx].to(torch.float32)
                grad_A = self.gradients[stream_idx].to(torch.float32)
                
                # Compute Importance Coefficient (beta) for Energy-Weighted Fusion
                beta = A.max().item()
                
                # 1. Take positive gradients (ReLU)
                positive_grads = F.relu(grad_A)
                
                # 2. Final Mathematical Hardening: alpha calculation remains in float64 until final weight summation
                A_f64 = A.to(torch.float64)
                grad_A_f64 = grad_A.to(torch.float64)
                positive_grads_f64 = positive_grads.to(torch.float64)
                
                # Log-safe math
                grad_2 = grad_A_f64.pow(2)
                grad_3 = grad_A_f64.pow(3)
                
                global_sum = (A_f64 * grad_3).sum(dim=(2, 3, 4), keepdim=True)
                alpha = grad_2 / (2.0 * grad_2 + global_sum + 1e-9)
                alpha = torch.where(grad_A_f64 != 0.0, alpha, torch.zeros_like(alpha))
                
                # Mathematical Stability in weights: maintain float64 for sum
                weights_f64 = (alpha * positive_grads_f64).sum(dim=(2, 3, 4), keepdim=True)
                
                # Downcast only final spatial weights to float32
                weights = weights_f64.to(torch.float32)
                
                # 4. Spatially weight the activations
                scaled_activations = weights * A
                
                # 5. Sum over channels and apply ReLU to isolate discriminative bounds
                heatmap_tensor = scaled_activations.sum(dim=1).unsqueeze(1) # (1, 1, T', H', W')
                heatmap_tensor = F.relu(heatmap_tensor)
                
                # Coordinate Integrity FIX: align_corners=False prevents sub-pixel coordinate shift
                heatmap_tensor = F.interpolate(
                    heatmap_tensor, size=(t_in, h_in, w_in), 
                    mode='trilinear', align_corners=False
                )
                
                # Apply Importance Coefficient (beta) directly via Tensor scaling
                heatmap_tensor = heatmap_tensor * beta
                
                stream_name = f"stream_{stream_idx}"
                raw_stream_heatmaps[stream_name] = heatmap_tensor.squeeze(0).squeeze(0).detach().cpu().numpy()
                
                # Append to Global fusion list (still on GPU)
                heatmaps_list.append(heatmap_tensor)
                
                # Structured GPU Memory Integrity: Explictly delete heavy locals immediately.
                del A, grad_A, A_f64, grad_A_f64, positive_grads, positive_grads_f64, grad_2, grad_3, global_sum, alpha, weights_f64, weights, scaled_activations, heatmap_tensor
                
            # Importance-Preserving Modal Saliency (Global Scalar Normalization)
            stream_heatmaps = {}
            if raw_stream_heatmaps:
                global_max = max(hm.max() for hm in raw_stream_heatmaps.values())
                global_min = min(hm.min() for hm in raw_stream_heatmaps.values())
                if global_max > global_min:
                    for name, hm in raw_stream_heatmaps.items():
                        stream_heatmaps[name] = (hm - global_min) / (global_max - global_min)
                else:
                    for name, hm in raw_stream_heatmaps.items():
                        stream_heatmaps[name] = np.zeros_like(hm)
                        
            # Mathematical Fusion: Summation of importance-weighted raw heatmaps (1, 1, T, H, W)
            raw_fused_heatmap = sum(heatmaps_list)
            
            # Temporal Integrity: Apply the valid_len masking BEFORE the Gaussian smoothing
            valid_len = int(lengths[0].item())
            if valid_len < t_in:
                # Mask out time steps beyond the sequence length natively
                raw_fused_heatmap[:, :, valid_len:, :, :] = 0.0
            
            # Accurate Temporal Importance derived strictly from UN-smoothed, raw heatmaps BEFORE clipping
            raw_fused_np = raw_fused_heatmap.squeeze(0).squeeze(0).detach().cpu().numpy() # Always (T, H, W)
            temporal_importance = raw_fused_np.sum(axis=(-2, -1))
            
            # Saliency Peak Detection (Thesis Metric) strictly anchored to raw data tensor
            apex_idx = int(np.argmax(temporal_importance))
            
            # Elimination of Temporal Bleeding: 2D Spatial-Only smoothing with T=1 padding to safeguard temporal actuation precision
            fused_heatmap = F.conv3d(raw_fused_heatmap, self.gaussian_kernel, padding=(0, 1, 1))
            
            # Sub-Pixel Spatial Alignment Fix + Coordinate Integrity FIX: align_corners=False
            if input_format == "BCTHW":
                target_size = input_video.grad.shape[2:] # (T, H, W)
            else:
                target_size = (input_video.grad.shape[1], input_video.grad.shape[3], input_video.grad.shape[4])
                
            # Explicitly interpolate the Grad-CAM heatmap tensor to match the exact spatial resolution
            fused_heatmap = F.interpolate(
                fused_heatmap, size=target_size, mode='trilinear', align_corners=False
            )
            
            # Valid-Frame Masking (Repeat masking after convolution to eliminate edge bleeding, satisfying Temporal Integrity perfectly)
            if valid_len < t_in:
                fused_heatmap[:, :, valid_len:, :, :] = 0.0
                
            # Final Global Min-Max Normalization
            f_min, f_max = fused_heatmap.min(), fused_heatmap.max()
            if f_max > f_min:
                fused_heatmap = (fused_heatmap - f_min) / (f_max - f_min)
            else:
                fused_heatmap = torch.zeros_like(fused_heatmap)
                
            # Extract to numpy
            fused_heatmap_np = fused_heatmap.squeeze(0).squeeze(0).detach().cpu().numpy() # (T, H, W)
                
            # Element-wise Hybrid Guided Grad-CAM++ Multiplication 
            guided_grads = input_video.grad.detach().cpu().numpy().squeeze(0) # (T, C, H, W) or (C, T, H, W)
            
            # Absolute Magnitude for Guided Fusion
            guided_grads = np.abs(guided_grads)
            
            # Final Mathematical Hardening: Epsilon-Safe Power Scale to boost the AU visibility
            guided_grads = np.power(guided_grads + 1e-12, 0.5)
            
            # Broadcast the fused heatmap safely across the channel dimension
            if input_format == "BCTHW":
                # guided_grads is (C, T, H, W)
                guided_fused = guided_grads * fused_heatmap_np[np.newaxis, :, :, :]
                
                # Guided Saliency Channel Aggregation (Representation Fix)
                guided_gray = guided_fused.sum(axis=0) # collapse C dimension
                
            else:
                # guided_grads is (T, C, H, W)
                guided_fused = guided_grads * fused_heatmap_np[:, np.newaxis, :, :]
                
                # Guided Saliency Channel Aggregation (Representation Fix)
                guided_gray = guided_fused.sum(axis=1) # collapse C dimension
                
            # Precision Outlier Clipping: Clip top 99.9% to prevent noise spikes yet preserve micro AU signals
            gf_vmax = np.percentile(guided_fused, 99.9)
            guided_fused = np.clip(guided_fused, a_min=None, a_max=gf_vmax)
                
            # Final Min-Max Normalization on Guided Fused
            gf_min, gf_max = guided_fused.min(), guided_fused.max()
            if gf_max > gf_min:
                guided_fused = (guided_fused - gf_min) / (gf_max - gf_min)
            else:
                guided_fused = np.zeros_like(guided_fused)
                
            # Final Mathematical Hardening: 99.9th percentile outlier clipper to the Guided Gray output
            gg_vmax = np.percentile(guided_gray, 99.9)
            guided_gray = np.clip(guided_gray, a_min=None, a_max=gg_vmax)
            
            gg_min, gg_max = guided_gray.min(), guided_gray.max()
            if gg_max > gg_min:
                guided_gray = (guided_gray - gg_min) / (gg_max - gg_min)
            else:
                guided_gray = np.zeros_like(guided_gray)
            
            return {
                "Modal Saliency": stream_heatmaps,
                "Guided Fused": guided_fused,
                "Guided Gray": guided_gray,
                "Temporal Importance": temporal_importance,
                "Apex Frame Index": apex_idx
            }
            
        finally:
            self.activations.clear()
            self.gradients.clear()
            self.model.zero_grad(set_to_none=True)
            
            # Automatic Graph Clearing to guarantee perfect memory parity:
            try: del target_score
            except NameError: pass
            except UnboundLocalError: pass
            
            try: del outputs
            except NameError: pass
            except UnboundLocalError: pass
            
            try: del logits
            except NameError: pass
            except UnboundLocalError: pass
            
            try: del heatmaps_list
            except NameError: pass
            except UnboundLocalError: pass
