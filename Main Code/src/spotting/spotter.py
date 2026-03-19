import os
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import concurrent.futures

class ShreveStrainFilter(nn.Module):
    """
    Stage 1: Spatio-Temporal Strain Filtering (Shreve Algorithm).
    Isolates micro-expression candidate intervals via dense optical flow.
    """
    def __init__(self, fps: int = 200, window_size: int = 5):
        super().__init__()
        self.fps = fps
        self.window_size = window_size
        self.min_frames = int(0.04 * fps)
        self.max_frames = int(0.50 * fps)
        
        # Sobel filters for spatial gradients
        # Shape: (1, 1, 3, 3)
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], 
                                [-2.0, 0.0, 2.0], 
                                [-1.0, 0.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], 
                                [0.0,  0.0,  0.0], 
                                [1.0,   2.0,  1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def calculate_optical_flow_stack(self, gray_frames_np: np.ndarray) -> torch.Tensor:
        """
        Computes dense optical flow for the entire sequence utilizing multi-threading.
        Limits concurrency to prevent CPU oversubscription as OpenCV uses OpenMP.
        
        Args:
            gray_frames_np: Numpy array of shape (T, H, W) with uint8 values.
            
        Returns:
            Flow tensor on device.
            Shape: (T-1, 2, H, W)
        """
        t_seq, h, w = gray_frames_np.shape
        flows = np.zeros((t_seq - 1, 2, h, w), dtype=np.float32)
        
        def compute_flow_for_pair(t):
            flow = cv2.calcOpticalFlowFarneback(
                gray_frames_np[t], gray_frames_np[t+1], None, 
                pyr_scale=0.5, levels=3, winsize=15, 
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            return t, flow

        # Throttle ThreadPoolExecutor to prevent catastrophic context-switching overhead
        cpu_workers = max(1, os.cpu_count() // 2) if os.cpu_count() else 4
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_workers) as executor:
            futures = [executor.submit(compute_flow_for_pair, t) for t in range(t_seq - 1)]
            for future in concurrent.futures.as_completed(futures):
                t, flow = future.result()
                flows[t, 0] = flow[..., 0]
                flows[t, 1] = flow[..., 1]
            
        # Move globally constructed tensor array to GPU once
        # Shape: (T-1, 2, H, W)
        return torch.from_numpy(flows).to(self.sobel_x.device)

    def compute_batched_strain(self, flow_tensor: torch.Tensor) -> torch.Tensor:
        """
        Calculates Optical Strain Magnitude for all frames simultaneously.
        
        Args:
            flow_tensor: (T-1, 2, H, W) representing u and v flows.
            
        Returns:
            Strain magnitude tensor.
            Shape: (T-1, H, W)
        """
        # Extract u and v and format for batched conv2d
        # Shape: (T-1, 1, H, W)
        u_tensor = flow_tensor[:, 0:1, :, :]
        v_tensor = flow_tensor[:, 1:2, :, :]
        
        # Calculate spatial derivatives
        # Shape of outputs: (T-1, 1, H, W)
        u_x = F.conv2d(u_tensor, self.sobel_x, padding=1)
        u_y = F.conv2d(u_tensor, self.sobel_y, padding=1)
        v_x = F.conv2d(v_tensor, self.sobel_x, padding=1)
        v_y = F.conv2d(v_tensor, self.sobel_y, padding=1)
        
        # Strain tensor entries
        e_xx = u_x
        e_yy = v_y
        e_xy = 0.5 * (u_y + v_x)
        
        # Strain magnitude
        # Shape: (T-1, 1, H, W) -> squeezed to (T-1, H, W)
        strain_mag = torch.sqrt(e_xx**2 + e_yy**2 + 2 * (e_xy**2))
        return strain_mag.squeeze(1)

    def extract_roi_strains(self, strain_mag: torch.Tensor) -> torch.Tensor:
        """
        Splits the strain tensor into a 3x3 grid, extracting mean strain for 8 ROIs.
        
        Args:
            strain_mag: (T-1, H, W)
            
        Returns:
            Region strains excluding the center face.
            Shape: (T-1, 8)
        """
        t_seq_minus_1, h, w = strain_mag.shape
        h_step = h // 3
        w_step = w // 3
        
        # Align grid divisibility
        # Shape: (T-1, h_step*3, w_step*3)
        cropped_strain = strain_mag[:, :h_step*3, :w_step*3]
        
        # Reshape to 3x3 volumetric grid structure
        # Shape: (T-1, 3, h_step, 3, w_step)
        grid = cropped_strain.view(t_seq_minus_1, 3, h_step, 3, w_step)
        
        # Average geographically inside the cells
        # Shape: (T-1, 3, 3)
        grid_means = grid.mean(dim=(2, 4))
        
        # Flatten and isolate the 8 outer rings while eliminating index 4 (Nose region)
        # Shape: (T-1, 9) -> indexed to (T-1, 8)
        flat_means = grid_means.view(t_seq_minus_1, 9)
        roi_indices = [0, 1, 2, 3, 5, 6, 7, 8]
        
        return flat_means[:, roi_indices]
        
    def forward(self, video_tensor: torch.Tensor) -> list:
        """
        Finds micro-expression candidates sequentially using vectorized block ops.
        
        Args:
            video_tensor: (T, 3, H, W) float32 in [0, 1].
            
        Returns:
            List of tuples: [(start_idx, end_idx, confidence), ...]
        """
        t_seq = video_tensor.shape[0]
        if t_seq < 2:
            return []
            
        # 1. Grayscale extraction
        # Fixed point 3: Ensure C-Contiguous memory specifically aligning to OpenCV
        gray_frames = np.ascontiguousarray((video_tensor.mean(dim=1).cpu().numpy() * 255).astype(np.uint8))
        
        # 2. Dense optical vectors globally tracked (Multi-threaded)
        # Shape: (T-1, 2, H, W)
        flow_tensor = self.calculate_optical_flow_stack(gray_frames)
        
        # 3. Compute parallel structural strain
        # Shape: (T-1, H, W)
        strain_mag = self.compute_batched_strain(flow_tensor)
        
        # 4. Regional reduction constraints
        # Shape: (T-1, 8)
        roi_strains = self.extract_roi_strains(strain_mag)
        
        # Offload logic flags linearly for sliding window sequence extraction
        # Shape: (T-1, 8)
        t_strains = roi_strains.cpu().numpy()
        candidates = []
        
        num_frames = t_strains.shape[0]
        if num_frames < self.window_size:
            return candidates
            
        # 5. Independent parallel column thresholding.
        # Fixed point 2: Edge padding for convolution mode valid to prevent drops in moving average bounds
        mu_r = np.zeros_like(t_strains)
        kernel = np.ones(self.window_size) / self.window_size
        pad_size = self.window_size // 2
        
        for col in range(8):
            padded_col = np.pad(t_strains[:, col], (pad_size, pad_size), mode='edge')
            mu_r[:, col] = np.convolve(padded_col, kernel, mode='valid')[:num_frames]
            
        # Condition fulfilled if ANY ROI is drastically structurally displaced
        # Shape: (T-1,)
        is_peak = np.any(t_strains > 2.0 * mu_r, axis=1)
        
        # Max confidence extracted
        max_strain = np.max(t_strains, axis=1)
        
        active = False
        start_idx = 0
        
        for t in range(num_frames):
            if is_peak[t]:
                if not active:
                    active = True
                    start_idx = t
            else:
                if active:
                    duration = t - start_idx
                    if self.min_frames <= duration <= self.max_frames:
                        candidates.append((start_idx, t, float(np.max(max_strain[start_idx:t]))))
                    active = False
                    
        # Concluding buffer constraint check
        if active:
            duration = num_frames - start_idx
            if self.min_frames <= duration <= self.max_frames:
                candidates.append((start_idx, num_frames, float(np.max(max_strain[start_idx:]))))
                
        return candidates


class LiFFTApexLocator(nn.Module):
    """
    Stage 2: 3D-FFT Apex Localization (Li Algorithm).
    Highly optimized PyTorch implementation eliminating nested loops through blocked batched operations.
    """
    def __init__(self, d0_ratio: float = 0.15):
        """
        Args:
            d0_ratio: Threshold percentage to dynamically size the D_0 sphere limit.
        """
        super().__init__()
        self.d0_ratio = d0_ratio
        
        # Fixed point 4: Pre-register memory optimized constant LBP weight values into VRAM statically
        lbp_weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.float32).view(1, 8, 1, 1)
        self.register_buffer('lbp_weights', lbp_weights)
        
    def compute_batched_lbp(self, clip_tensor: torch.Tensor) -> torch.Tensor:
        """
        Vectorized generation of LBP texture structural maps mapped explicitly parallel mathematically.
        Highly optimized to avoid VRAM exhaustion from intermediate tensors.
        
        Args:
            clip_tensor: (T, 1, H, W)
            
        Returns:
            LBP representation integer map.
            Shape: (T, 1, H, W)
        """
        t_seq, c, h, w = clip_tensor.shape
        
        # Unfold geometry extraction (unrolling 3x3 neighborhoods)
        # Shape: (T, 9, H*W)
        unfold = F.unfold(clip_tensor, kernel_size=3, padding=1)
        
        # Relocate shape logic dimensions
        # Shape: (T, 9, H, W)
        unfold = unfold.view(t_seq, 9, h, w)
        
        # Compare vs kernel center (Index 4 maps perfectly to the unroll center of a 3x3)
        # Shape: (T, 1, H, W)
        center = unfold[:, 4:5, :, :]
        
        # Stack the 8 neighborhood bit masks into a single tensor structurally
        # Shape: (T, 8, H, W)
        indices = [0, 1, 2, 3, 5, 6, 7, 8]
        neighbors = unfold[:, indices, :, :]
        
        bits = (neighbors >= center).float()
        
        # Accumulate bit map directly utilizing pre-allocated broadcast vectors matching structural layouts
        # Shape: (T, 1, H, W)
        lbp = (bits * self.lbp_weights).sum(dim=1, keepdim=True)
        return lbp
        
    def generate_dynamic_hbf_mask(self, window_size: int, h_block: int, w_block: int, device: torch.device) -> torch.Tensor:
        """
        Calculates High Frequency Bandpass Filter explicitly tied to regional max scaling coordinates.
        Uses precise integer coordinates to perfectly align origin with fftshift logic.
        
        Args:
            window_size: Temporal depth of structure.
            h_block: Height of extracted sub-cell.
            w_block: Width of extracted sub-cell.
            device: Tensor deployment device mapping.
            
        Returns:
            HBF Volume Filter Match.
            Shape: (1, 1, window_size, h_block, w_block)
        """
        cz, cy, cx = window_size // 2, h_block // 2, w_block // 2
        
        # Geometric grids mapped perfectly across frequency mapping sizes
        z = torch.arange(window_size, device=device).view(-1, 1, 1)
        y = torch.arange(h_block, device=device).view(1, -1, 1)
        x = torch.arange(w_block, device=device).view(1, 1, -1)
        
        dist = torch.sqrt((z - cz)**2 + (y - cy)**2 + (x - cx)**2)
        
        # Calculate maximum block bounding container sphere
        max_dist = math.sqrt(cz**2 + cy**2 + cx**2)
        d0 = self.d0_ratio * max_dist
        
        mask = (dist > d0).float()
        return mask.view(1, 1, window_size, h_block, w_block)
        
    def forward(self, clip_tensor: torch.Tensor) -> int:
        """
        Pinpoints exact apex metric optimally using pure volume transformation manipulation.
        Evaluates every frame symmetrically safely via temporal replicate padding.
        
        Args:
            clip_tensor: (T, 3, H, W)
            
        Returns:
            Apex frame index mapped internally pointing.
        """
        device = clip_tensor.device
        t_seq, _, h, w = clip_tensor.shape
        if t_seq == 0:
            return 0
            
        # Convert identically into a grayscale mapping frame batch
        # Shape: (T, 1, H, W)
        gray_clip = clip_tensor.mean(dim=1, keepdim=True)
        
        # Full batched textural LBP conversion. No loops.
        # Shape: (T, 1, H, W)
        lbp_clip = self.compute_batched_lbp(gray_clip)
        
        # Grid dimensional mapping alignment to mathematically support integer splitting
        h_step, w_step = h // 6, w // 6
        cropped_h, cropped_w = h_step * 6, w_step * 6
        
        # Shape: (T, 1, cropped_h, cropped_w)
        lbp_clip = lbp_clip[:, :, :cropped_h, :cropped_w]
        
        window_size = min(5, t_seq)
        pad_size = window_size // 2
        
        # Replicate padding deployed temporally resolving all sliding window boundary bottlenecks
        # Permuting T to the right mathematically safely triggers 1D padding dynamically
        # Shape transformations: (T, 1, H, W) -> (1, H, W, T) -> padded -> (padded_T, 1, H, W)
        lbp_clip_permuted = lbp_clip.permute(1, 2, 3, 0)
        lbp_clip_padded_permuted = F.pad(lbp_clip_permuted, (pad_size, pad_size), mode='replicate')
        lbp_clip_padded = lbp_clip_padded_permuted.permute(3, 0, 1, 2)
        
        max_amplitude = -1.0
        apex_idx = t_seq // 2
        
        # Extract once structurally mapping memory optimization correctly over full loops
        hbf_mask = self.generate_dynamic_hbf_mask(window_size, h_step, w_step, device)
        
        # Safely iterates perfectly capturing exact origin bounds via centered zero indexing
        for t in range(t_seq):
            start = t
            end = t + window_size
            
            # Extract sliding temporal window structurally
            # Shape: (Window, 1, cropped_h, cropped_w)
            window = lbp_clip_padded[start:end]
            
            # Map completely natively to blocked unrolled geometries:
            # Replace view with reshape preventing contiguity RuntimeErrors from safely slicing contiguous boundaries
            blocks = window.reshape(window_size, 1, 6, h_step, 6, w_step)
            
            # Relocate block grids into batched sequence arrays
            # Source -> (0:Window, 1:C, 2:GridY, 3:BlockY, 4:GridX, 5:BlockX)
            # Target -> (1:C, 2:GridY, 4:GridX, 0:Window, 3:BlockY, 5:BlockX)
            blocks = blocks.permute(1, 2, 4, 0, 3, 5)
            
            # Flat reduction
            # Shape: (1, 36, Window, H_block, W_block)
            blocks_flat = blocks.reshape(1, 36, window_size, h_step, w_step)
            
            # DC-Centering: Subtract structural mean spatially/temporally preventing amplitude blowouts
            blocks_mean = blocks_flat.mean(dim=(-3, -2, -1), keepdim=True)
            blocks_centered = blocks_flat - blocks_mean
            
            # Full parallel 3D dimensional frequency FFT processing strictly ignoring mapped boundaries.
            fft_res = torch.fft.fftn(blocks_centered, dim=(-3, -2, -1))
            fft_shift = torch.fft.fftshift(fft_res, dim=(-3, -2, -1))
            
            # High frequency threshold tracking multiplier
            # Broadcast mapping works directly covering (1, 36, window_size, h_step, w_step) 
            filtered_mag = torch.abs(fft_shift) * hbf_mask
            
            # Amplitude reduction globally capturing energy signals
            accum_high_freq = filtered_mag.sum().item()
            
            if accum_high_freq > max_amplitude:
                max_amplitude = accum_high_freq
                apex_idx = t
                
        return apex_idx

class MicroExpressionSpotter(nn.Module):
    """
    Two-stage Spotting Module.
    Combines Shreve Strain Filtering for isolation and Li 3D-FFT for apex localization.
    """
    def __init__(self, fps: int = 200, strain_window: int = 5, fft_d0_ratio: float = 0.15):
        super().__init__()
        self.stage1_filter = ShreveStrainFilter(fps=fps, window_size=strain_window)
        self.stage2_locator = LiFFTApexLocator(d0_ratio=fft_d0_ratio)
        
    @torch.no_grad()
    def forward(self, video_tensor: torch.Tensor) -> list:
        """
        Identifies peaks seamlessly end-to-end structurally.
        video_tensor: (T, 3, H, W) float32 generic tensor.
        Returns a list of dictionaries with spotted micro-expression metrics.
        """
        # Stage 1: candidate isolation mapping directly to GPU
        candidates = self.stage1_filter(video_tensor)
        
        results = []
        for start_idx, end_idx, conf in candidates:
            # Fix slice amputation by expanding end_idx properly bounds
            start_idx = max(0, start_idx)
            end_idx = min(video_tensor.shape[0] - 1, end_idx)
            
            # Include +1 interval bound dynamically recovering exact valid sequence lengths
            clip = video_tensor[start_idx : end_idx + 1]
            if clip.shape[0] < 3:
                continue
                
            # Stage 2: Parallel block batched pinpointing execution natively pointing arrays.
            local_apex = self.stage2_locator(clip)
            global_apex = start_idx + local_apex
            
            results.append({
                'start': start_idx,
                'end': end_idx,
                'apex': global_apex,
                'confidence': conf
            })
            
        return results
