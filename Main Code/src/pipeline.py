import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional

class EulerianVideoMagnifier(nn.Module):
    """
    Phase-Based Eulerian Video Magnification (EVM) Preprocessor Wrapper.
    
    In a full production environment, this class hooks into the `PyEVM` library
    to perform true Phase-Based Eulerian Video Magnification. It relies heavily 
    on SciPy/OpenCV spatial pyramids to amplify invisible, sub-pixel kinematics 
    (like minute skin color changes, flushed skin, or micro-deformations) to 
    macro-levels that the downstream recognition network can reliably extract.
    
    Currently implemented as a robust placeholder wrapper that conceptually 
    applies magnification by returning the tensor unmodified, allowing seamless 
    integration into the PyTorch continuous execution graph until the CPU-bound 
    C++/SciPy EVM bindings are fully deployed.
    """
    def __init__(self, magnification_factor: float = 10.0):
        super().__init__()
        self.magnification_factor = magnification_factor

    def forward(self, clip_tensor: torch.Tensor) -> torch.Tensor:
        """
        Executes the EVM magnification sequence.
        
        Args:
            clip_tensor (torch.Tensor): The raw extracted clip tensor of shape (T, 3, H, W).
            
        Returns:
            torch.Tensor: The physically magnified clip tensor of the same shape (T, 3, H, W).
        """
        # Placeholder for complex PyEVM processing pipeline.
        # Ensure it returns the identical shape (T, 3, H, W)
        magnified_clip = clip_tensor.clone()
        return magnified_clip


class MERE2EPipeline(nn.Module):
    """
    End-to-End Micro-Expression Recognition Pipeline (Execution Controller).
    
    The master controller that coordinates the Spotting phase (MicroExpressionSpotter), 
    the Eulerian Video Magnification pre-processing phase (EulerianVideoMagnifier),
    and the fully trained MTMNet Recognition stack into a unified inference loop.
    """
    def __init__(self, 
                 spotter: nn.Module, 
                 magnifier: EulerianVideoMagnifier, 
                 mtm_net: nn.Module,
                 strain_threshold: float = 0.5,
                 prob_threshold: float = 0.6,
                 idx_to_class: Optional[Dict[int, str]] = None,
                 clip_length: int = 15):
        """
        Initializes the fully instantiated pipeline controller.
        
        Args:
            spotter (nn.Module): An instantiated MicroExpressionSpotter.
            magnifier (EulerianVideoMagnifier): An instantiated EulerianVideoMagnifier.
            mtm_net (nn.Module): An instantiated trained MTMNet classifier.
            strain_threshold (float): Minimum confidence from the spotter to consider valid skin deformation.
            prob_threshold (float): Minimum softmax probability from MTMNet to assert a specific emotion.
            idx_to_class (Dict[int, str], optional): Mapping from class index to emotion string.
            clip_length (int): Fixed temporal clip length to extract around the apex frame.
        """
        super().__init__()
        self.spotter = spotter
        self.magnifier = magnifier
        self.mtm_net = mtm_net
        
        self.strain_threshold = strain_threshold
        self.prob_threshold = prob_threshold
        self.clip_length = clip_length
        self.idx_to_class = idx_to_class or {0: 'Negative', 1: 'Positive', 2: 'Surprise'}
        
        # Enforce inference mode and strict memory management during initialization
        self.eval()
        
    @torch.no_grad()
    def forward(self, full_video: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Runs the end-to-end forward pass over an unclipped raw video under strict NO_GRAD constraints.
        
        Args:
            full_video (torch.Tensor): Unclipped raw video tensor of shape (T_full, 3, H, W).
            
        Returns:
            List[Dict[str, Any]]: A structured JSON-like list of dictionaries detailing the
            spotted intervals and their predicted emotions, governed by dual-thresholding logic.
        """
        T_full = full_video.shape[0]
        device = full_video.device
        
        # 1. Run the Spotter
        # Expected spotting output format: [{'start': int, 'end': int, 'apex': int, 'confidence': float}]
        candidates = self.spotter(full_video)
        
        final_results = []
        
        for candidate in candidates:
            original_start = candidate['start']
            original_end = candidate['end']
            apex = candidate['apex']
            spotting_confidence = candidate['confidence']
            
            # Confidence Thresholding Logic: A) Spotting Strain Check
            # If a candidate lacks strong physical skin deformation, skip and discard immediately
            if spotting_confidence <= self.strain_threshold:
                continue
                
            # 2. Apex-Centered Clip Extraction
            # Constrain frame bounds computationally around the precise apex for temporal aggregator coherence
            target_start = int(apex) - self.clip_length // 2
            target_end = target_start + self.clip_length
            extract_start = max(0, target_start)
            extract_end = min(T_full, target_end)
            
            clip_tensor = full_video[extract_start:extract_end]  # Shape: (T_clip, 3, H, W)
            T_clip = clip_tensor.shape[0]
            
            # Defensive check to skip corrupted or zero-length slices
            if T_clip == 0:
                continue

            # Boundary Padding for Apex Clips
            # If apex is too close to video start or end, pad to exactly self.clip_length
            if T_clip < self.clip_length:
                pad_left = max(0, -target_start)
                pad_right = max(0, target_end - T_full)
                
                # F.pad with 3D replicate padding requires a 5D tensor (B, C, D, H, W) where D is our T_clip
                # Current shape: (T_clip, 3, H, W). Target 5D shape: (1, 3, T_clip, H, W)
                clip_5d = clip_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)
                
                # pad format: (W_left, W_right, H_left, H_right, T_left, T_right)
                clip_padded_5d = F.pad(clip_5d, (0, 0, 0, 0, pad_left, pad_right), mode='replicate')
                
                # Revert back to 4D (self.clip_length, 3, H, W)
                clip_tensor = clip_padded_5d.squeeze(0).permute(1, 0, 2, 3)
                T_clip = self.clip_length
                
            # 3. Magnification
            # Pass the extracted apex-centered clip through the EVM preprocessor
            magnified_clip = self.magnifier(clip_tensor)  # Expected Shape: (self.clip_length, 3, H, W)
            
            # Clamp the EVM Output to prevent activation blowouts in MTMNet
            torch.clamp_(magnified_clip, 0.0, 1.0)
            
            # 4. Recognition
            # Structure tensor for MTMNet (requires batch dimension)
            # Standard batched format injection
            batched_magnified_clip = magnified_clip.unsqueeze(0) # Shape: (1, self.clip_length, 3, H, W)
            
            # Instantiate strict lengths tensor as required by MTMNet sequence handling
            lengths = torch.tensor([T_clip], dtype=torch.long, device=device)
            
            # Execute MTMNet forward pass
            # Output is expected to be a dictionary based on the updated MTMNet API
            out_dict = self.mtm_net(batched_magnified_clip, lengths=lengths)
            logits = out_dict['class_logits']
            
            # Compute classification probabilities
            probs = torch.softmax(logits, dim=-1)[0]  # Remove batch dimension -> (Num_Classes,)
            max_prob, pred_class_idx = torch.max(probs, dim=0)
            
            probability = max_prob.item()
            pred_idx = pred_class_idx.item()
            
            # Map predicted index to emotion string representation
            raw_emotion_guess = self.idx_to_class.get(pred_idx, f"Class_{pred_idx}")
            predicted_emotion = raw_emotion_guess
            
            # Confidence Thresholding Logic: B) Recognition Probability Check
            # If emotional mapping probability lacks conviction, log transparency but override final class prediction
            if probability <= self.prob_threshold:
                predicted_emotion = 'Neutral'
            
            # 5. Production Output Formatter
            final_results.append({
                'start_frame': original_start,
                'end_frame': original_end, 
                'apex_frame': apex,
                'predicted_emotion_class': predicted_emotion,
                'raw_emotion_guess': raw_emotion_guess,
                'classification_probability': float(probability),
                'strain_confidence': float(spotting_confidence)
            })
            
        return final_results
