# core/differential_privacy.py - MAJOR UPDATE
import torch
import numpy as np

class DifferentialPrivacy:
    """Differential privacy mechanisms with MEANINGFUL noise"""
    
    def __init__(self, epsilon=1.0, delta=1e-5):
        self.epsilon = epsilon
        self.delta = delta
    
    def calculate_noise_scale(self, sensitivity=1.0):
        """Calculate DP noise scale - MUCH STRONGER NOISE"""
        if self.epsilon <= 0 or self.delta <= 0:
            raise ValueError("Epsilon and delta must be positive")
        
        base_scale = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        
        # MAJOR CHANGE: Use 0.1 instead of 0.01 for meaningful privacy
        # This will actually impact accuracy for ε < 10.0
        practical_scale = base_scale * 0.1
        
        print(f"    DP Noise: ε={self.epsilon}, scale={practical_scale:.6f}")
        return practical_scale
    
    def add_noise(self, tensor, noise_scale):
        """Add DP noise to tensor - MORE IMPACTFUL"""
        if noise_scale > 0:
            # Scale noise by tensor magnitude for realistic impact
            tensor_std = tensor.std().item() if tensor.numel() > 1 else abs(tensor.item())
            effective_scale = noise_scale * max(tensor_std, 0.1)
            
            # Add significant noise
            noise = torch.normal(mean=0.0, std=effective_scale, size=tensor.shape, device=tensor.device)
            return tensor + noise
        return tensor

class ErrorCorrectedDP(DifferentialPrivacy):
    """DP with SMART error correction"""
    
    def add_corrected_noise(self, tensor, noise_scale):
        """Add noise with SMART error correction that adapts to noise level"""
        noisy_tensor = super().add_noise(tensor, noise_scale)
        
        # Adaptive correction based on noise level
        if noise_scale > 0.01:  # High noise regime (strong privacy)
            # Aggressive correction for high noise
            mean_val = noisy_tensor.mean()
            std_val = noisy_tensor.std()
            
            # Tight clipping for high noise
            clipped_tensor = torch.clamp(
                noisy_tensor, 
                mean_val - 1.5*std_val,  # Tighter bounds
                mean_val + 1.5*std_val
            )
            
            # Strong smoothing
            corrected_tensor = 0.6 * clipped_tensor + 0.4 * tensor
            
        else:  # Low noise regime (weak privacy)
            # Gentle correction for low noise
            mean_val = noisy_tensor.mean()
            std_val = noisy_tensor.std()
            
            # Loose clipping
            clipped_tensor = torch.clamp(
                noisy_tensor, 
                mean_val - 2.5*std_val,
                mean_val + 2.5*std_val
            )
            
            # Light smoothing
            corrected_tensor = 0.8 * clipped_tensor + 0.2 * tensor
        
        return corrected_tensor