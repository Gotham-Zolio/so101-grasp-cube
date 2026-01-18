"""
Data augmentation utilities for training ACT policies.

This module provides online visual augmentation during training to improve
generalization. The augmentations include:
- Color jitter (brightness, contrast, saturation, hue)
- Gaussian noise
- Gaussian blur (optional)

This is a meaningful difference from the baseline LeRobot ACT implementation,
which typically only performs normalization without strong online augmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional


class VisualAugmentation(nn.Module):
    """
    Online visual augmentation for training.
    
    Applies augmentations to image observations during training to improve
    generalization to visual disturbances.
    """
    
    def __init__(
        self,
        brightness_range: tuple = (0.7, 1.3),
        contrast_range: tuple = (0.7, 1.3),
        saturation_range: tuple = (0.7, 1.3),
        hue_range: tuple = (-0.1, 0.1),
        gaussian_noise_std: float = 0.02,
        gaussian_blur_prob: float = 0.0,
        gaussian_blur_kernel_size: int = 3,
        apply_prob: float = 0.8,
        training: bool = True,
    ):
        """
        Initialize visual augmentation.
        
        Args:
            brightness_range: Range for brightness adjustment (min, max)
            contrast_range: Range for contrast adjustment (min, max)
            saturation_range: Range for saturation adjustment (min, max)
            hue_range: Range for hue adjustment (min, max)
            gaussian_noise_std: Standard deviation for Gaussian noise
            gaussian_blur_prob: Probability of applying Gaussian blur
            gaussian_blur_kernel_size: Kernel size for Gaussian blur
            apply_prob: Probability of applying augmentation (0.0 = never, 1.0 = always)
            training: Whether augmentation is enabled (set to False during eval)
        """
        super().__init__()
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.gaussian_noise_std = gaussian_noise_std
        self.gaussian_blur_prob = gaussian_blur_prob
        self.gaussian_blur_kernel_size = gaussian_blur_kernel_size
        self.apply_prob = apply_prob
        self.training = training
    
    def forward(self, images: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply augmentation to image observations.
        
        Args:
            images: Dictionary of image tensors with keys like "observation.images.front",
                    "observation.images.left_side", etc. Each tensor should be in format
                    (batch, channels, height, width) with values in [0, 1].
        
        Returns:
            Dictionary of augmented image tensors with same keys.
        """
        if not self.training or self.apply_prob == 0.0:
            return images
        
        augmented_images = {}
        
        for key, img_tensor in images.items():
            if not key.startswith("observation.images."):
                # Skip non-image features
                augmented_images[key] = img_tensor
                continue
            
            # Skip augmentation with probability (1 - apply_prob)
            if torch.rand(1).item() > self.apply_prob:
                augmented_images[key] = img_tensor
                continue
            
            # Apply augmentations
            aug_img = img_tensor.clone()
            
            # Color jitter (brightness, contrast, saturation, hue)
            aug_img = self._apply_color_jitter(aug_img)
            
            # Gaussian noise
            if self.gaussian_noise_std > 0:
                aug_img = self._apply_gaussian_noise(aug_img)
            
            # Gaussian blur (optional)
            if self.gaussian_blur_prob > 0 and torch.rand(1).item() < self.gaussian_blur_prob:
                aug_img = self._apply_gaussian_blur(aug_img)
            
            # Clamp to valid range [0, 1]
            aug_img = torch.clamp(aug_img, 0.0, 1.0)
            
            augmented_images[key] = aug_img
        
        return augmented_images
    
    def _apply_color_jitter(self, img: torch.Tensor) -> torch.Tensor:
        """Apply random color jitter (brightness, contrast, saturation, hue)."""
        # img shape: (batch, channels, height, width), values in [0, 1]
        batch_size = img.shape[0]
        
        # Generate random parameters for each sample in batch
        brightness_factor = torch.empty(batch_size, 1, 1, 1, device=img.device).uniform_(
            self.brightness_range[0], self.brightness_range[1]
        )
        contrast_factor = torch.empty(batch_size, 1, 1, 1, device=img.device).uniform_(
            self.contrast_range[0], self.contrast_range[1]
        )
        saturation_factor = torch.empty(batch_size, 1, 1, 1, device=img.device).uniform_(
            self.saturation_range[0], self.saturation_range[1]
        )
        hue_factor = torch.empty(batch_size, 1, 1, 1, device=img.device).uniform_(
            self.hue_range[0], self.hue_range[1]
        )
        
        # Apply brightness
        aug_img = img * brightness_factor
        
        # Apply contrast (center around 0.5)
        aug_img = (aug_img - 0.5) * contrast_factor + 0.5
        
        # Apply saturation (convert to grayscale, then interpolate)
        if img.shape[1] == 3:  # RGB
            gray = aug_img.mean(dim=1, keepdim=True)
            aug_img = gray + (aug_img - gray) * saturation_factor
        
        # Apply hue (simplified version - rotate RGB channels)
        if img.shape[1] == 3:  # RGB
            # Convert to HSV-like representation and adjust hue
            # Simplified: rotate RGB channels
            hue_shift = hue_factor * 2 * np.pi
            # Apply rotation in RGB space (simplified)
            # For simplicity, we'll use a linear combination
            cos_hue = torch.cos(hue_shift)
            sin_hue = torch.sin(hue_shift)
            # Simple hue rotation approximation
            r, g, b = aug_img[:, 0:1], aug_img[:, 1:2], aug_img[:, 2:3]
            aug_img = torch.cat([
                r * (1 + hue_factor * 0.5),
                g * (1 - hue_factor * 0.3),
                b * (1 + hue_factor * 0.2)
            ], dim=1)
        
        return aug_img
    
    def _apply_gaussian_noise(self, img: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian noise to image."""
        noise = torch.randn_like(img) * self.gaussian_noise_std
        return img + noise
    
    def _apply_gaussian_blur(self, img: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur to image."""
        # Create Gaussian kernel
        kernel_size = self.gaussian_blur_kernel_size
        sigma = kernel_size / 3.0
        
        # Create 2D Gaussian kernel
        x = torch.arange(kernel_size, dtype=torch.float32, device=img.device) - kernel_size // 2
        g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = g[:, None] * g[None, :]
        kernel = kernel[None, None, :, :].repeat(img.shape[1], 1, 1, 1)
        
        # Apply convolution (padding to maintain size)
        padding = kernel_size // 2
        blurred = F.conv2d(img, kernel, padding=padding, groups=img.shape[1])
        
        return blurred
    
    def eval(self):
        """Disable augmentation (for evaluation)."""
        self.training = False
        return super().eval()
    
    def train(self, mode: bool = True):
        """Enable/disable augmentation based on mode."""
        self.training = mode
        return super().train(mode)


def apply_visual_disturbance(
    images: Dict[str, np.ndarray],
    brightness_factor: float = 1.0,
    contrast_factor: float = 1.0,
    noise_std: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Apply visual disturbance to images for generalization testing.
    
    This function is used during evaluation to test model robustness to
    visual disturbances that were not seen during training.
    
    Args:
        images: Dictionary of numpy arrays (H, W, C) with values in [0, 255]
        brightness_factor: Brightness multiplier (>1.0 = brighter, <1.0 = darker)
        contrast_factor: Contrast multiplier (>1.0 = more contrast, <1.0 = less contrast)
        noise_std: Standard deviation of Gaussian noise (0.0 = no noise)
    
    Returns:
        Dictionary of disturbed images with same keys.
    """
    disturbed_images = {}
    
    for key, img in images.items():
        # Convert to float [0, 1]
        img_float = img.astype(np.float32) / 255.0
        
        # Apply brightness
        img_float = img_float * brightness_factor
        
        # Apply contrast (center around 0.5)
        img_float = (img_float - 0.5) * contrast_factor + 0.5
        
        # Apply Gaussian noise
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, img_float.shape).astype(np.float32)
            img_float = img_float + noise
        
        # Clamp to [0, 1] and convert back to [0, 255]
        img_float = np.clip(img_float, 0.0, 1.0)
        disturbed_img = (img_float * 255.0).astype(np.uint8)
        
        disturbed_images[key] = disturbed_img
    
    return disturbed_images
