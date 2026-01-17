#!/usr/bin/env python3
"""
Mock DiffusionPolicy Inference Engine for testing without real checkpoints

This creates a simple neural network that mimics the behavior of a real
DiffusionPolicy without requiring actual trained checkpoints.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, Any


class SimpleMockDiffusionPolicy(torch.nn.Module):
    """A simple mock diffusion policy that generates plausible actions"""
    
    def __init__(self, state_dim: int = 6, action_dim: int = 6, horizon: int = 16):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        
        # Simple MLP to process image features
        self.image_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
        )
        
        # Simple action decoder
        self.action_decoder = torch.nn.Sequential(
            torch.nn.Linear(64 + state_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, horizon * action_dim),
        )
    
    def forward(self, batch):
        """
        Args:
            batch: dict with keys:
                - "observation.images.front": (B, C, H, W) float tensor
                - "observation.state": (B, state_dim) float tensor
        
        Returns:
            actions: (B, horizon, action_dim) - but select_action expects (B, horizon * action_dim)
                     Actually looking at LeRobot, select_action returns list[tensor]
                     So we return [actions] where actions shape is (B, horizon, action_dim)
        """
        images = batch["observation.images.front"]  # (B, C, H, W)
        states = batch["observation.state"]  # (B, state_dim)
        
        # Encode image
        image_features = self.image_encoder(images)  # (B, 64)
        
        # Concatenate with state
        combined = torch.cat([image_features, states], dim=-1)  # (B, 64 + state_dim)
        
        # Decode to actions
        action_logits = self.action_decoder(combined)  # (B, horizon * action_dim)
        
        # Reshape to (B, horizon, action_dim)
        actions = action_logits.reshape(-1, self.horizon, self.action_dim)
        
        # Return in the format that LeRobot expects from select_action
        return (actions,)
    
    def select_action(self, batch):
        """Wrapper to match LeRobot's select_action interface"""
        return self.forward(batch)


class MockDiffusionPolicyConfig:
    """Mock config object"""
    def __init__(self, state_dim=6, action_dim=6, n_action_steps=16):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_size = action_dim
        self.n_action_steps = n_action_steps


class MockDiffusionPolicyInferenceEngine:
    """
    Mock DiffusionPolicy Inference Engine for testing

    This is a simplified version that doesn't need actual checkpoints.
    """
    
    def __init__(
        self,
        model_path: str = "mock",
        device: str = "cuda",
        verbose: bool = True,
        state_dim: int = 6,
        action_dim: int = 6,
        horizon: int = 16,
    ):
        """
        Initialize the mock inference engine
        
        Args:
            model_path: unused in mock, just for compatibility
            device: "cuda" or "cpu"
            verbose: whether to print info
            state_dim: state dimension
            action_dim: action dimension
            horizon: action horizon
        """
        self.device = torch.device(device)
        self.model_path = model_path
        self.verbose = verbose
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        
        # Create mock model
        self.model = SimpleMockDiffusionPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            horizon=horizon
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Mock config
        self.model.config = MockDiffusionPolicyConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            n_action_steps=horizon
        )
        
        # Mock normalize/unnormalize
        self.model.normalize_inputs = torch.nn.Identity()
        self.model.unnormalize_outputs = torch.nn.Identity()
        
        # Mock stats - simple identity transform
        self.stats = {
            "observation.state": {
                "mean": [0.0] * state_dim,
                "std": [1.0] * state_dim,
            },
            "observation.images.front": {
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
            },
            "action": {
                "mean": [0.0] * action_dim,
                "std": [1.0] * action_dim,
            }
        }
        
        if self.verbose:
            print(f"✓ Mock DiffusionPolicyInferenceEngine initialized")
            print(f"  State dim: {self.state_dim}")
            print(f"  Action dim: {self.action_dim}")
            print(f"  Horizon: {self.horizon}")
    
    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        state: np.ndarray,
        return_raw: bool = False
    ) -> np.ndarray:
        """
        Predict action sequence from image and state
        
        Args:
            image: (3, H, W) float32 [0, 1]
            state: (state_dim,) float32
            return_raw: whether to return raw normalized predictions
        
        Returns:
            actions: (horizon, action_dim) float32 array
        """
        # Prepare batch
        batch = {
            "observation.images.front": torch.from_numpy(image)
                .float()
                .to(self.device)
                .unsqueeze(0)  # (1, 3, H, W)
                .unsqueeze(0),  # (1, 1, 3, H, W)
            "observation.state": torch.from_numpy(state)
                .float()
                .to(self.device)
                .unsqueeze(0),  # (1, state_dim)
        }
        
        # Flatten image batch dimension for encoder
        B, T, C, H, W = batch["observation.images.front"].shape
        batch["observation.images.front"] = batch["observation.images.front"].reshape(B * T, C, H, W)
        
        # Run inference
        output = self.model.select_action(batch)
        actions = output[0].cpu().numpy()  # (1, horizon, action_dim)
        
        # Remove batch dimension
        actions = actions[0]  # (horizon, action_dim)
        
        if self.verbose:
            print(f"  [Mock Engine] Predicted action shape: {actions.shape}")
        
        return actions
    
    def reset_chunk(self):
        """Reset for compatibility"""
        pass
