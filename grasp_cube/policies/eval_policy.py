"""
Evaluation interface for deployable Diffusion Policy.

Implements the get_actions function required for real-world deployment.
"""

import pathlib
from typing import Dict, Any, Optional
import numpy as np
import torch

try:
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig, PreTrainedConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("Warning: LeRobot not installed. Install it to use Diffusion Policy.")


class DeployablePolicy:
    """
    Deployable policy interface that implements get_actions function.
    
    This class wraps a trained Diffusion Policy and provides the interface
    required for real-world deployment.
    """
    
    def __init__(self, policy_path: pathlib.Path, device: str = "cuda", robot_type: str = "bi_so101"):
        """
        Initialize deployable policy.
        
        Args:
            policy_path: Path to trained policy directory
            device: Device to run inference on ("cuda" or "cpu")
            robot_type: Robot type ("so101" for single-arm, "bi_so101" for dual-arm)
        """
        if not LEROBOT_AVAILABLE:
            raise ImportError("LeRobot is required for Diffusion Policy. Please install it.")
        
        self.device = torch.device(device)
        self.robot_type = robot_type
        
        # Load policy configuration
        policy_config = PreTrainedConfig.from_pretrained(str(policy_path))
        assert isinstance(policy_config, DiffusionConfig), \
            f"Expected DiffusionConfig, got {type(policy_config)}"
        
        # Load policy model
        self.policy = DiffusionPolicy.from_pretrained(str(policy_path), config=policy_config)
        self.policy.to(self.device)
        self.policy.eval()
        
        # Store config for normalization
        self.policy_config = policy_config
        self.n_obs_steps = policy_config.n_obs_steps
        self.horizon = policy_config.horizon
        
        # Load normalization stats from checkpoint if available
        import json
        stats_file = pathlib.Path(policy_path) / "training_info.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                training_info = json.load(f)
                self.dataset_stats = training_info.get("dataset_stats", {})
        else:
            self.dataset_stats = {}
    
    def get_actions(self, observation: Dict[str, Any]) -> np.ndarray:
        """
        Get actions from observation.
        
        This is the main interface function required for deployment.
        
        Args:
            observation: Dict with keys:
                - "states": Dict with "arm" or "left_arm"/"right_arm" (robot qpos)
                - "images": Dict with "front" and "wrist" or "left_wrist"/"right_wrist"
                - "task": Task prompt string
        
        Returns:
            Action array (joint positions) for the next step
        """
        # Prepare observation in LeRobot format
        # Get state
        if self.robot_type == "bi_so101":
            left_arm = observation["states"].get("left_arm", np.zeros(6))
            right_arm = observation["states"].get("right_arm", np.zeros(6))
            state = np.concatenate([left_arm, right_arm], axis=-1)
        else:
            state = observation["states"].get("arm", np.zeros(6))
        
        # Convert to tensor - select_action expects single timestep observations (batch dimension only)
        # State: (batch_size, state_dim)
        state_tensor = torch.from_numpy(state).float().to(self.device)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)  # (1, state_dim)
        
        # Get images
        front_img = observation["images"].get("front", np.zeros((480, 640, 3), dtype=np.uint8))
        # Convert to tensor: (H, W, C) -> (1, C, H, W)
        front_img_tensor = torch.from_numpy(front_img).float()
        if front_img_tensor.dim() == 3:
            front_img_tensor = front_img_tensor.permute(2, 0, 1)  # (C, H, W)
        front_img_tensor = front_img_tensor / 255.0  # Normalize to [0, 1]
        front_img_tensor = front_img_tensor.unsqueeze(0).to(self.device)  # (1, C, H, W)
        
        # Prepare batch dict for select_action (single timestep, not stacked)
        # select_action will handle queue management and stacking internally
        model_batch = {
            "observation.state": state_tensor,  # (1, state_dim)
            "observation.images.front": front_img_tensor,  # (1, C, H, W)
        }
        
        # Add wrist cameras if available
        if self.robot_type == "bi_so101":
            left_wrist = observation["images"].get("left_wrist", np.zeros((480, 640, 3), dtype=np.uint8))
            right_wrist = observation["images"].get("right_wrist", np.zeros((480, 640, 3), dtype=np.uint8))
            
            left_wrist_tensor = torch.from_numpy(left_wrist).float()
            if left_wrist_tensor.dim() == 3:
                left_wrist_tensor = left_wrist_tensor.permute(2, 0, 1)  # (C, H, W)
            left_wrist_tensor = left_wrist_tensor / 255.0
            left_wrist_tensor = left_wrist_tensor.unsqueeze(0).to(self.device)  # (1, C, H, W)
            
            right_wrist_tensor = torch.from_numpy(right_wrist).float()
            if right_wrist_tensor.dim() == 3:
                right_wrist_tensor = right_wrist_tensor.permute(2, 0, 1)  # (C, H, W)
            right_wrist_tensor = right_wrist_tensor / 255.0
            right_wrist_tensor = right_wrist_tensor.unsqueeze(0).to(self.device)  # (1, C, H, W)
            
            model_batch["observation.images.left_wrist"] = left_wrist_tensor
            model_batch["observation.images.right_wrist"] = right_wrist_tensor
        else:
            wrist = observation["images"].get("wrist", np.zeros((480, 640, 3), dtype=np.uint8))
            wrist_tensor = torch.from_numpy(wrist).float()
            if wrist_tensor.dim() == 3:
                wrist_tensor = wrist_tensor.permute(2, 0, 1)  # (C, H, W)
            wrist_tensor = wrist_tensor / 255.0
            wrist_tensor = wrist_tensor.unsqueeze(0).to(self.device)  # (1, C, H, W)
            model_batch["observation.images.wrist"] = wrist_tensor
        
        # Use select_action which handles queue management automatically
        # It will populate queues, generate action chunks when needed, and return a single action
        with torch.no_grad():
            action = self.policy.select_action(model_batch)  # (action_dim,)
        
        # Convert to numpy
        action = action.cpu().numpy()
        
        return action
    
    def reset(self):
        """Reset policy state (for episode boundaries)."""
        self.policy.reset()


def create_policy_from_config(policy_path: pathlib.Path, 
                              device: str = "cuda",
                              robot_type: str = "bi_so101") -> DeployablePolicy:
    """
    Factory function to create a deployable policy from a trained model.
    
    Args:
        policy_path: Path to trained policy directory
        device: Device to run inference on
        robot_type: Robot type
    
    Returns:
        DeployablePolicy instance
    """
    return DeployablePolicy(policy_path, device=device, robot_type=robot_type)


if __name__ == "__main__":
    from dataclasses import dataclass
    import tyro
    
    @dataclass
    class Args:
        policy_path: pathlib.Path
        device: str = "cuda"
        robot_type: str = "bi_so101"
    
    args = tyro.cli(Args)
    policy = create_policy_from_config(args.policy_path, args.device, args.robot_type)
    print("Deployable Policy initialized successfully!")
