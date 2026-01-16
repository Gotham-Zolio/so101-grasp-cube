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
        
        # Debug: print loaded config
        if hasattr(policy_config, 'input_features'):
            input_features = getattr(policy_config, 'input_features', None)
            if input_features is not None and isinstance(input_features, dict):
                feature_keys = [str(k) for k in input_features.keys()]
                print(f"Loaded model config - Input features: {feature_keys}")
        
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
        # Check what's actually in the observation to determine single-arm vs dual-arm
        # This is more reliable than relying on self.robot_type
        states_dict = observation.get("states", {})
        if "left_arm" in states_dict and "right_arm" in states_dict:
            # Dual-arm observation
            left_arm = np.asarray(states_dict["left_arm"]).flatten()
            right_arm = np.asarray(states_dict["right_arm"]).flatten()
            # Ensure each arm is 6-dim
            if len(left_arm) > 6:
                left_arm = left_arm[:6]
            if len(right_arm) > 6:
                right_arm = right_arm[:6]
            state = np.concatenate([left_arm, right_arm], axis=-1)
        elif "arm" in states_dict:
            # Single-arm observation
            state = np.asarray(states_dict["arm"]).flatten()
            # Ensure state is 6-dim (not 12-dim)
            if len(state) > 6:
                state = state[:6]
        else:
            # Fallback: try to infer from robot_type
            if self.robot_type == "bi_so101":
                left_arm = np.asarray(states_dict.get("left_arm", np.zeros(6))).flatten()[:6]
                right_arm = np.asarray(states_dict.get("right_arm", np.zeros(6))).flatten()[:6]
                state = np.concatenate([left_arm, right_arm], axis=-1)
            else:
                state = np.asarray(states_dict.get("arm", np.zeros(6))).flatten()[:6]
        
        # Final check: ensure state dimension matches model expectation
        # Check model's expected state dimension from config
        if hasattr(self, 'policy_config') and self.policy_config is not None:
            input_features = getattr(self.policy_config, 'input_features', None)
            if input_features is not None and isinstance(input_features, dict):
                state_feature = input_features.get("observation.state", None)
                if state_feature is not None:
                    # Get expected state dimension from config
                    if isinstance(state_feature, dict):
                        expected_state_dim = state_feature.get("shape", [6])[0]
                    else:
                        # PolicyFeature object
                        expected_state_dim = state_feature.shape[0] if hasattr(state_feature, 'shape') else 6
                    
                    # Adjust state to match expected dimension
                    if len(state) != expected_state_dim:
                        if len(state) > expected_state_dim:
                            state = state[:expected_state_dim]
                        else:
                            # Pad with zeros if too short
                            state = np.pad(state, (0, expected_state_dim - len(state)), mode='constant')
        
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
        
        # Determine what cameras the model expects based on config
        # Check model's input_features to see what it was trained with
        # Start with only state and front (always included)
        model_batch = {
            "observation.state": state_tensor,  # (1, state_dim)
            "observation.images.front": front_img_tensor,  # (1, C, H, W)
        }
        
        # Check model's input_features to see if it expects wrist cameras
        # Only add wrist cameras if they're in the model's input_features
        if hasattr(self, 'policy_config') and self.policy_config is not None:
            input_features = getattr(self.policy_config, 'input_features', None)
            if input_features is not None and isinstance(input_features, dict):
                # Get all feature keys (convert to strings for comparison)
                feature_keys = list(input_features.keys())
                feature_keys_str = [str(k) for k in feature_keys]
                
                # Debug: print expected features (only once)
                if not hasattr(self, '_printed_features'):
                    print(f"Model expects input features: {feature_keys_str}")
                    self._printed_features = True
                
                # Check if model has wrist cameras in input_features
                has_left_wrist = "observation.images.left_wrist" in feature_keys or any("left_wrist" in k for k in feature_keys_str)
                has_right_wrist = "observation.images.right_wrist" in feature_keys or any("right_wrist" in k for k in feature_keys_str)
                has_wrist = "observation.images.wrist" in feature_keys or any(k == "observation.images.wrist" or (k.endswith("wrist") and "left" not in k.lower() and "right" not in k.lower()) for k in feature_keys_str)
                
                # Only add wrist cameras if model expects them
                if has_left_wrist or has_right_wrist:
                    # Model expects left_wrist and/or right_wrist
                    # For single-arm tasks, observation may have "right_wrist" but not "left_wrist"
                    # For dual-arm tasks, observation should have both "left_wrist" and "right_wrist"
                    left_wrist = observation["images"].get("left_wrist", np.zeros((480, 640, 3), dtype=np.uint8))
                    right_wrist = observation["images"].get("right_wrist", np.zeros((480, 640, 3), dtype=np.uint8))
                    
                    if has_left_wrist:
                        left_wrist_tensor = torch.from_numpy(left_wrist).float()
                        if left_wrist_tensor.dim() == 3:
                            left_wrist_tensor = left_wrist_tensor.permute(2, 0, 1)  # (C, H, W)
                        left_wrist_tensor = left_wrist_tensor / 255.0
                        left_wrist_tensor = left_wrist_tensor.unsqueeze(0).to(self.device)  # (1, C, H, W)
                        model_batch["observation.images.left_wrist"] = left_wrist_tensor
                    
                    if has_right_wrist:
                        right_wrist_tensor = torch.from_numpy(right_wrist).float()
                        if right_wrist_tensor.dim() == 3:
                            right_wrist_tensor = right_wrist_tensor.permute(2, 0, 1)  # (C, H, W)
                        right_wrist_tensor = right_wrist_tensor / 255.0
                        right_wrist_tensor = right_wrist_tensor.unsqueeze(0).to(self.device)  # (1, C, H, W)
                        model_batch["observation.images.right_wrist"] = right_wrist_tensor
                elif has_wrist:
                    # Model expects single wrist camera
                    wrist = observation["images"].get("wrist", np.zeros((480, 640, 3), dtype=np.uint8))
                    wrist_tensor = torch.from_numpy(wrist).float()
                    if wrist_tensor.dim() == 3:
                        wrist_tensor = wrist_tensor.permute(2, 0, 1)  # (C, H, W)
                    wrist_tensor = wrist_tensor / 255.0
                    wrist_tensor = wrist_tensor.unsqueeze(0).to(self.device)  # (1, C, H, W)
                    model_batch["observation.images.wrist"] = wrist_tensor
                # If model doesn't expect any wrist cameras, don't add them (model_batch already has only state and front)
        
        # Debug: print what we're sending to select_action
        if not hasattr(self, '_printed_batch'):
            print(f"Model batch keys: {list(model_batch.keys())}")
            print(f"Model batch shapes:")
            for k, v in model_batch.items():
                print(f"  {k}: {v.shape}")
            self._printed_batch = True
        
        # Use select_action which handles queue management automatically
        # It will populate queues, generate action chunks when needed, and return a single action
        with torch.no_grad():
            action = self.policy.select_action(model_batch)  # (action_dim,)
        
        # Convert to numpy
        action = action.cpu().numpy()
        
        return action
    
    def reset(self):
        """Reset policy state (for episode boundaries)."""
        # Reset observation queues in the policy
        self.policy.reset()
        
        # Reset printed flags
        if hasattr(self, '_printed_features'):
            delattr(self, '_printed_features')
        if hasattr(self, '_printed_batch'):
            delattr(self, '_printed_batch')


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
