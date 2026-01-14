"""
LeRobot Diffusion Policy wrapper for real robot deployment.

This module provides a wrapper around LeRobot's Diffusion Policy for real-world deployment,
similar to the ACT policy wrapper in act_policy.py.
"""

import dataclasses
import tyro
import pathlib
import numpy as np
import torch
from typing import Any, Literal

try:
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig, PreTrainedConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.utils import prepare_observation_for_inference
    from lerobot.policies.factory import make_pre_post_processors
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("Warning: LeRobot not installed. Install it to use Diffusion Policy.")


@dataclasses.dataclass
class LeRobotDiffusionPolicyConfig:
    path: pathlib.Path
    robot_type: Literal["so101", "bi_so101"] = "bi_so101"
    device: str = "cuda"
    act_steps: int | None = None  # Number of action steps to execute from chunk


class LeRobotDiffusionPolicy:
    """
    Wrapper for LeRobot Diffusion Policy for real robot deployment.
    
    This class provides the get_actions interface required for deployment,
    handling observation preprocessing and action postprocessing.
    """
    
    def __init__(self, config: LeRobotDiffusionPolicyConfig):
        if not LEROBOT_AVAILABLE:
            raise ImportError("LeRobot is required for Diffusion Policy. Please install it.")
        
        policy_config = PreTrainedConfig.from_pretrained(str(config.path))
        assert isinstance(policy_config, DiffusionConfig), \
            f"Expected DiffusionConfig, got {type(policy_config)}"
        
        # Load policy
        policy = DiffusionPolicy.from_pretrained(str(config.path), config=policy_config)
        self.policy = policy
        self.robot_type = config.robot_type
        self.device = torch.device(config.device)
        self.policy.to(self.device)
        self.policy.eval()
        
        # Load preprocessors and postprocessors
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_config,
            pretrained_path=str(config.path),
            preprocessor_overrides={
                "device_processor": {"device": config.device},
            },
        )
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.act_steps = config.act_steps
    
    def get_actions(self, observation: dict[str, Any]) -> np.ndarray:
        """
        Get actions from observation.
        
        This is the main interface function required for deployment.
        
        Args:
            observation: Dict with keys:
                - "states": Dict with "arm" or "left_arm"/"right_arm" (robot qpos)
                - "images": Dict with "front" and "wrist" or "left_wrist"/"right_wrist"
                - "task": Task prompt string
        
        Returns:
            Action array (joint positions) for the next step(s)
        """
        # Prepare observation for inference
        obs_infer = prepare_observation_for_inference(
            observation, self.device, observation.get("task", ""), self.robot_type
        )
        
        # Preprocess observation
        obs_processed = self.preprocessor(obs_infer)
        
        # Predict action chunk (future N steps)
        with torch.no_grad():
            action_chunk = self.policy.predict_action_chunk(obs_processed)
        
        # Postprocess actions
        action_chunk_processed = []
        for i in range(len(action_chunk)):
            action_chunk_processed.append(self.postprocessor(action_chunk[i]))
        
        # Return action(s)
        if self.act_steps is None:
            # Return first step only
            return action_chunk_processed[0].cpu().numpy() if isinstance(action_chunk_processed[0], torch.Tensor) else action_chunk_processed[0]
        else:
            # Return first N steps
            actions = []
            for i in range(min(self.act_steps, len(action_chunk_processed))):
                action = action_chunk_processed[i]
                actions.append(action.cpu().numpy() if isinstance(action, torch.Tensor) else action)
            return np.array(actions)
    
    def reset(self):
        """Reset policy state (for episode boundaries)."""
        print("Resetting LeRobotDiffusionPolicy")
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()


if __name__ == "__main__":
    config = tyro.cli(LeRobotDiffusionPolicyConfig)
    print(config)
    policy = LeRobotDiffusionPolicy(config)
    print("Diffusion Policy initialized.")
