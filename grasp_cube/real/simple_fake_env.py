"""
Simple fake environment that doesn't require actual LeRobot datasets.

This is useful for testing the server-client architecture without
needing to have the full LeRobot dataset structure in place.
"""

import dataclasses
import gymnasium as gym
import numpy as np
import time
import pathlib
from typing import Any, SupportsFloat


@dataclasses.dataclass
class SimpleFakeEnvConfig:
    """Configuration for SimpleFakeEnv"""
    task: str = "lift"  # "lift", "sort", or "stack"
    num_episodes: int = 10
    episode_length: int = 100


class SimpleFakeEnv(gym.Env):
    """
    A simple fake environment that generates random observations
    without requiring actual LeRobot datasets.
    
    Useful for testing server-client integration.
    """
    
    def __init__(self, config: SimpleFakeEnvConfig):
        super().__init__()
        self.config = config
        self.task = config.task
        
        # Create run_dir for compatibility with other environments
        self.run_dir = pathlib.Path("/tmp/simple_fake_env_runs")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine dimensions based on task
        if self.task in ["lift", "stack"]:
            self.state_dim = 6
            self.action_dim = 6
        elif self.task == "sort":
            self.state_dim = 12
            self.action_dim = 12
        else:
            raise ValueError(f"Unknown task: {self.task}")
        
        self.action_space = gym.spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.action_dim,), 
            dtype=np.float32
        )
        
        self.observation_space = gym.spaces.Dict({
            "states": gym.spaces.Dict({
                "arm": gym.spaces.Box(
                    low=-np.inf, 
                    high=np.inf, 
                    shape=(6,), 
                    dtype=np.float32
                ),
            }) if self.state_dim == 6 else gym.spaces.Dict({
                "left_arm": gym.spaces.Box(
                    low=-np.inf, 
                    high=np.inf, 
                    shape=(6,), 
                    dtype=np.float32
                ),
                "right_arm": gym.spaces.Box(
                    low=-np.inf, 
                    high=np.inf, 
                    shape=(6,), 
                    dtype=np.float32
                ),
            }),
            "images": gym.spaces.Dict({
                "front": gym.spaces.Box(
                    low=0, 
                    high=255, 
                    shape=(480, 640, 3), 
                    dtype=np.uint8
                ),
                "wrist": gym.spaces.Box(
                    low=0, 
                    high=255, 
                    shape=(480, 640, 3), 
                    dtype=np.uint8
                ),
            }) if self.state_dim == 6 else gym.spaces.Dict({
                "front": gym.spaces.Box(
                    low=0, 
                    high=255, 
                    shape=(480, 640, 3), 
                    dtype=np.uint8
                ),
                "left_wrist": gym.spaces.Box(
                    low=0, 
                    high=255, 
                    shape=(480, 640, 3), 
                    dtype=np.uint8
                ),
                "right_wrist": gym.spaces.Box(
                    low=0, 
                    high=255, 
                    shape=(480, 640, 3), 
                    dtype=np.uint8
                ),
            }),
        })
        
        # Episode tracking
        self.current_episode = 0
        self.current_step = 0
        self.episode_length = config.episode_length
    
    def _get_observation(self) -> dict[str, Any]:
        """Generate a random observation."""
        if self.state_dim == 6:
            states = {
                "arm": np.random.randn(6).astype(np.float32),
            }
            images = {
                "front": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
                "wrist": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
            }
        else:  # state_dim == 12
            states = {
                "left_arm": np.random.randn(6).astype(np.float32),
                "right_arm": np.random.randn(6).astype(np.float32),
            }
            images = {
                "front": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
                "left_wrist": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
                "right_wrist": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
            }
        
        return {
            "states": states,
            "images": images,
        }
    
    def reset(
        self, 
        *, 
        seed: int | None = None, 
        options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        self.current_step = 0
        self.current_episode += 1
        
        observation = self._get_observation()
        info = {"episode": self.current_episode}
        
        return observation, info
    
    def step(self, action: np.ndarray) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step of the environment."""
        self.current_step += 1
        
        observation = self._get_observation()
        reward = 0.0
        terminated = self.current_step >= self.episode_length
        truncated = False
        
        # Generate a ground truth action for comparison
        gt_action = np.random.randn(self.action_dim).astype(np.float32)
        
        info = {
            "step": self.current_step,
            "episode": self.current_episode,
            "success": terminated,
            "action": gt_action,  # Add action to info for gt_action comparison
        }
        
        return observation, reward, terminated, truncated, info
    
    def teleop_step(self):
        """Simulate a teleoperation step."""
        print(
            f"Episode: {self.current_episode}, "
            f"Step: {self.current_step} - Fake teleop step called."
        )
        time.sleep(1)
