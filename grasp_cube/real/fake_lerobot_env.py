import dataclasses
import tyro
import pathlib
import json
import gymnasium as gym
import time
import numpy as np
import torch
from typing import Any, SupportsFloat

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import busy_wait

@dataclasses.dataclass
class FakeLeRobotEnvConfig:
    dataset_path: pathlib.Path
    episode_index: int | None = None
    
class FakeLeRobotEnv(gym.Env):
    def __init__(self, config: FakeLeRobotEnvConfig):
        super().__init__()
        dataset = LeRobotDataset(
            repo_id="", # unused
            root=config.dataset_path,
            episodes=[config.episode_index] if config.episode_index is not None else None, 
        )
        self.dataset = dataset
        self.current_episode = 0
        self.current_idx = -1
        self.robot_type = "so101" if len(dataset[0]["observation.state"]) == 6 else "bi_so101"
        self.action_dim = 6 if self.robot_type == "so101" else 12
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "states": gym.spaces.Dict({
                "arm": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            }) if self.robot_type == "so101" else gym.spaces.Dict({
                "left_arm": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
                "right_arm": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            }),
            "images": gym.spaces.Dict({
                "front": gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
                "wrist": gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
            }) if self.robot_type == "so101" else gym.spaces.Dict({
                "front": gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
                "left_wrist": gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
                "right_wrist": gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
            }),
            "task": gym.spaces.Text(max_length=100, charset="1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ \n!@#$%^&*()-_=+[]{}|;:'\",.<>?/`~"),
        })
    
    def get_observation(self, idx: int) -> dict[str, Any]:
        if self.robot_type == "so101":
            states = dict(
                arm=self.dataset[idx]["observation.state"],
            )
            images = dict(
                front=self.dataset[idx]["observation.images.front"],
                wrist=self.dataset[idx]["observation.images.wrist"],
            )
        else:
            states = dict(
                left_arm=self.dataset[idx]["observation.state"][:6],
                right_arm=self.dataset[idx]["observation.state"][6:],
            )
            images = dict(
                front=self.dataset[idx]["observation.images.front"],
                left_wrist=self.dataset[idx]["observation.images.left_wrist"],
                right_wrist=self.dataset[idx]["observation.images.right_wrist"],
            )
        for key in images.keys():
            images[key] = np.array(images[key] * 255, dtype=np.uint8)
            images[key] = np.transpose(images[key], (1, 2, 0))  # CHW to HWC
        for key in states.keys():
            states[key] = np.array(states[key], dtype=np.float32)
        task = self.dataset[idx]["task"]
        return dict(
            states=states,
            images=images,
            task=task,
        )
        
    def teleop_step(self):
        print(f"Current episode: {self.current_episode}, idx: {self.current_idx} Fake teleop step called.")
        busy_wait(1)
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        self.current_idx = (i := (self.current_idx + 1) % len(self.dataset))
        self.current_episode = self.dataset[i]['episode_index']
        observation = self.get_observation(i)
        info = {}
        return observation, info
    
    def step(self, action: np.ndarray) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:    
        self.current_idx = (i := (self.current_idx + 1) % len(self.dataset))
        observation = self.get_observation(i)
        reward = 0.0
        terminated = bool(self.dataset[i]['episode_index'] != self.current_episode)
        truncated = False
        info = {'gt_action': self.dataset[i][ACTION], 'success': terminated}
        return observation, reward, terminated, truncated, info