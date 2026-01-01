import dataclasses
import tyro
import pathlib
import json
import gymnasium as gym
import time
import numpy as np
import torch
from typing import Any, SupportsFloat

from lerobot.robots import make_robot_from_config
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.processor.factory import make_default_processors
from lerobot.policies.utils import make_robot_action
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots import so101_follower, bi_so101_follower
from lerobot.teleoperators import so101_leader, bi_so101_leader
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import busy_wait

@dataclasses.dataclass
class LeRobotEnvConfig:
    robot: so101_follower.SO101FollowerConfig | bi_so101_follower.BiSO101FollowerConfig
    camera_config_path: pathlib.Path
    teleop: so101_leader.SO101LeaderConfig | bi_so101_leader.BiSO101LeaderConfig | None = None
    task: str = ""
    dry_run: bool = False
    episode_time_s: float = 600.0
    fps: int = 30
    
    def __post_init__(self):
        camera_config = json.loads(self.camera_config_path.read_text())
        camera_cls = {
            "opencv": OpenCVCameraConfig,
            "realsense": RealSenseCameraConfig,
        }    
        self.robot.cameras = {name: camera_cls[params["type"]](**{k: v for k, v in params.items() if k != "type"}) for name, params in camera_config.items()}

class LeRobotEnv(gym.Env):
    def __init__(self, config: LeRobotEnvConfig):
        self.robot = make_robot_from_config(config.robot)
        self.teleop = None if config.teleop is None else make_teleoperator_from_config(config.teleop)

        teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
        self.teleop_action_processor = teleop_action_processor
        self.robot_action_processor = robot_action_processor
        self.robot_observation_processor = robot_observation_processor
        self.robot.connect()
        if self.teleop is not None:
            self.teleop.connect()
        
        self.dry_run = config.dry_run
        self.timestamp = 0
        self.episode_time_s = config.episode_time_s
        self.fps = config.fps
        self.task = config.task
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "states": gym.spaces.Dict({
                "arm": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            }) if self.robot.robot_type == "so101_follower" else gym.spaces.Dict({
                "left_arm": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
                "right_arm": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            }),
            "images": gym.spaces.Dict({
                "front": gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
                "wrist": gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
            }) if self.robot.robot_type == "so101_follower" else gym.spaces.Dict({
                "front": gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
                "left_wrist": gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
                "right_wrist": gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
            }),
            "task": gym.spaces.Text(max_length=100, charset="1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ \n!@#$%^&*()-_=+[]{}|;:'\",.<>?/`~"),
        })
        
    @property
    def action_dim(self) -> int:
        return len(self.robot.action_features)
        
    def prepare_observation(self, obs_processed: dict[str, Any]) -> dict[str, Any]:
        obs = {"images": {}, "states": {}, "task": self.task}
        for key, value in obs_processed.items():
            if "pos" in key:
                if "left" in key:
                    if "left_arm" not in obs["states"]:
                        obs["states"]["left_arm"] = []
                    obs["states"]["left_arm"].append(value)
                elif "right" in key:
                    if "right_arm" not in obs["states"]:
                        obs["states"]["right_arm"] = []
                    obs["states"]["right_arm"].append(value)
                else:
                    if "arm" not in obs["states"]:
                        obs["states"]["arm"] = []
                    obs["states"]["arm"].append(value)
            else:
                obs["images"][key] = value
        for arm_key in obs["states"].keys():
            obs["states"][arm_key] = np.array(obs["states"][arm_key], dtype=np.float32).flatten()
        return obs
        
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        self.timestamp = 0
        self.start_episode_t = time.perf_counter()
        obs = self.robot.get_observation()
        obs_processed = self.robot_observation_processor(obs)
        obs_prepared = self.prepare_observation(obs_processed)
        return obs_prepared, {}
    
    def teleop_step(self) -> None:
        start_loop_t = time.perf_counter()
        if self.teleop is None:
            return
        obs = self.robot.get_observation()
        act = self.teleop.get_action()

        # Applies a pipeline to the raw teleop action, default is IdentityProcessor
        act_processed_teleop = self.teleop_action_processor((act, obs))
        action_values = act_processed_teleop
        robot_action_to_send = self.robot_action_processor((act_processed_teleop, obs))
        _sent_action = self.robot.send_action(robot_action_to_send)
        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / self.fps - dt_s)
        
    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        start_loop_t = time.perf_counter()
        robot = self.robot
        obs = robot.get_observation()
        obs_processed = self.robot_observation_processor(obs)
        if action is not None:
            action = torch.as_tensor(action)[None, ...]
            act_processed = make_robot_action(action, {ACTION: {"names": list(robot.action_features.keys())}})
        else:
            act_processed = None
        if act_processed is not None:
            robot_action_to_send = self.robot_action_processor((act_processed, obs))
            print(f"Sending action: {robot_action_to_send}")
            if not self.dry_run:
                _sent_action = robot.send_action(robot_action_to_send)
        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / self.fps - dt_s)

        self.timestamp = time.perf_counter() - self.start_episode_t
        done = self.timestamp >= self.episode_time_s
        reward = 0.0
        info = {
            "timestamp": self.timestamp,
            "loop_time_s": time.perf_counter() - start_loop_t,
        }
        obs_prepared = self.prepare_observation(obs_processed)
        return obs_prepared, reward, done, False, info
        
    def close(self):
        self.robot.disconnect()
        if self.teleop is not None:
            self.teleop.disconnect()
    
if __name__ == "__main__":
    config = tyro.cli(LeRobotEnvConfig)
    env = LeRobotEnv(config)
    print("Environment initialized.")
    for i in range(3):
        obs, info = env.reset()
        done = False
        while not done:
            action = None
            obs, reward, done, _, info = env.step(action)
            print(info)
    obs, info = env.reset()
    
"""
python ./lerobot_env.py --no-dry-run --camera-config-path configs/so101.json  robot:so101-follower-config --robot.id eai-robot-right-follower-arm --robot.port /dev/ttyACM0 teleop:so101-leader-config --teleop.port /dev/ttyACM1 --teleop.id eai-robot-right-leader-arm
python ./lerobot_env.py --no-dry-run --camera-config-path configs/bi_so101.json  robot:bi-so101-follower-config --robot.id eai-robot-bi-follower --robot.right-arm-port /dev/ttyACM0 --robot.left-arm-port /dev/ttyACM2 teleop:bi-so101-leader-config --teleop.right-arm-port /dev/ttyACM1 --teleop.left-arm-port /dev/ttyACM3 --teleop.id eai-robot-bi-leader
"""