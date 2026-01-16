from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.building import actors
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from grasp_cube.envs.tasks.pick_cube_so101 import PickCubeSO101Env
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

@register_env("LiftCubeSO101-v1", max_episode_steps=50)
class LiftCubeSO101Env(PickCubeSO101Env):
    """
    Task: Lift red cube.
    Success: Cube height > 5.0 cm.
    Uses single-arm setup (right arm only).
    Red cube spawns in the right arm's front area (right region, y < -0.1).
    No green goal sphere.
    """
    
    # Task prompt for Diffusion Policy
    TASK_PROMPT = "Pick up the red cube and lift it."
    
    def __init__(self, *args, robot_uids=("so101",), robot_init_qpos_noise=0.02, **kwargs):
        """Initialize with single arm (right arm only)."""
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
    
    @property
    def _default_sensor_configs(self):
        """Configure cameras for LeRobot Dataset format.
        - Front camera: 480×640, third-person view
        - Wrist camera: 480×640, attached to right arm's camera_link (defined in SO101 agent class)
        """
        configs = []
        
        # Front camera: third-person view, 480×640 resolution
        front_pose = sapien_utils.look_at(
            eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos
        )
        configs.append(CameraConfig("front", front_pose, 640, 480, np.deg2rad(50), 0.01, 100))
        
        # Note: Wrist camera is defined in SO101 agent's _sensor_configs property
        # It will be automatically attached to camera_link and named "wrist_camera"
        # We'll map it to "right_wrist" in data collection
        
        return configs
    
    
    def _load_agent(self, options: dict):
        """Load single arm (right arm only) at y = -0.15."""
        # Call BaseEnv._load_agent directly with initial_agent_poses
        BaseEnv._load_agent(self, options, [
            sapien.Pose(p=[-0.665, -0.15, 0])  # Right arm position
        ])
    
    def initialize_agent(self, env_idx: torch.Tensor):
        """Initialize single arm (right arm only) qpos for each episode."""
        b = len(env_idx)
        qpos = np.array([0, 0, 0, np.pi / 2, 0, 0])
        
        # Add noise for randomization
        qpos_right = qpos + self._episode_rng.normal(0, self.robot_init_qpos_noise, (b, len(qpos)))
        
        # Reset single agent - check if it's MultiAgent or single agent
        if hasattr(self.agent, "agents"):
            # MultiAgent wrapper (shouldn't happen with single robot_uids, but handle it)
            self.agent.agents[0].reset(qpos_right)
            self.agent.agents[0].robot.set_pose(sapien.Pose([-0.665, -0.15, 0]))
        else:
            # Direct agent object (single robot)
            self.agent.reset(qpos_right)
            self.agent.robot.set_pose(sapien.Pose([-0.665, -0.15, 0]))
    
    def _load_scene(self, options: dict):
        """Load scene without goal_site (green sphere)."""
        # Load table scene and boundary lines
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self._build_boundary_lines()
        
        # Create cube spawn visualization
        spawn_center = [self.cube_spawn_center[0], self.cube_spawn_center[1], self.cube_half_size]
        spawn_half_size = [self.cube_spawn_half_size, self.cube_spawn_half_size, 1e-4]
        self.cube_spawn_vis = actors.build_box(
            self.scene,
            half_sizes=spawn_half_size,
            color=[0, 0, 1, 0.2],
            name="cube_spawn_region",
            add_collision=False,
            initial_pose=sapien.Pose(p=spawn_center),
        )
        
        # Create red cube (no goal_site for lift task)
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        
        # Do NOT create goal_site (green sphere) for lift task
        self.goal_site = None
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize episode with red cube in right arm's front area."""
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.initialize_agent(env_idx)
            
            # Red cube spawns in right arm's front area (right region: y < -0.1)
            # Right arm is at y = -0.15, so right region is y < -0.1
            # Spawn in the right compartment: y range approximately [-0.29, -0.11]
            xyz = torch.zeros((b, 3))
            # X position: within the spawn area
            xyz[:, 0] = (
                torch.rand((b,)) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            ) + self.cube_spawn_center[0]
            # Y position: in right region (y < -0.1), specifically around -0.2 (right arm front)
            xyz[:, 1] = (
                torch.rand((b,)) * 0.08  # Randomize within right region
                - 0.24  # Center around -0.2 (right arm front area)
            )
            xyz[:, 2] = self.cube_half_size
            
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=self.lock_z)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))
    
    def evaluate(self):
        # Use single arm for lift task
        if hasattr(self.agent, "agents"):
            agent = self.agent.agents[0]
        else:
            agent = self.agent
        is_grasped = agent.is_grasping(self.cube)
        # Success if height > 0.05m (5cm)
        is_lifted = self.cube.pose.p[:, 2] > 0.05
        
        return {
            "success": is_lifted,
            "is_lifted": is_lifted,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # Use single arm tcp pose
        if hasattr(self.agent, "agents"):
            agent = self.agent.agents[0]
        else:
            agent = self.agent
        tcp_pose = agent.tcp_pose.p
        
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - tcp_pose, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        # Reward for height
        height_reward = torch.clamp(self.cube.pose.p[:, 2], max=0.1) * 10
        reward += height_reward * is_grasped

        reward[info["success"]] = 5
        return reward
    
    def _get_obs_extra(self, info: Dict):
        """Override to use single arm for observations. No goal_pos since goal_site is removed."""
        if hasattr(self.agent, "agents"):
            agent = self.agent.agents[0]
        else:
            agent = self.agent
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=agent.tcp_pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p - agent.tcp_pose.p,
            )
        return obs
    
